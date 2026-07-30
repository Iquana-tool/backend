"""Cross-image exemplar retrieval: pick the exemplars most relevant to a target.

Given a target image (and optionally the concept being annotated), a strategy ranks the
dataset's existing annotations (contours) by how good an exemplar each would make for
cross-image concept transfer -- the shortlist then feeds the SAM3-concat handler.

Strategies live in a registry, mirroring the annotation-queue sort-strategy registry: a new
ranking (e.g. a learned hybrid) can be added without touching the request/response contract
-- register it and it appears in the picker. A strategy may be a *placeholder*
(``available=False``): shown in the UI, not yet runnable.

Each strategy declares the embedding ``required_kinds`` it consumes (``image_cls`` for
whole-image scene similarity, ``region_mean`` for object-level similarity). That declaration
is the coupling to the Embed capability: enabling a strategy tells the lifecycle layer which
kinds must be precomputed. All ranking runs off the pgvector store (see
``app.database.embeddings``); nothing is embedded on this path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.database.contours import Contours
from app.database.embeddings import (
    SUBJECT_CONTOUR,
    SUBJECT_IMAGE,
    get_embedding_vector,
    search_similar,
)
from app.database.images import Images
from app.database.masks import Masks
from app.schemas.exemplar_retrieval import RetrievalStrategyOption

IMAGE_CLS = "image_cls"
REGION_MEAN = "region_mean"


@dataclass(frozen=True)
class RetrievalQuery:
    """What to retrieve exemplars for.

    ``dataset_id`` scopes the bank. ``target_image_id`` is the image being annotated (the
    scene-similarity query for ``global_scene``). ``concept_label_id`` restricts exemplars to
    one label. ``query_vector`` is an explicit region embedding for object-level ranking
    (``concept_region``). ``top_k`` caps the shortlist. Not every strategy uses every field --
    each validates the ones it needs.
    """

    dataset_id: int
    target_image_id: int | None = None
    concept_label_id: int | None = None
    query_vector: Sequence[float] | None = None
    top_k: int = 5


@dataclass(frozen=True)
class ExemplarMatch:
    """One ranked exemplar: the contour, its source image, and a similarity in ``[0, 1]``."""

    contour_id: int
    image_id: int
    score: float  # 1 - cosine distance; higher is more similar.


#: A strategy's ranking function: (session, query, model_id) -> ranked exemplars.
RetrieveFn = Callable[[Session, RetrievalQuery, str], "list[ExemplarMatch]"]


@dataclass(frozen=True)
class RetrievalStrategy:
    key: str
    label: str
    description: str
    required_kinds: tuple[str, ...]
    available: bool
    retrieve: RetrieveFn | None  # None for placeholder strategies.


RETRIEVAL_STRATEGIES: dict[str, RetrievalStrategy] = {}


def register_retrieval_strategy(
    key: str,
    label: str,
    description: str,
    required_kinds: Sequence[str],
    available: bool = True,
):
    """Register an exemplar-retrieval strategy. Registered keys appear in the picker."""

    def decorator(fn: RetrieveFn) -> RetrieveFn:
        RETRIEVAL_STRATEGIES[key] = RetrievalStrategy(
            key=key, label=label, description=description,
            required_kinds=tuple(required_kinds), available=available, retrieve=fn,
        )
        return fn

    return decorator


# --------------------------------------------------------------------------- #
# Shared DB helpers
# --------------------------------------------------------------------------- #
def _exemplar_contours_in_image(session: Session, image_id: int,
                                concept_label_id: int | None) -> list[int]:
    """Non-temporary contour ids in one image, optionally filtered to one concept label."""
    q = (
        session.query(Contours.id)
        .join(Masks, Masks.id == Contours.mask_id)
        .filter(Masks.image_id == image_id, Contours.temporary.is_(False))
    )
    if concept_label_id is not None:
        q = q.filter(Contours.label_id == concept_label_id)
    return [row[0] for row in q.all()]


def _concept_contour_ids(session: Session, dataset_id: int,
                         concept_label_id: int) -> set[int]:
    """Non-temporary contour ids of one concept across a dataset (the region search's candidate set)."""
    rows = (
        session.query(Contours.id)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id,
                Contours.temporary.is_(False),
                Contours.label_id == concept_label_id)
        .all()
    )
    return {row[0] for row in rows}


def _contour_image_map(session: Session, contour_ids: Sequence[int]) -> dict[int, int]:
    """Map each contour id to its source image id (via its mask)."""
    if not contour_ids:
        return {}
    rows = (
        session.query(Contours.id, Masks.image_id)
        .join(Masks, Masks.id == Contours.mask_id)
        .filter(Contours.id.in_(list(contour_ids)))
        .all()
    )
    return {cid: image_id for cid, image_id in rows}


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
@register_retrieval_strategy(
    "global_scene",
    "Global scene match",
    "Rank exemplars by how visually similar their source image is to the target image "
    "(whole-image DINOv3 CLS cosine). Best cold-start choice: it needs no annotation on the "
    "target yet.",
    required_kinds=(IMAGE_CLS,),
)
def _global_scene(session: Session, query: RetrievalQuery, model_id: str) -> list[ExemplarMatch]:
    if query.target_image_id is None:
        raise ValueError("global_scene retrieval requires target_image_id.")
    q_vec = get_embedding_vector(
        session, image_id=query.target_image_id, kind=IMAGE_CLS, model_id=model_id
    )
    if q_vec is None:
        raise ValueError(
            f"No '{IMAGE_CLS}' embedding for image {query.target_image_id} (model {model_id!r}); "
            "embed the image before retrieving."
        )

    # Rank the dataset's other images by scene similarity, then walk them in order, taking
    # their exemplar contours until the shortlist is full. Contours from one image share that
    # image's score (scene similarity is per-image), so their relative order within an image
    # is arbitrary.
    image_hits = search_similar(
        session, q_vec, subject_type=SUBJECT_IMAGE, kind=IMAGE_CLS, model_id=model_id,
        dataset_id=query.dataset_id, exclude_ids=[query.target_image_id], top_k=query.top_k,
    )
    matches: list[ExemplarMatch] = []
    for hit in image_hits:
        for contour_id in _exemplar_contours_in_image(session, hit.subject_id, query.concept_label_id):
            matches.append(ExemplarMatch(contour_id=contour_id, image_id=hit.subject_id,
                                         score=1.0 - hit.distance))
            if len(matches) >= query.top_k:
                return matches
    return matches


@register_retrieval_strategy(
    "concept_region",
    "Concept region match",
    "Rank exemplar objects by masked-region feature similarity to a query region "
    "(DINOv3 region-mean cosine). Use when a candidate region for the concept already exists "
    "(e.g. from a text/geometry proposal or a just-annotated object).",
    required_kinds=(REGION_MEAN,),
)
def _concept_region(session: Session, query: RetrievalQuery, model_id: str) -> list[ExemplarMatch]:
    if query.query_vector is None:
        raise ValueError("concept_region retrieval requires a query_vector (a region embedding).")

    restrict: set[int] | None = None
    if query.concept_label_id is not None:
        restrict = _concept_contour_ids(session, query.dataset_id, query.concept_label_id)
        if not restrict:
            return []

    hits = search_similar(
        session, query.query_vector, subject_type=SUBJECT_CONTOUR, kind=REGION_MEAN,
        model_id=model_id, dataset_id=query.dataset_id, restrict_ids=restrict, top_k=query.top_k,
    )
    image_map = _contour_image_map(session, [h.subject_id for h in hits])
    return [
        ExemplarMatch(contour_id=h.subject_id, image_id=image_map[h.subject_id],
                      score=1.0 - h.distance)
        for h in hits
    ]


@register_retrieval_strategy(
    "hybrid",
    "Hybrid (scene + region)",
    "Coarse scene pre-filter then object-level re-ranking. Coming soon.",
    required_kinds=(IMAGE_CLS, REGION_MEAN),
    available=False,
)
def _hybrid(session: Session, query: RetrievalQuery, model_id: str):  # pragma: no cover - placeholder
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="The hybrid retrieval strategy is not available yet.",
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def strategy_options() -> list[RetrievalStrategyOption]:
    return [
        RetrievalStrategyOption(
            key=s.key, label=s.label, description=s.description,
            available=s.available, required_kinds=list(s.required_kinds),
        )
        for s in RETRIEVAL_STRATEGIES.values()
    ]


def retrieve_exemplars(
    session: Session, strategy_key: str, query: RetrievalQuery, *, model_id: str
) -> list[ExemplarMatch]:
    """Run a registered strategy and return its ranked exemplar shortlist.

    Raises 400 for an unknown or not-yet-available strategy (mirroring the queue builder).
    """
    strategy = RETRIEVAL_STRATEGIES.get(strategy_key)
    if strategy is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown retrieval strategy {strategy_key!r}. "
                   f"Known: {sorted(RETRIEVAL_STRATEGIES)}.",
        )
    if not strategy.available or strategy.retrieve is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The {strategy_key!r} retrieval strategy is not available yet.",
        )
    return strategy.retrieve(session, query, model_id)

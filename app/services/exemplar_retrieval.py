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
    Embeddings,
    dataset_scoped_ids,
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
    (``concept_region``). ``top_k`` caps the shortlist, counted in *contours* -- several of the
    best may share a source image, so a caller that cares about images (the concat handler
    pastes one tile per exemplar image) asks for a wider shortlist and thins it itself. Not
    every strategy uses every field -- each validates the ones it needs.
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
    "concept_annotations",
    "Existing annotations of the label",
    "Use the label's existing annotations elsewhere in the dataset as exemplars, reviewed "
    "ones first, then the most recent. Needs no embeddings -- annotate a few images by hand, "
    "then let an in-context model carry the concept across the rest.",
    required_kinds=(),
)
def _concept_annotations(session: Session, query: RetrievalQuery, model_id: str) -> list[ExemplarMatch]:
    """Rank the concept's own annotations, no embedding store involved.

    The other strategies rank by *visual* similarity, which buys precision but costs a
    populated pgvector store. This one asks a blunter question -- "what has a human already
    marked as this concept?" -- which needs nothing but the annotations themselves. That makes
    it the only strategy usable on a fresh dataset, and the one a batch run over a dataset
    actually wants: the exemplars are the hand-annotated seed images.

    Reviewed contours rank above unreviewed ones (an approved object is a better example of
    the concept than an unverified one), then newest first, on the assumption that later
    annotations reflect a settled understanding of the label. ``model_id`` is unused; it is
    part of the strategy signature because the embedding-based strategies need it.
    """
    if query.concept_label_id is None:
        raise ValueError("concept_annotations retrieval requires a concept_label_id.")

    rows = (
        session.query(Contours.id, Images.id, Contours.reviewed_by.any())
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == query.dataset_id,
                Contours.temporary.is_(False),
                Contours.label_id == query.concept_label_id)
        # An object cannot be its own exemplar: the target image is what is being annotated,
        # and feeding its existing objects back would bias the model toward what is already
        # there instead of transferring the concept from elsewhere.
        .filter(Images.id != query.target_image_id)
        .order_by(Contours.reviewed_by.any().desc(), Contours.created_at.desc())
        .limit(query.top_k)
        .all()
    )
    # Scores are a rank-derived stand-in, not a similarity: this strategy has no metric space.
    # They exist so the response shape matches the embedding-based strategies, and so the best
    # exemplar still sorts first when the caller preserves order.
    return [
        ExemplarMatch(contour_id=contour_id, image_id=image_id,
                      score=1.0 - (rank / max(len(rows), 1)))
        for rank, (contour_id, image_id, _reviewed) in enumerate(rows)
    ]


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
def _kinds_present(session: Session, dataset_id: int) -> set[str]:
    """Embedding kinds that actually exist for a dataset's images and contours.

    One query per subject type, and only the *kind* column -- this is a "does anything at all
    exist" check, not a search, so it must stay cheap enough to run on every page load.
    """
    present: set[str] = set()
    for subject_type, column in ((SUBJECT_IMAGE, Embeddings.image_id),
                                 (SUBJECT_CONTOUR, Embeddings.contour_id)):
        ids = dataset_scoped_ids(session, subject_type, dataset_id)
        if not ids:
            continue
        rows = (
            session.query(Embeddings.kind)
            .filter(column.in_(ids))
            .distinct()
            .all()
        )
        present.update(row[0] for row in rows)
    return present


def strategy_options(
    session: Session | None = None, dataset_id: int | None = None
) -> list[RetrievalStrategyOption]:
    """The selectable strategies, optionally narrowed to what can run on one dataset.

    Without a session this reports what is *implemented*, which is all a generic listing can
    say. Given a dataset it also reports what is *usable*: a strategy that ranks by visual
    similarity is dead weight until somebody has embedded that dataset, and offering it
    anyway means the user picks it and every image fails. The two notions are kept in one
    flag on purpose -- a caller that has to distinguish "not written yet" from "not usable
    here" reads ``unavailable_reason``, and everything else just filters on ``available``.
    """
    present = _kinds_present(session, dataset_id) if session is not None and dataset_id else None

    options: list[RetrievalStrategyOption] = []
    for strategy in RETRIEVAL_STRATEGIES.values():
        available, reason = strategy.available, None
        if not available:
            reason = "Not implemented yet."
        elif present is not None:
            missing = [kind for kind in strategy.required_kinds if kind not in present]
            if missing:
                available = False
                reason = (
                    f"Needs {', '.join(missing)} embeddings, which this dataset does not have "
                    f"yet. Run the embedding backfill to enable it."
                )
        options.append(RetrievalStrategyOption(
            key=strategy.key, label=strategy.label, description=strategy.description,
            available=available, required_kinds=list(strategy.required_kinds),
            unavailable_reason=reason,
        ))
    return options


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

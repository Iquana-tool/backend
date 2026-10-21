"""Building and persisting annotation queues: the order an annotator works images.

Mirrors ``app.services.review_queue`` but with two deliberate differences:

* **Whole-image ordering, not instance scoring.** A strategy orders the dataset's
  images (a scan stack counts as one unit whose frames stay together) — there is
  nothing to score per contour.
* **Persisted, not a snapshot.** The built order is saved per (dataset, user) in
  ``annotation_queues`` so re-entering resumes it; rebuilding overwrites the row.

Orderings live in a registry: register a strategy and it appears in the builder. A
strategy receives a ``QueueContext`` (so it may read the database) and returns the
ordered image ids plus why each sits where it does. The active-learning orderings --
Diversity, Density (image vectors only) and Uncertainty (a trained model's hidden
predictions) -- read data that ``app.services.queue_prep`` fills in beforehand; anything
still missing is placed after the ranked images rather than blocking the build.
Confidence ordering lives in the review queue instead: it is about checking predictions
people can see, not about choosing what to label. A strategy may be
registered as a *placeholder* (``available=False``) — it shows in the UI but cannot be
built yet.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.annotation_queues import AnnotationQueues
from app.database.contours import Contours
from app.database.embeddings import Embeddings
from app.database.image_scores import ImageScores
from app.database.images import Images
from app.database.masks import Masks
from app.schemas.annotation_queue import (
    AnnotationQueueRead,
    AnnotationQueueStrategyOption,
    AnnotationQueueSummary,
)
from app.services import al_ranking
from config import EMBEDDING_MODEL_ID


@dataclass(frozen=True)
class QueueContext:
    """Everything an ordering may read when it builds a queue."""

    db: Session
    dataset_id: int
    username: str
    #: Every image of the dataset in upload order, as (id, stack id, frame index).
    images: list[al_ranking.ImageRef]
    model_key: str | None = None


@dataclass
class QueueOrder:
    image_ids: list[int]
    #: {"<image id>": {"reason": str, "score": float | None}}; empty for plain orderings.
    details: dict[str, dict]


#: An ordering strategy: context -> ordered image ids plus why each sits where it does.
#: Runs once at build time.
OrderFn = Callable[[QueueContext], QueueOrder]


@dataclass(frozen=True)
class QueueStrategy:
    key: str
    label: str
    description: str
    available: bool
    order: OrderFn | None  # None for placeholder strategies.
    #: Needs a trained model's scores (the model is chosen when building).
    needs_model: bool = False
    #: Needs a DINOv3 image vector for every image.
    needs_embeddings: bool = False
    #: Needs a model score for every unfinished image.
    needs_scores: bool = False


SORT_STRATEGIES: dict[str, QueueStrategy] = {}


def register_strategy(key: str, label: str, description: str, available: bool = True, *,
                      needs_model: bool = False, needs_embeddings: bool = False,
                      needs_scores: bool = False):
    """Register a queue ordering. Registered keys appear in the builder."""

    def decorator(fn: OrderFn) -> OrderFn:
        SORT_STRATEGIES[key] = QueueStrategy(
            key=key, label=label, description=description, available=available, order=fn,
            needs_model=needs_model, needs_embeddings=needs_embeddings,
            needs_scores=needs_scores,
        )
        return fn

    return decorator


def get_strategy(strategy_key: str) -> QueueStrategy:
    """The registered, buildable strategy for a key, or a 400 explaining why not."""
    strategy = SORT_STRATEGIES.get(strategy_key)
    if strategy is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown ordering '{strategy_key}'. "
                   f"Available: {sorted(SORT_STRATEGIES)}.",
        )
    if not strategy.available or strategy.order is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The '{strategy_key}' ordering is not available yet.",
        )
    return strategy


@register_strategy(
    "as_uploaded",
    "As uploaded",
    "Annotate images in the order they were added to the dataset.",
)
def _order_as_uploaded(ctx: QueueContext) -> QueueOrder:
    return QueueOrder(image_ids=[image_id for image_id, _, _ in ctx.images], details={})


@register_strategy(
    "random",
    "Randomized order",
    "Shuffle the images. The order is fixed once built, so it stays stable across sessions.",
)
def _order_random(ctx: QueueContext) -> QueueOrder:
    shuffled = [image_id for image_id, _, _ in ctx.images]
    random.shuffle(shuffled)
    return QueueOrder(image_ids=shuffled, details={})


@register_strategy(
    "diversity",
    "Diversity",
    "Images that look least like anything already annotated come first.",
    needs_embeddings=True,
)
def _order_diversity(ctx: QueueContext) -> QueueOrder:
    return _ranked_order(ctx, "diversity")


@register_strategy(
    "density",
    "Density",
    "Typical images, ones that look like many others, come first.",
    needs_embeddings=True,
)
def _order_density(ctx: QueueContext) -> QueueOrder:
    return _ranked_order(ctx, "density")


@register_strategy(
    "uncertainty",
    "Uncertainty",
    "Images where the chosen model is most torn about an object come first. "
    "These teach it the most.",
    needs_model=True, needs_scores=True,
)
def _order_uncertainty(ctx: QueueContext) -> QueueOrder:
    return _ranked_order(ctx, "uncertainty")


def strategy_options() -> list[AnnotationQueueStrategyOption]:
    return [
        AnnotationQueueStrategyOption(
            key=s.key, label=s.label, description=s.description, available=s.available,
            needs_model=s.needs_model,
        )
        for s in SORT_STRATEGIES.values()
    ]


def scoring_models_for_dataset(models: list[dict], dataset_id: int) -> list[dict]:
    """Instance-segmentation models trained on this dataset, for the queue's model picker.

    ``models`` is the ``result`` list of ``list_available_models("instance-segmentation")``.
    Training stamps ``trained_on_dataset_id`` (and the older ``dataset_id``) on the model's
    tags; a base model carries neither and cannot score a dataset it has never seen.
    """
    wanted = str(dataset_id)
    picked: list[dict] = []
    for model in models:
        tags = model.get("tags") or {}
        if str(tags.get("trained_on_dataset_id") or tags.get("dataset_id") or "") != wanted:
            continue
        key = model.get("registry_key")
        picked.append({"registry_key": key, "name": model.get("name") or key})
    return picked


# -- Active-learning ranking --------------------------------------------------

def image_states(db: Session, dataset_id: int) -> dict[int, str]:
    """``annotate_status`` per image that has a mask; images without one are not started.

    Computed in a subquery for the reason given in ``_status_counts``.
    """
    status_sq = (
        db.query(Masks.image_id.label("image_id"), Masks.annotate_status.label("status"))
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
        .subquery()
    )
    return dict(db.query(status_sq.c.image_id, status_sq.c.status).all())


def _image_authors(db: Session, dataset_id: int) -> dict[int, set[str]]:
    """Who has drawn on each image (non-temporary objects with a known author)."""
    rows = (
        db.query(Masks.image_id, Contours.author_username)
        .join(Contours, Contours.mask_id == Masks.id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id, Contours.temporary.is_(False),
                Contours.author_username.isnot(None))
        .distinct()
        .all()
    )
    authors: dict[int, set[str]] = {}
    for image_id, author in rows:
        authors.setdefault(image_id, set()).add(author)
    return authors


def _unit_state(unit: al_ranking.Unit, states: dict[int, str],
                authors: dict[int, set[str]], username: str) -> str:
    """"finished", "others" (someone else has started it) or "open"."""
    frame_states = [states.get(image_id, "not_started") for image_id in unit.image_ids]
    if all(state == "finished" for state in frame_states):
        return "finished"
    started = [i for i, s in zip(unit.image_ids, frame_states) if s == "in_progress"]
    unit_authors = set().union(*(authors.get(i, set()) for i in started))
    if unit_authors and username not in unit_authors:
        return "others"
    return "open"


def _image_vectors(db: Session, image_ids: list[int]) -> dict[int, np.ndarray]:
    rows = (db.query(Embeddings.image_id, Embeddings.vector)
            .filter(Embeddings.kind == "image_cls",
                    Embeddings.model_id == EMBEDDING_MODEL_ID,
                    Embeddings.image_id.in_(image_ids))
            .all())
    return {image_id: np.asarray(vector, dtype=float) for image_id, vector in rows}


def _image_confidences(db: Session, image_ids: list[int],
                       model_key: str) -> dict[int, list[float]]:
    """Every scored image's detection confidences for one model."""
    rows = (db.query(ImageScores.image_id, ImageScores.confidences)
            .filter(ImageScores.model_key == model_key,
                    ImageScores.confidences.isnot(None),
                    ImageScores.image_id.in_(image_ids))
            .all())
    return {image_id: list(confidences) for image_id, confidences in rows}


def _ranked_order(ctx: QueueContext, mode: str) -> QueueOrder:
    """Shared body of the three active-learning orderings.

    Ranked open units first, then units someone else has started, then finished units.
    Units without the data their ordering needs go after the ranked ones.
    """
    units = al_ranking.build_units(ctx.images)
    all_ids = [image_id for image_id, _, _ in ctx.images]
    states = image_states(ctx.db, ctx.dataset_id)
    authors = _image_authors(ctx.db, ctx.dataset_id)
    unit_states = {u.key: _unit_state(u, states, authors, ctx.username) for u in units}
    open_units = [u for u in units if unit_states[u.key] == "open"]
    scores: dict[int, float] = {}

    if mode == "uncertainty":
        confidences = (_image_confidences(ctx.db, all_ids, ctx.model_key)
                       if ctx.model_key else {})
        keyed = {u.key: al_ranking.unit_uncertainty(u, confidences) for u in open_units}
        ranked = al_ranking.order_by_score(keyed)
        reasons = {k: ("unsure" if keyed[k] is not None else "not_scored") for k in ranked}
        scores = {k: v[0] for k, v in keyed.items() if v is not None}
    else:
        vectors = _image_vectors(ctx.db, all_ids)
        unit_vectors = {u.key: al_ranking.unit_vector(u, vectors) for u in units}
        with_vectors = {u.key: unit_vectors[u.key] for u in open_units
                        if unit_vectors[u.key] is not None}
        if mode == "diversity":
            seeds = [unit_vectors[u.key] for u in units
                     if unit_states[u.key] == "finished" and unit_vectors[u.key] is not None]
            ranked = al_ranking.k_center_order(with_vectors, seeds)
            reason = "varied"
        else:
            scores = al_ranking.density_scores(with_vectors)
            ranked = al_ranking.order_by_score(scores)
            reason = "typical"
        reasons = {key: reason for key in ranked}
        missing = [u.key for u in open_units if u.key not in with_vectors]
        reasons.update({key: "no_vector" for key in missing})
        ranked += missing

    ordered = [(key, reasons[key]) for key in ranked]
    ordered += [(u.key, "started_by_other") for u in units if unit_states[u.key] == "others"]
    ordered += [(u.key, "finished") for u in units if unit_states[u.key] == "finished"]

    by_key = {u.key: u for u in units}
    image_ids: list[int] = []
    details: dict[str, dict] = {}
    for key, reason in ordered:
        score = scores.get(key)
        for image_id in by_key[key].image_ids:
            image_ids.append(image_id)
            details[str(image_id)] = {"reason": reason,
                                      "score": round(score, 4) if score is not None else None}
    return QueueOrder(image_ids=image_ids, details=details)


# -- Persistence ---------------------------------------------------------------

def _uploaded_images(dataset_id: int, db: Session) -> list[al_ranking.ImageRef]:
    """The dataset's images in upload order (ascending id), with their stack position."""
    rows = (
        db.query(Images.id, Images.stack_id, Images.frame_index)
        .filter(Images.dataset_id == dataset_id)
        .order_by(Images.id.asc())
        .all()
    )
    return [(image_id, stack_id, frame_index) for image_id, stack_id, frame_index in rows]


def _status_counts(dataset_id: int, db: Session) -> dict[str, int]:
    """Per-state image counts for the dataset's Annotate phase.

    This card is about annotating, so it asks ``Masks.annotate_status`` rather than
    the image's combined status: an image whose annotation is done but which was
    never calibrated should not show up here as unfinished work for the annotator.

    ``annotate_status`` is a CASE expression built from correlated subqueries over
    ``masks.id``. Grouping by it directly makes SQLAlchemy render that CASE twice --
    once in SELECT, once in GROUP BY -- with *different* bound parameters each time, so
    PostgreSQL no longer recognises the two as the same expression and rejects
    ``masks.id`` with "subquery uses ungrouped column" (SQLite tolerated the mismatch).
    Computing the status once in a subquery and grouping by the resulting plain column
    avoids the double render.
    """
    status_sq = (
        db.query(Masks.annotate_status.label("status"))
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
        .subquery()
    )
    rows = (
        db.query(status_sq.c.status, func.count())
        .group_by(status_sq.c.status)
        .all()
    )
    return {mask_status: count for mask_status, count in rows}


def get_saved_queue(dataset_id: int, username: str, db: Session) -> AnnotationQueueRead | None:
    """The caller's saved queue for the dataset, or None if they have not built one."""
    row = (
        db.query(AnnotationQueues)
        .filter(AnnotationQueues.dataset_id == dataset_id,
                AnnotationQueues.username == username)
        .one_or_none()
    )
    if row is None:
        return None
    image_ids = list(row.image_order or [])
    return AnnotationQueueRead(
        strategy=row.strategy,
        model_key=row.model_key,
        image_ids=image_ids,
        total=len(image_ids),
        details=dict(row.details or {}),
        updated_at=row.updated_at,
    )


def build_and_save_queue(
    dataset_id: int, username: str, strategy_key: str, db: Session,
    model_key: str | None = None,
) -> AnnotationQueueRead:
    """Build the ordered image list for the strategy and persist it (upsert).

    Never waits for missing data: images without a vector or score are placed after the
    ranked ones (see the reasons in ``details``). ``queue_prep.start_prep`` fills that data.
    """
    strategy = get_strategy(strategy_key)
    if strategy.needs_model and not model_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The '{strategy_key}' ordering needs a trained model. Choose one first.",
        )
    ctx = QueueContext(db=db, dataset_id=dataset_id, username=username,
                       images=_uploaded_images(dataset_id, db),
                       model_key=model_key if strategy.needs_model else None)
    order = strategy.order(ctx)

    row = (
        db.query(AnnotationQueues)
        .filter(AnnotationQueues.dataset_id == dataset_id,
                AnnotationQueues.username == username)
        .one_or_none()
    )
    if row is None:
        row = AnnotationQueues(dataset_id=dataset_id, username=username)
        db.add(row)
    row.strategy = strategy_key
    row.model_key = ctx.model_key
    row.image_order = order.image_ids
    row.details = order.details or None
    db.commit()
    return get_saved_queue(dataset_id, username, db)


def summarize(dataset_id: int, username: str, db: Session) -> AnnotationQueueSummary:
    """Counts behind the Annotation card's subcaption plus the saved-queue state."""
    counts = _status_counts(dataset_id, db)
    total = db.query(func.count(Images.id)).filter(Images.dataset_id == dataset_id).scalar() or 0
    saved = get_saved_queue(dataset_id, username, db)
    in_progress = counts.get("in_progress", 0)
    finished = counts.get("finished", 0)
    return AnnotationQueueSummary(
        # Derived rather than read off the tally: an image that has never been
        # opened has no mask row, so it contributes to no bucket at all. It is
        # not started, and the three counts have to add up to the total.
        not_started=max(total - in_progress - finished, 0),
        in_progress=in_progress,
        finished=finished,
        total=total,
        has_saved_queue=saved is not None,
        saved_strategy=saved.strategy if saved else None,
        saved_model_key=saved.model_key if saved else None,
        strategies=strategy_options(),
    )

"""Building and persisting annotation queues: the order an annotator works images.

Mirrors ``app.services.review_queue`` but with two deliberate differences:

* **Whole-image ordering, not instance scoring.** A strategy maps the dataset's
  images (in upload order) to an ordered list of image ids — there is nothing to
  score per contour.
* **Persisted, not a snapshot.** The built order is saved per (dataset, user) in
  ``annotation_queues`` so re-entering resumes it; rebuilding overwrites the row.

Orderings live in a registry so an active-learning ordering (e.g. diversity
sampling) can be added without touching the request/response contract: register a
strategy and it appears in the builder. A strategy may be registered as a
*placeholder* (``available=False``) — it shows in the UI but cannot be built yet.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.annotation_queues import AnnotationQueues
from app.database.images import Images
from app.database.masks import Masks
from app.schemas.annotation_queue import (
    AnnotationQueueRead,
    AnnotationQueueStrategyOption,
    AnnotationQueueSummary,
)

#: An ordering strategy: uploaded-order image ids -> queue-order image ids. Must be
#: pure; it runs once at build time.
OrderFn = Callable[[list[int]], list[int]]


@dataclass(frozen=True)
class QueueStrategy:
    key: str
    label: str
    description: str
    available: bool
    order: OrderFn | None  # None for placeholder strategies.


SORT_STRATEGIES: dict[str, QueueStrategy] = {}


def register_strategy(key: str, label: str, description: str, available: bool = True):
    """Register a queue ordering. Registered keys appear in the builder."""

    def decorator(fn: OrderFn) -> OrderFn:
        SORT_STRATEGIES[key] = QueueStrategy(
            key=key, label=label, description=description, available=available, order=fn
        )
        return fn

    return decorator


@register_strategy(
    "as_uploaded",
    "As uploaded",
    "Annotate images in the order they were added to the dataset.",
)
def _order_as_uploaded(image_ids: list[int]) -> list[int]:
    return list(image_ids)


@register_strategy(
    "random",
    "Randomized order",
    "Shuffle the images. The order is fixed once built, so it stays stable across sessions.",
)
def _order_random(image_ids: list[int]) -> list[int]:
    shuffled = list(image_ids)
    random.shuffle(shuffled)
    return shuffled


@register_strategy(
    "diversity",
    "Diversity sampling (active learning)",
    "Order images to maximise visual diversity early on. Coming soon.",
    available=False,
)
def _order_diversity(image_ids: list[int]) -> list[int]:  # pragma: no cover - placeholder
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="The diversity-sampling ordering is not available yet.",
    )


def strategy_options() -> list[AnnotationQueueStrategyOption]:
    return [
        AnnotationQueueStrategyOption(
            key=s.key, label=s.label, description=s.description, available=s.available
        )
        for s in SORT_STRATEGIES.values()
    ]


def _uploaded_image_ids(dataset_id: int, db: Session) -> list[int]:
    """The dataset's image ids in upload order (ascending id)."""
    rows = (
        db.query(Images.id)
        .filter(Images.dataset_id == dataset_id)
        .order_by(Images.id.asc())
        .all()
    )
    return [image_id for (image_id,) in rows]


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
        image_ids=image_ids,
        total=len(image_ids),
        updated_at=row.updated_at,
    )


def build_and_save_queue(
    dataset_id: int, username: str, strategy_key: str, db: Session
) -> AnnotationQueueRead:
    """Build the ordered image list for the strategy and persist it (upsert)."""
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

    ordered_ids = strategy.order(_uploaded_image_ids(dataset_id, db))

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
    row.image_order = ordered_ids
    db.commit()
    db.refresh(row)

    return AnnotationQueueRead(
        strategy=row.strategy,
        image_ids=list(row.image_order or []),
        total=len(row.image_order or []),
        updated_at=row.updated_at,
    )


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
        strategies=strategy_options(),
    )

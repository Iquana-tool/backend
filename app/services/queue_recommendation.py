"""Recommended queues: the settings a curator suggests for a dataset's annotation or review
queue, with a one-line reason.

A recommendation is a starting point, not a rule: the builder offers "Use recommended queue",
which fills in these settings and leaves them open to change. It is stored on the dataset
(``datasets.recommended_queues``), one per queue kind, and validated like a queue being built,
so a saved recommendation always builds. A study plan can later write the same contract.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.database.datasets import Datasets
from app.schemas.annotation_queue import QueueSettings
from app.schemas.queue import QueueKind, RecommendedQueue, RecommendedQueueWrite
from app.schemas.review import ReviewRecommendedSettings
from app.services import annotation_queue, queue_core
from app.services.queue_methods import OBJECTS


def _dataset(db: Session, dataset_id: int) -> Datasets:
    dataset = db.get(Datasets, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Dataset {dataset_id} not found.")
    return dataset


def _validated(kind: QueueKind, settings: dict) -> dict:
    """The settings as the builder will read them; 400 when they could not build a queue."""
    try:
        if kind == "annotation":
            parsed = QueueSettings.model_validate(settings)
            annotation_queue.required_models(parsed)
            # Whether the builder opens is each person's own choice.
            return parsed.model_dump(exclude={"show_builder"})
        parsed = ReviewRecommendedSettings.model_validate(settings)
        queue_core.chosen_methods(OBJECTS, parsed)
        return parsed.model_dump()
    except ValidationError as exc:
        raise queue_core.bad_request(
            "; ".join(error["msg"] for error in exc.errors(include_url=False))) from exc


def set_recommended(db: Session, dataset_id: int, kind: QueueKind,
                    body: RecommendedQueueWrite, username: str) -> RecommendedQueue:
    dataset = _dataset(db, dataset_id)
    recommended = RecommendedQueue(
        settings=_validated(kind, body.settings), reason=body.reason.strip(),
        updated_by=username, updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    # A new dict, not an in-place change: SQLAlchemy only notices a reassigned JSON value.
    dataset.recommended_queues = {**(dataset.recommended_queues or {}),
                                  kind: recommended.model_dump(mode="json")}
    db.commit()
    return recommended


def clear_recommended(db: Session, dataset_id: int, kind: QueueKind) -> None:
    dataset = _dataset(db, dataset_id)
    remaining = {k: v for k, v in (dataset.recommended_queues or {}).items() if k != kind}
    dataset.recommended_queues = remaining or None
    db.commit()

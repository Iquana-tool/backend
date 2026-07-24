"""Saved annotation queues: the order one annotator works a dataset's images in.

Unlike the review queue (a throwaway snapshot built per session, see
``app.services.review_queue``), an annotation queue is *persisted* per
(dataset, user): the annotator builds it once — as-uploaded, randomized, or a
future active-learning ordering — and re-clicking the Annotation card resumes the
same order instead of asking again. One row per (dataset, user); rebuilding
overwrites it.

The stored ``image_order`` is the frozen list of image ids in queue order. Storing
the resolved list (rather than only the strategy) keeps a randomized or
active-learning order stable across sessions, and makes "as uploaded" and a
future model-scored order look identical to the consumer that applies it.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)

from app.database import database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnnotationQueues(database):
    """One annotator's saved image ordering for a dataset."""

    __tablename__ = "annotation_queues"
    __table_args__ = (
        UniqueConstraint("dataset_id", "username", name="uq_annotation_queue_dataset_user"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(
        Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    username = Column(String, ForeignKey("users.username", ondelete="CASCADE"), nullable=False)
    # Key of the strategy the order was built with (see app.services.annotation_queue).
    strategy = Column(String(32), nullable=False)
    # Frozen list of image ids in queue order — a JSON array.
    image_order = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)

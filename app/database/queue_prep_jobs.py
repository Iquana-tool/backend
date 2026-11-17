"""Background jobs that fill in what an annotation-queue ordering needs.

A job fills what an ordering's methods need (see app.services.queue_methods): vectors from an
embedding model for every image, and/or a trained model's scores for every unfinished image.
A job walks its image list one image per Celery task (see
app.services.queue_prep), so it can be cancelled between images and resumes from
``next_index`` after a worker crash.
"""
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text

from app.database import database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class QueuePrepJobs(database):
    __tablename__ = "queue_prep_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    created_by = Column(String, ForeignKey("users.username", ondelete="SET NULL",
                                           onupdate="CASCADE"), nullable=True)
    #: What to prepare: {"embedding_model": key, "trained_model": key}, either optional.
    needs = Column(JSON, nullable=False)
    #: The images to prepare, in processing order.
    image_ids = Column(JSON, nullable=False, default=list)
    #: Index into image_ids of the next image to process; the job's cursor.
    next_index = Column(Integer, nullable=False, default=0)
    #: pending, running, cancelling, cancelled, succeeded, partial (finished with failures),
    #: failed (the worker stopped and the job was given up; see ``app.services.queue_prep``).
    status = Column(String(16), nullable=False, default="pending", index=True)
    failed_image_ids = Column(JSON, nullable=False, default=list)
    error = Column(Text, nullable=True)
    celery_task_id = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    #: Last time a worker finished an image of this job. A job that is active but has not
    #: beaten for a while lost its worker, and must not block new preparation forever.
    heartbeat_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

"""Background jobs that fill in what an annotation-queue ordering needs.

Diversity and Density need an image vector for every image; Uncertainty needs a model score
for every unfinished image. A job walks its image list one image per Celery task (see
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
    strategy = Column(String(32), nullable=False)
    model_key = Column(String(255), nullable=True)
    #: The images to prepare, in processing order.
    image_ids = Column(JSON, nullable=False, default=list)
    #: Index into image_ids of the next image to process; the job's cursor.
    next_index = Column(Integer, nullable=False, default=0)
    #: pending, running, cancelling, cancelled, succeeded, partial (finished with failures).
    status = Column(String(16), nullable=False, default="pending", index=True)
    failed_image_ids = Column(JSON, nullable=False, default=list)
    error = Column(Text, nullable=True)
    celery_task_id = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    finished_at = Column(DateTime, nullable=True)

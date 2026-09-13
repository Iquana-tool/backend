"""Batch inference jobs: annotating a whole dataset with an orchestration of models.

A job is a frozen plan plus a work list. The plan (`plan_steps`) says which model produces
which label; the work list (`InferenceJobItems`, one row per step x image) is what the worker
actually consumes. Both are materialised up front, at submit time, which buys three things:

* **Hierarchy order for free.** Every item carries the `level` of its label (0 = root). The
  worker always takes the lowest pending level, so every root instance in the dataset exists
  before a single child-level model runs -- which is what makes parent lookup possible at all.
* **A real progress bar.** Total work is `len(steps) * len(images)`, known before the first
  model loads, so "how much is left" is a `COUNT`, not an estimate.
* **Crash resumption.** The cursor *is* the item table. A worker that dies mid-job leaves the
  unfinished items `pending`; restarting the job picks up exactly where it stopped.

Contours produced by a job are ordinary contours -- `added_by` names the model and
`author_username` the person who started the job -- so they arrive unreviewed and flow into
the existing review queue like any other annotation.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.database import database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


#: Job lifecycle. `cancelling` is a request, not a state the worker set: the API writes it and
#: the worker turns it into `cancelled` when it next comes up for air between items.
JOB_STATUSES = ("pending", "running", "cancelling", "cancelled", "succeeded", "partial", "failed")

#: Terminal job statuses -- nothing will move again without a new job.
TERMINAL_JOB_STATUSES = frozenset({"cancelled", "succeeded", "partial", "failed"})

#: Item lifecycle. `skipped` covers items the job never reached (cancelled mid-run).
ITEM_STATUSES = ("pending", "running", "done", "failed", "skipped")


class InferenceJobs(database):
    """One batch-inference run over a dataset."""

    __tablename__ = "inference_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(
        Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # SET NULL rather than CASCADE: a deleted account must not take the run history of a
    # dataset with it -- the contours it wrote are still there.
    created_by = Column(String, ForeignKey("users.username", ondelete="SET NULL"), nullable=True)
    name = Column(String(80), nullable=True)

    status = Column(String(16), nullable=False, default="pending", index=True)
    # "patch" (add to what is there, dropping duplicates) or "replace" (wipe first).
    write_mode = Column(String(16), nullable=False, default="patch")

    #: The resolved plan: a JSON list of steps, each already carrying its hierarchy `level`.
    #: Frozen at submit time so a later label or model change cannot rewrite history.
    plan_steps = Column(JSON, nullable=False, default=list)
    #: Merge/threshold settings for this run (see `app.schemas.inference.InferenceOptions`).
    options = Column(JSON, nullable=False, default=dict)
    #: Frozen list of image ids in scope, resolved from the requested selection at submit
    #: time -- images uploaded while the job runs are not silently pulled in.
    image_ids = Column(JSON, nullable=False, default=list)

    total_units = Column(Integer, nullable=False, default=0)
    contours_created = Column(Integer, nullable=False, default=0)
    contours_suppressed = Column(Integer, nullable=False, default=0)
    contours_deleted = Column(Integer, nullable=False, default=0)
    contours_unparented = Column(Integer, nullable=False, default=0)

    #: Celery id of the currently scheduled `inference.run_next`, kept so a cancel can revoke
    #: a task that has not been picked up yet instead of waiting for it to start.
    celery_task_id = Column(String, nullable=True)
    error = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=_utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    items = relationship(
        "InferenceJobItems", back_populates="job", passive_deletes=True,
        order_by="InferenceJobItems.id",
    )


class InferenceJobItems(database):
    """One unit of work: run one plan step over one image."""

    __tablename__ = "inference_job_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(
        Integer, ForeignKey("inference_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: Depth of the step's label in the label hierarchy (0 = root). The worker's ordering key.
    level = Column(Integer, nullable=False, default=0)
    #: Position of the step in the job's `plan_steps`.
    step_index = Column(Integer, nullable=False, default=0)
    image_id = Column(
        Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )

    status = Column(String(16), nullable=False, default="pending", index=True)
    contours_created = Column(Integer, nullable=False, default=0)
    contours_suppressed = Column(Integer, nullable=False, default=0)
    contours_unparented = Column(Integer, nullable=False, default=0)
    duration_ms = Column(Float, nullable=True)
    error = Column(Text, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    job = relationship("InferenceJobs", back_populates="items")

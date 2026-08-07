"""The Celery tasks that walk a job's work list.

**Why a self-chaining task and not a chord.** The obvious shape -- fan every image out as a
group and join with a callback -- does not survive contact with this deployment. The Celery
workers run with ``--pool=solo`` (the prefork pool does not work on Windows here), so a task
that blocks waiting for its own subtasks deadlocks the single worker that would have to run
them. And there is nothing to gain from the fan-out anyway: every unit ends in the same
AI service on the same GPU, so parallel dispatch would only trade throughput for VRAM
contention.

So each :func:`run_next` handles exactly one unit and then queues itself again. That costs a
broker round-trip per image -- microseconds against a forward pass -- and buys three things:

* **Cancellation lands within one unit.** The status is re-read at the top of every task, so
  a cancel takes effect after the current image rather than at the end of the run.
* **Crash resumption is free.** The cursor is the item table; a worker that dies leaves its
  unfinished items ``pending``.
* **Hierarchy order is structural.** "The next unit" is the lowest-``level`` pending row, so
  no child-level step can start before every root-level unit in the dataset is done -- see
  :mod:`app.services.inference.planning`.

A unit that fails does not fail the run: it is recorded on its item row and the walk goes on,
so one unreadable image cannot cost a user a six-hour job. A run that ends with failures is
``partial``, not ``succeeded``, and the failed items are listed in the UI.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from logging import getLogger

from app.database import get_context_session
from app.database.images import Images
from app.database.inference_jobs import InferenceJobItems, InferenceJobs
from app.schemas.inference import InferenceOptions, ResolvedStep, WriteMode
from app.services.celery_app import celery_app
from app.services.inference.execution import InferenceUnitError, run_unit, wipe_images

logger = getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _next_item(db, job_id: int) -> InferenceJobItems | None:
    """The next unit to run: lowest hierarchy level first, then plan order, then image order.

    ``level`` leading the sort is the entire hierarchy guarantee. Everything else is only
    there to make a run reproducible.
    """
    return (
        db.query(InferenceJobItems)
        .filter(InferenceJobItems.job_id == job_id, InferenceJobItems.status == "pending")
        .order_by(
            InferenceJobItems.level,
            InferenceJobItems.step_index,
            InferenceJobItems.image_id,
        )
        .first()
    )


def finish(db, job: InferenceJobs, status: str, error: str | None = None) -> None:
    job.status = status
    job.error = error
    job.finished_at = _utcnow()
    job.celery_task_id = None
    db.commit()
    logger.info("Inference job %s finished as %s.", job.id, status)


def abandon_pending(db, job_id: int) -> None:
    """Mark everything still queued as skipped, so the counts add up after a cancel."""
    db.query(InferenceJobItems).filter(
        InferenceJobItems.job_id == job_id, InferenceJobItems.status.in_(("pending", "running"))
    ).update({InferenceJobItems.status: "skipped"}, synchronize_session=False)


@celery_app.task(name="inference.run_job", bind=True)
def run_job(self, job_id: int) -> dict:
    """Start a job: claim it, apply the write mode, then hand over to the walk.

    A replace run does its deletion here, once, before any prediction exists -- not per
    image. Doing it up front means a cancel halfway through cannot leave a dataset where the
    first half was wiped and re-annotated and the second half was only wiped.
    """
    with get_context_session() as db:
        job = db.get(InferenceJobs, job_id)
        if job is None:
            logger.warning("Inference job %s vanished before it started.", job_id)
            return {"status": "missing"}
        if job.status not in ("pending", "running"):
            return {"status": job.status}

        job.status = "running"
        job.started_at = job.started_at or _utcnow()
        db.commit()

        if job.write_mode == WriteMode.REPLACE.value:
            options = InferenceOptions.model_validate(job.options)
            try:
                job.contours_deleted = wipe_images(
                    db, list(job.image_ids), preserve_reviewed=options.preserve_reviewed
                )
                db.commit()
            except Exception as exc:
                logger.exception("Replace wipe failed for job %s.", job_id)
                finish(db, job, "failed", f"Could not clear existing annotations: {exc}")
                return {"status": "failed"}

    first = run_next.apply_async((job_id,))
    with get_context_session() as db:
        job = db.get(InferenceJobs, job_id)
        if job is not None:
            job.celery_task_id = first.id
            db.commit()
    return {"status": "running"}


@celery_app.task(name="inference.run_next", bind=True)
def run_next(self, job_id: int) -> dict:
    """Run one unit of a job, then queue the next one."""
    with get_context_session() as db:
        job = db.get(InferenceJobs, job_id)
        if job is None:
            return {"status": "missing"}
        if job.status == "cancelling":
            abandon_pending(db, job_id)
            finish(db, job, "cancelled")
            return {"status": "cancelled"}
        if job.status != "running":
            return {"status": job.status}

        item = _next_item(db, job_id)
        if item is None:
            failed = db.query(InferenceJobItems).filter(
                InferenceJobItems.job_id == job_id, InferenceJobItems.status == "failed"
            ).count()
            finish(db, job, "partial" if failed else "succeeded")
            return {"status": job.status}

        steps = [ResolvedStep.model_validate(step) for step in job.plan_steps]
        options = InferenceOptions.model_validate(job.options)
        _run_one(db, job, item, steps[item.step_index], options)

    # Queued outside the session so the row lock is gone before the next task can pick it up.
    followup = run_next.apply_async((job_id,))
    with get_context_session() as db:
        job = db.get(InferenceJobs, job_id)
        if job is not None and job.status == "running":
            job.celery_task_id = followup.id
            db.commit()
    return {"status": "running"}


def _run_one(
    db, job: InferenceJobs, item: InferenceJobItems, step: ResolvedStep, options: InferenceOptions
) -> None:
    """Execute one work item and fold its outcome into the job's counters.

    A failure is recorded and swallowed: a run over thousands of images must not be lost to a
    single corrupt file or a model that chokes on one input.
    """
    item.status = "running"
    db.commit()

    started = time.perf_counter()
    image = db.get(Images, item.image_id)
    try:
        if image is None:
            raise InferenceUnitError(f"Image {item.image_id} no longer exists.")
        result = run_unit(db, step, image, options, job.created_by or "system")
        db.commit()
    except Exception as exc:
        db.rollback()
        message = str(exc) if isinstance(exc, InferenceUnitError) else f"{type(exc).__name__}: {exc}"
        logger.warning("Inference job %s, image %s, step %s failed: %s",
                       job.id, item.image_id, item.step_index, message)
        item = db.get(InferenceJobItems, item.id)
        item.status = "failed"
        item.error = message[:2000]
        item.duration_ms = (time.perf_counter() - started) * 1000
        item.finished_at = _utcnow()
        db.commit()
        return

    item.status = "done"
    item.contours_created = result.created
    item.contours_suppressed = result.suppressed
    item.contours_unparented = result.unparented
    item.duration_ms = (time.perf_counter() - started) * 1000
    item.finished_at = _utcnow()

    job.contours_created += result.created
    job.contours_suppressed += result.suppressed
    job.contours_unparented += result.unparented
    db.commit()

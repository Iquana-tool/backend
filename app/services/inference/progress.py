"""Reading a job's progress back out of the work list.

The item table is both the cursor and the progress bar: "how far along is this run" is a
``GROUP BY status`` over it, not a number the worker has to remember to increment. That is
what makes the bar survive a worker restart -- there is no in-memory state to lose.

The ETA is deliberately naive: mean duration of the most recently finished units times the
number left. Per-unit cost is dominated by one forward pass on one image, so the mean is
stable within a level; it jumps when the run moves to a level whose model is slower, and
recovers within a few units because the window is short.
"""
from __future__ import annotations

from logging import getLogger

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.images import Images
from app.database.inference_jobs import InferenceJobItems, InferenceJobs
from app.schemas.inference import (
    ActivityEntry,
    InferenceJobItemRead,
    InferenceJobSnapshot,
    InferenceOptions,
    LevelProgress,
    ResolvedStep,
)

logger = getLogger(__name__)

#: How many recently finished units the ETA averages over. Short enough to react when the run
#: moves to a slower level, long enough not to swing on one unusually crowded image.
_ETA_WINDOW = 20

#: Units that must finish before an ETA is worth showing at all.
_ETA_MIN_SAMPLES = 3

#: How many finished images the live feed shows. Enough to see the run moving, small enough
#: that the payload stays flat however long the run is.
_ACTIVITY_WINDOW = 8


def _status_counts(db: Session, job_id: int) -> dict[str, int]:
    rows = (
        db.query(InferenceJobItems.status, func.count(InferenceJobItems.id))
        .filter(InferenceJobItems.job_id == job_id)
        .group_by(InferenceJobItems.status)
        .all()
    )
    return {status: int(count) for status, count in rows}


def _level_progress(db: Session, job_id: int, steps: list[ResolvedStep]) -> list[LevelProgress]:
    rows = (
        db.query(
            InferenceJobItems.level,
            InferenceJobItems.status,
            func.count(InferenceJobItems.id),
        )
        .filter(InferenceJobItems.job_id == job_id)
        .group_by(InferenceJobItems.level, InferenceJobItems.status)
        .all()
    )
    by_level: dict[int, LevelProgress] = {}
    for level, status, count in rows:
        progress = by_level.setdefault(level, LevelProgress(level=level))
        progress.total += int(count)
        if status in ("done", "skipped"):
            progress.done += int(count)
        elif status == "failed":
            progress.done += int(count)
            progress.failed += int(count)
    for step in steps:
        progress = by_level.setdefault(step.level, LevelProgress(level=step.level))
        if step.label_name not in progress.label_names:
            progress.label_names.append(step.label_name)
    return [by_level[level] for level in sorted(by_level)]


def _eta_seconds(db: Session, job_id: int, remaining: int) -> float | None:
    """Seconds left, from the mean duration of the last few finished units."""
    if remaining <= 0:
        return None
    durations = [
        row[0] for row in
        db.query(InferenceJobItems.duration_ms)
        .filter(
            InferenceJobItems.job_id == job_id,
            InferenceJobItems.status == "done",
            InferenceJobItems.duration_ms.isnot(None),
        )
        .order_by(InferenceJobItems.finished_at.desc())
        .limit(_ETA_WINDOW)
        .all()
    ]
    if len(durations) < _ETA_MIN_SAMPLES:
        return None
    return (sum(durations) / len(durations)) * remaining / 1000.0


def _current_step(db: Session, job: InferenceJobs, steps: list[ResolvedStep]) -> ResolvedStep | None:
    """The step the worker is on -- the running unit's, or the next pending one's."""
    item = (
        db.query(InferenceJobItems)
        .filter(InferenceJobItems.job_id == job.id, InferenceJobItems.status == "running")
        .first()
    )
    if item is None:
        item = (
            db.query(InferenceJobItems)
            .filter(InferenceJobItems.job_id == job.id, InferenceJobItems.status == "pending")
            .order_by(
                InferenceJobItems.level,
                InferenceJobItems.step_index,
                InferenceJobItems.image_id,
            )
            .first()
        )
    if item is None or item.step_index >= len(steps):
        return None
    return steps[item.step_index]


def _recent_activity(
    db: Session, job: InferenceJobs, steps: list[ResolvedStep]
) -> list[ActivityEntry]:
    """The last few finished images, newest first.

    This is the cheap half of "watch it work". Streaming the actual contours would mean
    pushing every polygon the model found through the SSE channel every couple of seconds --
    hundreds of objects per image, none of which the progress view can usefully draw. Names
    and counts answer the same question ("is it finding things, and are they plausible?") for
    a few hundred bytes, and each row links to the image for anyone who wants to look.
    """
    items = (
        db.query(InferenceJobItems)
        .filter(InferenceJobItems.job_id == job.id, InferenceJobItems.status == "done")
        .order_by(InferenceJobItems.finished_at.desc())
        .limit(_ACTIVITY_WINDOW)
        .all()
    )
    if not items:
        return []
    names = {
        image_id: file_name for image_id, file_name in
        db.query(Images.id, Images.file_name)
        .filter(Images.id.in_([item.image_id for item in items]))
        .all()
    }
    return [
        ActivityEntry(
            image_id=item.image_id,
            image_name=names.get(item.image_id),
            label_name=steps[item.step_index].label_name if item.step_index < len(steps) else None,
            contours_created=item.contours_created,
            contours_suppressed=item.contours_suppressed,
            finished_at=item.finished_at,
        )
        for item in items
    ]


def snapshot(db: Session, job: InferenceJobs) -> InferenceJobSnapshot:
    """Build the progress payload the UI renders."""
    steps = [ResolvedStep.model_validate(step) for step in job.plan_steps or []]
    counts = _status_counts(db, job.id)
    failed = counts.get("failed", 0)
    done = counts.get("done", 0) + counts.get("skipped", 0) + failed
    remaining = max(0, (job.total_units or 0) - done)

    return InferenceJobSnapshot(
        id=job.id,
        dataset_id=job.dataset_id,
        name=job.name,
        created_by=job.created_by,
        status=job.status,
        write_mode=job.write_mode,
        options=InferenceOptions.model_validate(job.options or {}),
        steps=steps,
        total_units=job.total_units or 0,
        done_units=done,
        failed_units=failed,
        image_count=len(job.image_ids or []),
        contours_created=job.contours_created,
        contours_suppressed=job.contours_suppressed,
        contours_deleted=job.contours_deleted,
        contours_unparented=job.contours_unparented,
        levels=_level_progress(db, job.id, steps),
        current_step=_current_step(db, job, steps) if job.status == "running" else None,
        recent_activity=_recent_activity(db, job, steps),
        eta_seconds=_eta_seconds(db, job.id, remaining) if job.status == "running" else None,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def read_items(
    db: Session, job: InferenceJobs, *, status: str | None = None, limit: int = 200
) -> list[InferenceJobItemRead]:
    """Work items of a job, optionally filtered by status (the failed list uses this)."""
    steps = [ResolvedStep.model_validate(step) for step in job.plan_steps or []]
    query = db.query(InferenceJobItems).filter(InferenceJobItems.job_id == job.id)
    if status:
        query = query.filter(InferenceJobItems.status == status)
    items = query.order_by(InferenceJobItems.level, InferenceJobItems.step_index).limit(limit).all()

    names = {
        image_id: file_name for image_id, file_name in
        db.query(Images.id, Images.file_name)
        .filter(Images.id.in_([item.image_id for item in items] or [0]))
        .all()
    }
    reads: list[InferenceJobItemRead] = []
    for item in items:
        step = steps[item.step_index] if item.step_index < len(steps) else None
        reads.append(InferenceJobItemRead(
            id=item.id,
            level=item.level,
            step_index=item.step_index,
            image_id=item.image_id,
            image_name=names.get(item.image_id),
            label_name=step.label_name if step else None,
            model_registry_key=step.model_registry_key if step else None,
            status=item.status,
            contours_created=item.contours_created,
            contours_suppressed=item.contours_suppressed,
            contours_unparented=item.contours_unparented,
            duration_ms=item.duration_ms,
            error=item.error,
        ))
    return reads

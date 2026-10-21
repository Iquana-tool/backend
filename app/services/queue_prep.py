"""Filling in the data an annotation-queue ordering needs, in the background.

Diversity and Density need an image vector for every image (finished ones too: they
are Diversity's seeds). Uncertainty needs a model score for every unfinished image. ``start_prep``
works out what is missing and starts a job; each Celery task handles one image and queues
the next, the same self-chaining pattern as batch inference (see
app.services.inference.tasks for why not a chord). Cancellation lands between images, a
crashed worker resumes from ``next_index``, and one bad image is recorded, not fatal.

Scoring calls the ai-service's instance-segmentation surface, like batch inference does.
Training runs in the ai-service's own worker on the same GPU, so scoring during a training
run can compete for GPU memory.
"""
from __future__ import annotations

from datetime import datetime, timezone
from logging import getLogger

from sqlalchemy.orm import Session

from app.database import get_context_session
from app.database.embeddings import Embeddings
from app.database.image_scores import ImageScores
from app.database.images import Images
from app.database.queue_prep_jobs import QueuePrepJobs
from app.schemas.annotation_queue import QueuePrepJobRead
from app.services import embedding_lifecycle, image_scoring
from app.services.annotation_queue import get_strategy, image_states
from app.services.celery_app import celery_app
from config import EMBEDDING_MODEL_ID

logger = getLogger(__name__)

ACTIVE_STATES = ("pending", "running", "cancelling")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ids_with_vectors(db: Session, image_ids: list[int]) -> set[int]:
    return {i for (i,) in db.query(Embeddings.image_id).filter(
        Embeddings.kind == "image_cls", Embeddings.model_id == EMBEDDING_MODEL_ID,
        Embeddings.image_id.in_(image_ids))}


def _ids_with_scores(db: Session, image_ids: list[int], model_key: str) -> set[int]:
    # Rows without confidences predate the current scoring and are scored again.
    return {i for (i,) in db.query(ImageScores.image_id).filter(
        ImageScores.model_key == model_key, ImageScores.confidences.isnot(None),
        ImageScores.image_id.in_(image_ids))}


def missing_work(db: Session, dataset_id: int, strategy_key: str,
                 model_key: str | None) -> list[int]:
    """Image ids that still need a vector or a score for this ordering, ascending."""
    strategy = get_strategy(strategy_key)
    all_ids = [i for (i,) in db.query(Images.id).filter(Images.dataset_id == dataset_id)]
    missing: set[int] = set()
    if strategy.needs_embeddings:
        missing |= set(all_ids) - _ids_with_vectors(db, all_ids)
    if strategy.needs_scores and model_key:
        states = image_states(db, dataset_id)
        unfinished = [i for i in all_ids if states.get(i) != "finished"]
        missing |= set(unfinished) - _ids_with_scores(db, unfinished, model_key)
    return sorted(missing)


def to_read(job: QueuePrepJobs) -> QueuePrepJobRead:
    return QueuePrepJobRead(
        id=job.id, status=job.status, strategy=job.strategy, model_key=job.model_key,
        total=len(job.image_ids or []), done=job.next_index,
        failed=len(job.failed_image_ids or []),
    )


def start_prep(db: Session, dataset_id: int, username: str, strategy_key: str,
               model_key: str | None, enqueue: bool = True) -> QueuePrepJobs | None:
    """Start filling missing data, or return None when the ordering can be built now.

    A job already running for the same dataset, ordering and model is returned instead of
    starting a second one, so two annotators never score the same images twice.
    """
    image_ids = missing_work(db, dataset_id, strategy_key, model_key)
    if not image_ids:
        return None
    existing = (db.query(QueuePrepJobs)
                .filter(QueuePrepJobs.dataset_id == dataset_id,
                        QueuePrepJobs.strategy == strategy_key,
                        QueuePrepJobs.model_key == model_key,
                        QueuePrepJobs.status.in_(ACTIVE_STATES))
                .first())
    if existing is not None:
        return existing
    job = QueuePrepJobs(dataset_id=dataset_id, created_by=username, strategy=strategy_key,
                        model_key=model_key, image_ids=image_ids)
    db.add(job)
    db.commit()
    if enqueue:
        result = run_prep_next.apply_async((job.id,))
        job.celery_task_id = result.id
        db.commit()
    return job


def cancel(db: Session, job: QueuePrepJobs) -> None:
    """Ask the job to stop; it ends after the image it is working on."""
    if job.status in ("pending", "running"):
        job.status = "cancelling"
        db.commit()


def _finish(db: Session, job: QueuePrepJobs, status: str) -> str:
    job.status = status
    job.finished_at = _utcnow()
    job.celery_task_id = None
    db.commit()
    return status


def _run_unit(db: Session, job: QueuePrepJobs, image_id: int, embed_client, score_client) -> None:
    strategy = get_strategy(job.strategy)
    image = db.get(Images, image_id)
    if image is None:
        return  # Deleted since the job started: nothing to prepare.
    if strategy.needs_embeddings and not _ids_with_vectors(db, [image_id]):
        embedding_lifecycle.embed_image(db, image, client=embed_client)
    if (strategy.needs_scores and job.model_key
            and not _ids_with_scores(db, [image_id], job.model_key)):
        image_scoring.score_image(db, image, job.model_key, job.created_by or "system",
                                  client=score_client)


def step(db: Session, job_id: int, embed_client=None, score_client=None) -> str:
    """Process the job's next image and return its state afterwards."""
    job = db.get(QueuePrepJobs, job_id)
    if job is None:
        return "missing"
    if job.status == "cancelling":
        return _finish(db, job, "cancelled")
    if job.status not in ("pending", "running"):
        return job.status
    if job.next_index >= len(job.image_ids):
        return _finish(db, job, "partial" if job.failed_image_ids else "succeeded")

    job.status = "running"
    image_id = job.image_ids[job.next_index]
    try:
        _run_unit(db, job, image_id, embed_client, score_client)
        db.commit()
    except Exception as exc:  # one bad image must not stop the job
        db.rollback()
        logger.warning("Queue prep job %s, image %s failed: %s", job_id, image_id, exc)
        job = db.get(QueuePrepJobs, job_id)
        job.failed_image_ids = [*(job.failed_image_ids or []), image_id]
        job.status = "running"
    job.next_index += 1
    db.commit()
    return "running"


@celery_app.task(name="queue_prep.run_next")
def run_prep_next(job_id: int) -> str:
    """One image per task, then queue the next; see the module docstring."""
    with get_context_session() as db:
        state = step(db, job_id)
    if state == "running":
        run_prep_next.apply_async((job_id,))
    return state

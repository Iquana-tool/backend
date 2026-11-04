"""Filling in the data an annotation-queue ordering needs, in the background.

An ordering needs (see ``app.services.queue_methods``) vectors from an embedding model for
every image -- finished ones too, they are what Diversity measures distance from -- and/or a
trained model's detections on every unfinished image. ``start_prep`` works out what is
missing and starts a job; each Celery task handles one image and queues the next, the same
self-chaining pattern as batch inference (see app.services.inference.tasks for why not a
chord). Cancellation lands between images, a crashed worker resumes from ``next_index``, and
one bad image is recorded, not fatal.

Both calls go to the ai-service, like batch inference. Training runs in the ai-service's own
worker on the same GPU, so preparing during a training run can compete for GPU memory.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from logging import getLogger
from typing import Protocol

from sqlalchemy.orm import Session

from app.database import get_context_session
from app.database.image_scores import ImageScores
from app.database.image_vectors import ImageVectors
from app.database.images import Images
from app.database.queue_prep_jobs import QueuePrepJobs
from app.schemas.annotation_queue import QueuePrepJobRead
from app.services import image_scoring
from app.services.annotation_queue import image_states
from app.services.celery_app import celery_app
from app.services.queue_methods import EMBEDDING_MODEL, TRAINED_MODEL

logger = getLogger(__name__)

ACTIVE_STATES = ("pending", "running", "cancelling")
IMAGE_KIND = "image_cls"
#: An active job whose worker has not finished an image for this long is given up. One image
#: takes seconds; the first one of a job may also load a model in the ai-service.
STALE_AFTER = timedelta(minutes=10)


class EmbedClient(Protocol):
    def request_embeddings(self, request): ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    # DateTime columns come back naive on PostgreSQL and SQLite; they are written in UTC.
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def is_stale(job: QueuePrepJobs, now: datetime | None = None) -> bool:
    """Whether an active job has lost its worker (no image finished for ``STALE_AFTER``)."""
    if job.status not in ACTIVE_STATES:
        return False
    last = _as_utc(job.heartbeat_at) or _as_utc(job.created_at)
    return last is not None and (now or _utcnow()) - last > STALE_AFTER


def _ids_with_vectors(db: Session, image_ids: list[int], model_key: str) -> set[int]:
    return {i for (i,) in db.query(ImageVectors.image_id).filter(
        ImageVectors.model_key == model_key, ImageVectors.image_id.in_(image_ids))}


def _ids_with_scores(db: Session, image_ids: list[int], model_key: str) -> set[int]:
    # Rows without confidences predate the current scoring and are scored again.
    return {i for (i,) in db.query(ImageScores.image_id).filter(
        ImageScores.model_key == model_key, ImageScores.confidences.isnot(None),
        ImageScores.image_id.in_(image_ids))}


def missing_work(db: Session, dataset_id: int, needs: dict[str, str]) -> list[int]:
    """Image ids that still lack a vector or a score the needs call for, ascending."""
    all_ids = [i for (i,) in db.query(Images.id).filter(Images.dataset_id == dataset_id)]
    missing: set[int] = set()
    if needs.get(EMBEDDING_MODEL):
        missing |= set(all_ids) - _ids_with_vectors(db, all_ids, needs[EMBEDDING_MODEL])
    if needs.get(TRAINED_MODEL):
        states = image_states(db, dataset_id)
        unfinished = [i for i in all_ids if states.get(i) != "finished"]
        missing |= set(unfinished) - _ids_with_scores(db, unfinished, needs[TRAINED_MODEL])
    return sorted(missing)


def to_read(job: QueuePrepJobs) -> QueuePrepJobRead:
    return QueuePrepJobRead(id=job.id, status=job.status, total=len(job.image_ids or []),
                            done=job.next_index, failed=len(job.failed_image_ids or []))


def start_prep(db: Session, dataset_id: int, username: str, needs: dict[str, str],
               enqueue: bool = True) -> QueuePrepJobs | None:
    """Start filling missing data, or return None when the queue can be built now.

    A job already running for the same dataset and needs is returned instead of starting a
    second one, so two annotators never compute the same data twice. A job that lost its
    worker (see ``is_stale``) is given up first, so it cannot block preparation forever.
    """
    image_ids = missing_work(db, dataset_id, needs)
    if not image_ids:
        return None
    running = (db.query(QueuePrepJobs)
               .filter(QueuePrepJobs.dataset_id == dataset_id,
                       QueuePrepJobs.status.in_(ACTIVE_STATES)))
    existing = next((job for job in running if job.needs == needs), None)
    if existing is not None and is_stale(existing):
        existing.error = "The worker stopped responding; preparation was started again."
        _finish(db, existing, "failed")
        existing = None
    if existing is not None:
        return existing
    job = QueuePrepJobs(dataset_id=dataset_id, created_by=username, needs=needs,
                        image_ids=image_ids)
    db.add(job)
    db.commit()
    if enqueue:
        job.celery_task_id = run_prep_next.apply_async((job.id,)).id
        db.commit()
    return job


def can_cancel(job: QueuePrepJobs, username: str) -> bool:
    """Only whoever started a job may stop it: others may be waiting on the same job."""
    return job.created_by in (None, username)


def cancel(db: Session, job: QueuePrepJobs) -> None:
    """Ask the job to stop; it ends after the image it is working on.

    Only a pending or running job moves to ``cancelling``, and the worker never writes a
    status back over it (see ``step``), so a cancel that lands mid-image is not lost.
    """
    db.query(QueuePrepJobs).filter(
        QueuePrepJobs.id == job.id, QueuePrepJobs.status.in_(("pending", "running"))
    ).update({QueuePrepJobs.status: "cancelling"}, synchronize_session=False)
    db.commit()
    db.refresh(job)


def _finish(db: Session, job: QueuePrepJobs, status: str) -> str:
    job.status = status
    job.finished_at = _utcnow()
    job.celery_task_id = None
    db.commit()
    return status


def _embed(db: Session, image: Images, model_key: str, client: EmbedClient | None) -> None:
    from iquana_toolbox.schemas.networking.http.services import EmbedRequest
    from app.services.ai_services.embedding import EmbeddingService

    request = EmbedRequest(image_url=str(image.file_path), user_id="system",
                           model_registry_key=model_key, image_kinds=[IMAGE_KIND], regions=[])
    vector = next(v.vector for v in (client or EmbeddingService()).request_embeddings(request)
                  if v.kind == IMAGE_KIND)
    db.add(ImageVectors(image_id=image.id, model_key=model_key, vector=list(vector)))
    db.flush()


def _run_unit(db: Session, job: QueuePrepJobs, image_id: int, embed_client, score_client) -> None:
    image = db.get(Images, image_id)
    if image is None:
        return  # Deleted since the job started: nothing to prepare.
    needs = job.needs
    if needs.get(EMBEDDING_MODEL) and not _ids_with_vectors(db, [image_id], needs[EMBEDDING_MODEL]):
        _embed(db, image, needs[EMBEDDING_MODEL], embed_client)
    if needs.get(TRAINED_MODEL) and not _ids_with_scores(db, [image_id], needs[TRAINED_MODEL]):
        image_scoring.score_image(db, image, needs[TRAINED_MODEL], job.created_by or "system",
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

    image_id = job.image_ids[job.next_index]
    failed = False
    try:
        _run_unit(db, job, image_id, embed_client, score_client)
        db.commit()
    except Exception as exc:  # one bad image must not stop the job
        db.rollback()
        logger.warning("Queue prep job %s, image %s failed: %s", job_id, image_id, exc)
        failed = True
    # Never write the status back: the API may have set "cancelling" while this image ran,
    # and the next step must see it. Only pending -> running is a transition of ours, and
    # both it and the cursor are applied in SQL rather than over a stale in-memory copy.
    jobs = db.query(QueuePrepJobs).filter(QueuePrepJobs.id == job_id)
    jobs.filter(QueuePrepJobs.status == "pending").update(
        {QueuePrepJobs.status: "running"}, synchronize_session=False)
    jobs.update({QueuePrepJobs.next_index: QueuePrepJobs.next_index + 1,
                 QueuePrepJobs.heartbeat_at: _utcnow()}, synchronize_session=False)
    if failed:
        job = db.get(QueuePrepJobs, job_id)
        job.failed_image_ids = [*(job.failed_image_ids or []), image_id]
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

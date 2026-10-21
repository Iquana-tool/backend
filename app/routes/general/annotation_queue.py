"""Annotation-queue endpoints: the persisted order an annotator works images in.

Unlike the review queue (a per-session snapshot), an annotation queue is saved per
(dataset, user): the builder on the Annotation card writes it, and re-entering the
editor resumes it. See ``app.services.annotation_queue`` for the ordering registry
(where the active-learning orderings live) and ``app.services.queue_prep`` for
the background job that fills the vectors and scores they read.
"""
import asyncio
from logging import getLogger

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.queue_prep_jobs import QueuePrepJobs
from app.schemas.annotation_queue import AnnotationQueueRequest
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services import annotation_queue, queue_prep
from app.services.ai_tools import AiTool, is_enabled
from app.services.model_registry import list_available_models
from app.services.permissions import require

router = APIRouter(prefix="/annotation-queue", tags=["annotation-queue"])
logger = getLogger(__name__)


@router.get("/datasets/{dataset_id}/summary")
async def get_annotation_queue_summary(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_CREATE, "dataset_id")),
):
    """Counts behind the Annotation card's subcaption, the available orderings, and
    whether the caller already has a saved queue to resume."""
    summary = annotation_queue.summarize(dataset_id, user.username, db)
    return {
        "success": True,
        "message": f"Annotation queue summary for dataset {dataset_id}.",
        "summary": summary.model_dump(mode="json"),
    }


@router.get("/datasets/{dataset_id}")
async def get_annotation_queue(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_CREATE, "dataset_id")),
):
    """The caller's saved queue for the dataset, or ``queue: null`` if none exists."""
    queue = annotation_queue.get_saved_queue(dataset_id, user.username, db)
    return {
        "success": True,
        "message": "Retrieved saved annotation queue."
                   if queue else "No saved annotation queue.",
        "queue": queue.model_dump(mode="json") if queue else None,
    }


@router.post("/datasets/{dataset_id}")
async def build_annotation_queue(
        dataset_id: int,
        body: AnnotationQueueRequest,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_CREATE, "dataset_id")),
):
    """Build the image order for the chosen strategy and save it (overwriting any
    earlier queue for this dataset+user)."""
    queue = annotation_queue.build_and_save_queue(
        dataset_id, user.username, body.strategy, db, model_key=body.model_key
    )
    return {
        "success": True,
        "message": f"Built an annotation queue of {queue.total} images.",
        "queue": queue.model_dump(mode="json"),
    }


@router.get("/datasets/{dataset_id}/scoring-models")
async def get_scoring_models(
        dataset_id: int,
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_CREATE, "dataset_id")),
):
    """Models trained on this dataset, which the model-based orderings can score with."""
    listing = await asyncio.to_thread(list_available_models, "instance-segmentation")
    models = annotation_queue.scoring_models_for_dataset(listing.get("result", []), dataset_id)
    return {"success": True, "models": models}


@router.post("/datasets/{dataset_id}/prepare")
async def prepare_annotation_queue(
        dataset_id: int,
        body: AnnotationQueueRequest,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_CREATE, "dataset_id")),
):
    """Fill in what the ordering needs. ``ready: true`` means build now; otherwise poll the job.

    Scoring runs a model over the dataset's images, so it obeys the dataset's batch
    inference switch.
    """
    strategy = annotation_queue.get_strategy(body.strategy)
    if strategy.needs_model and not body.model_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"The '{body.strategy}' ordering needs a trained model.")
    if strategy.needs_scores and not is_enabled(dataset_id, AiTool.BATCH_INFERENCE, db):
        # Said in the annotator's terms, with where to switch it back on; the generic
        # message names the tool by its internal key.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"{strategy.label} needs Batch inference, which is switched off for this "
                   "dataset. It can be switched on in Manage access, under Review policy.",
        )
    job = queue_prep.start_prep(db, dataset_id, user.username, body.strategy,
                                body.model_key if strategy.needs_model else None)
    return {
        "success": True,
        "ready": job is None,
        "job": queue_prep.to_read(job).model_dump() if job is not None else None,
    }


def _job_in_dataset(db: Session, dataset_id: int, job_id: int) -> QueuePrepJobs:
    job = db.get(QueuePrepJobs, job_id)
    if job is None or job.dataset_id != dataset_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return job


@router.get("/datasets/{dataset_id}/prep-jobs/{job_id}")
async def get_prep_job(
        dataset_id: int,
        job_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_CREATE, "dataset_id")),
):
    """Progress of a queue preparation job."""
    job = _job_in_dataset(db, dataset_id, job_id)
    return {"success": True, "job": queue_prep.to_read(job).model_dump()}


@router.post("/datasets/{dataset_id}/prep-jobs/{job_id}/cancel")
async def cancel_prep_job(
        dataset_id: int,
        job_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_CREATE, "dataset_id")),
):
    """Stop a preparation job after the image it is working on."""
    job = _job_in_dataset(db, dataset_id, job_id)
    queue_prep.cancel(db, job)
    return {"success": True, "job": queue_prep.to_read(job).model_dump()}

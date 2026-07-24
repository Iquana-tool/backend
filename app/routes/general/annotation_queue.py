"""Annotation-queue endpoints: the persisted order an annotator works images in.

Unlike the review queue (a per-session snapshot), an annotation queue is saved per
(dataset, user): the builder on the Annotation card writes it, and re-entering the
editor resumes it. See ``app.services.annotation_queue`` for the ordering registry
(where active-learning orderings plug in).
"""
from logging import getLogger

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_session
from app.schemas.annotation_queue import AnnotationQueueRequest
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services import annotation_queue
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
    queue = annotation_queue.build_and_save_queue(dataset_id, user.username, body.strategy, db)
    return {
        "success": True,
        "message": f"Built an annotation queue of {queue.total} images.",
        "queue": queue.model_dump(mode="json"),
    }

"""Review endpoints: rejecting work, resolving rejections and listing them.

Approving is per contour and lives on the contour router (`/contours/{id}/reviews/add`);
rejecting is recorded here because a rejection can be about the mask as a whole
("objects are missing") rather than any single contour.
"""
from logging import getLogger

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.contours import Contours
from app.database.rejections import AnnotationRejections
from app.database.users import Users
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.schemas.review import (
    CorrectionQueueRequest,
    RejectionCreate,
    RejectionResolve,
    ReviewQueueRequest,
)
from app.services import correction_queue, review_queue
from app.services.auth import get_current_user
from app.services.database_access import contours as contours_db
from app.services.database_access import rejections as rejections_db
from app.services.permissions import ensure_permission_for, require

router = APIRouter(prefix="/reviews", tags=["reviews"])
logger = getLogger(__name__)


@router.get("/reasons")
async def get_rejection_reasons(user: AuthenticatedUser = Depends(get_current_user)):
    """The predefined rejection reasons, with display labels for the UI."""
    return {
        "success": True,
        "reasons": [option.model_dump(mode="json") for option in rejections_db.reason_options()],
    }


@router.get("/datasets/{dataset_id}/summary")
async def get_review_summary(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.REVIEW_APPROVE, "dataset_id")),
):
    """How much review work the dataset holds, plus the available queue orderings.

    Backs the "There are x instances to review" line on the management card and
    the strategy dropdown on the review setup page.
    """
    summary = review_queue.summarize(dataset_id, db)
    return {
        "success": True,
        "message": f"Review summary for dataset {dataset_id}.",
        "summary": summary.model_dump(mode="json"),
    }


@router.post("/datasets/{dataset_id}/queue")
async def build_review_queue(
        dataset_id: int,
        body: ReviewQueueRequest,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.REVIEW_APPROVE, "dataset_id")),
):
    """Build the ordered work list for one review session.

    The queue is a snapshot, not a reservation: nothing is locked, and an item
    someone else handles mid-session simply no-ops when acted on. See
    `app.services.review_queue` for the granularities and the sort-strategy
    registry (where active-learning orderings plug in).
    """
    queue = review_queue.build_queue(dataset_id, body, db)
    return {
        "success": True,
        "message": f"Built a review queue of {queue.total} items.",
        "queue": queue.model_dump(mode="json"),
    }


@router.get("/datasets/{dataset_id}/correction-summary")
async def get_correction_summary(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN, "dataset_id")),
):
    """How much correction work the dataset holds — the count behind the "x
    instances sent back for correction" line on the management card."""
    summary = correction_queue.summarize(dataset_id, db)
    return {
        "success": True,
        "message": f"Correction summary for dataset {dataset_id}.",
        "summary": summary.model_dump(mode="json"),
    }


@router.post("/datasets/{dataset_id}/correction-queue")
async def build_correction_queue(
        dataset_id: int,
        body: CorrectionQueueRequest,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN, "dataset_id")),
):
    """Build the ordered work list for one correction session.

    Every open rejection in the dataset qualifies. The queue is a snapshot, not a
    reservation: an item someone else resolves mid-session simply no-ops when acted
    on. See `app.services.correction_queue` for ordering and grouping.
    """
    queue = correction_queue.build_queue(dataset_id, body, db)
    return {
        "success": True,
        "message": f"Built a correction queue of {queue.total} items.",
        "queue": queue.model_dump(mode="json"),
    }


@router.post("/masks/{mask_id}/approve")
async def approve_mask(
        mask_id: int,
        include_reviewed: bool = False,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.REVIEW_APPROVE, "mask_id")),
):
    """Approve every not-yet-reviewed contour of a mask at once.

    This is the image-level "Accept" of the review queue. With `include_reviewed`
    the caller's approval is also added to contours other reviewers already signed
    off on (matching a queue built with the same flag) — approvals are additive,
    never replaced. Separation of duties still applies per contour: the caller's
    own annotations are skipped rather than self-approved, and reported back in
    `skipped`.
    """
    untouched_by_caller = (
        ~Contours.reviewed_by.any(Users.username == user.username)
        if include_reviewed else ~Contours.reviewed_by.any()
    )
    pending = (
        db.query(Contours.id)
        .filter(Contours.mask_id == mask_id,
                Contours.temporary.is_(False),
                untouched_by_caller)
        .all()
    )
    approved, skipped = [], []
    for (contour_id,) in pending:
        if await contours_db.review_contour(contour_id, user, db, strict=False):
            approved.append(contour_id)
        else:
            skipped.append(contour_id)
    return {
        "success": True,
        "message": f"Approved {len(approved)} contours on mask {mask_id}"
                   + (f", skipped {len(skipped)} of your own." if skipped else "."),
        "approved": approved,
        "skipped": skipped,
    }


@router.post("/masks/{mask_id}/reject")
async def reject_mask(
        mask_id: int,
        body: RejectionCreate,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.REVIEW_REJECT, "mask_id")),
):
    """Send a mask back to its annotator with a reason.

    Set `contour_id` in the body to complain about one object; omit it for a
    mask-level problem. Rejecting clears `fully_annotated`, so the mask moves out
    of the review queue and back into the annotator's work list; the mask's status
    becomes `rejected` until every open rejection is resolved.
    """
    rejection = await rejections_db.reject(mask_id, body, username=user.username, db=db)
    return {
        "success": True,
        "message": f"Mask {mask_id} rejected ({rejection.reason}).",
        "rejection": rejections_db.to_read(rejection).model_dump(mode="json"),
    }


@router.get("/masks/{mask_id}/rejections")
async def list_mask_rejections(
        mask_id: int,
        open_only: bool = False,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ, "mask_id")),
):
    """Rejections on a mask, newest first.

    Readable by annotators as well as reviewers — an annotator has to be able to
    see why their work came back.
    """
    rejections = await rejections_db.list_for_mask(mask_id, db, open_only=open_only)
    return {
        "success": True,
        "message": f"Retrieved {len(rejections)} rejections for mask {mask_id}.",
        "rejections": [rejection.model_dump(mode="json") for rejection in rejections],
    }


@router.patch("/rejections/{rejection_id}/resolve")
async def resolve_rejection(
        rejection_id: int,
        body: RejectionResolve | None = None,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(get_current_user),
):
    """Mark one rejection as addressed.

    Resolving needs `annotation.edit_own` on the dataset — the annotator fixing
    the problem is the one who closes it, not only the reviewer who raised it.
    Resolving deliberately does not require an edit: "I looked, the annotation is
    correct as it is" is a legitimate resolution. The optional body's `resolution`
    records how it was closed — the correction queue sends `fixed` for "Mark as
    done" and `wont_fix` for "Won't fix".
    """
    rejection = db.query(AnnotationRejections).filter_by(id=rejection_id).first()
    if rejection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rejection not found.")
    ensure_permission_for(user, "mask_id", rejection.mask_id, Permission.ANNOTATION_EDIT_OWN, db)

    resolved = await rejections_db.resolve(
        rejection_id, username=user.username, db=db,
        resolution=body.resolution if body else None,
    )
    return {
        "success": True,
        "message": f"Rejection {rejection_id} resolved.",
        "rejection": rejections_db.to_read(resolved).model_dump(mode="json"),
    }


@router.patch("/masks/{mask_id}/rejections/resolve")
async def resolve_all_rejections(
        mask_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN, "mask_id")),
):
    """Clear every open rejection on a mask, e.g. after reworking it."""
    count = await rejections_db.resolve_all_for_mask(mask_id, username=user.username, db=db)
    return {
        "success": True,
        "message": f"Resolved {count} open rejections on mask {mask_id}.",
        "resolved_count": count,
    }

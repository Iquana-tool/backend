"""Review endpoints: rejecting work, resolving rejections and listing them.

Approving is per contour and lives on the contour router (`/contours/{id}/reviews/add`);
rejecting is recorded here because a rejection can be about the mask as a whole
("objects are missing") rather than any single contour.
"""
from logging import getLogger

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.rejections import AnnotationRejections
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.schemas.review import RejectionCreate
from app.services.auth import get_current_user
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
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(get_current_user),
):
    """Mark one rejection as addressed.

    Resolving needs `annotation.edit_own` on the dataset — the annotator fixing
    the problem is the one who closes it, not only the reviewer who raised it.
    """
    rejection = db.query(AnnotationRejections).filter_by(id=rejection_id).first()
    if rejection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rejection not found.")
    ensure_permission_for(user, "mask_id", rejection.mask_id, Permission.ANNOTATION_EDIT_OWN, db)

    resolved = await rejections_db.resolve(rejection_id, username=user.username, db=db)
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

"""Undo / redo endpoints for the annotation workspace.

Deliberately REST rather than WebSocket messages. The annotation socket's message
types are an enum in the shared `iquana_toolbox` package, where UNDO and REDO sit
commented out; adding them means shipping that package first. These three routes
need nothing from the socket -- they act on the database and hand back the mask's
whole contour hierarchy, which is the same payload the socket's OBJECTS message
carries, so the client refreshes through the code path it already has.

Returning the full hierarchy rather than a delta is the point: after an undo the
client's object list must match the database exactly, and one refresh path cannot
drift the way a set of hand-applied inverse deltas would.
"""
from logging import getLogger

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_session
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services.database_access import annotation_history as history_db
from app.services.database_access import masks as masks_db
from app.services.permissions import require

router = APIRouter(prefix="/annotation-history", tags=["annotation history"])
logger = getLogger(__name__)


def _mask_id_or_404(image_id: int, db: Session) -> int:
    mask_id = history_db.mask_id_for_image(image_id, db)
    if mask_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Image {image_id} has no mask to undo work on.")
    return mask_id


@router.get("/{image_id}")
async def get_history_status(
        image_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN, "image_id")),
):
    """Whether this annotator has anything to undo or redo on this image.

    Backs the enabled state and the tooltip of the toolbar's two buttons, so the
    user is never offered an undo that would immediately fail.
    """
    mask_id = _mask_id_or_404(image_id, db)
    return {"success": True, **history_db.get_status(mask_id, user.username, db)}


async def _apply(image_id: int, db: Session, user: AuthenticatedUser, redo: bool):
    """Shared body of the two mutating routes."""
    mask_id = _mask_id_or_404(image_id, db)
    try:
        result = (history_db.redo if redo else history_db.undo)(mask_id, user.username, db)
    except history_db.HistoryError as exc:
        # A 409 rather than a 400: the request was well formed, the world just moved
        # on (the object was deleted by someone else, the label is gone).
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    _, payload = await masks_db.get_cached_contour_hierarchy_of_mask(mask_id, db)
    return {
        "success": True,
        "message": result["message"],
        "action_type": result["action_type"],
        "contours": payload,
        **history_db.get_status(mask_id, user.username, db),
    }


@router.post("/{image_id}/undo")
async def undo_last_action(
        image_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN, "image_id")),
):
    """Revert this annotator's most recent action on this image."""
    return await _apply(image_id, db, user, redo=False)


@router.post("/{image_id}/redo")
async def redo_last_action(
        image_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN, "image_id")),
):
    """Re-apply the action this annotator most recently undid on this image."""
    return await _apply(image_id, db, user, redo=True)

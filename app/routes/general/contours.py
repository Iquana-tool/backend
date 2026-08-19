from logging import getLogger

from fastapi import APIRouter
from fastapi import Depends, HTTPException, status
from iquana_toolbox.schemas.database.contours import Contour
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.contours import Contours
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services.database_access import annotation_history as history_db
from app.services.database_access import contours as contours_db
from app.services.permissions import ensure_permission_for, require

router = APIRouter(prefix="/contours", tags=["contours"])
logger = getLogger(__name__)


async def _ensure_may_edit(contour_id: int, user: AuthenticatedUser, db: Session) -> Contours:
    """Fetch a contour, allowing the edit only if the caller owns it or may edit any.

    Annotators hold `annotation.edit_own` and may only touch their own work;
    reviewers and above hold `annotation.edit_any` and may correct anyone's.
    """
    contour = db.query(Contours).filter_by(id=contour_id).first()
    if contour is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Contour not found.")

    dataset_id = ensure_permission_for(user, "contour_id", contour_id,
                                       Permission.ANNOTATION_EDIT_OWN, db)
    is_author = contour.author_username == user.username
    if not is_author and not user.has_permission(dataset_id, Permission.ANNOTATION_EDIT_ANY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This annotation was created by someone else and you may only edit your own.",
        )
    return contour


@router.get("/{contour_id}")
async def get_contour(
        contour_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ, "contour_id"))
):
    try:
        return {
            "success": True,
            "message": "Contour retrieved successfully.",
            "contour": await contours_db.get_contour(contour_id, db)
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Contour not found.")


@router.patch("/{contour_id}")
async def modify_contour(
        contour_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN, "contour_id")),
        **kwargs
):
    """
    Edit a contour by updating its coordinates or label.

    Args:
        contour_id (int): The ID of the contour to edit.
        db (Session): The database session.
        user (AuthenticatedUser): Caller, checked for permission to edit this contour.
        **kwargs: Arbitrary keyword arguments to update the contour attributes.

    Returns:
        dict: A dictionary containing the success status, message, and the ID of the edited contour.
    """
    contour = await _ensure_may_edit(contour_id, user, db)
    previous_label_id, mask_id = contour.label_id, contour.mask_id
    # `reviewed_by` is a review action, not an edit: setting it here would let an
    # annotator approve their own work through the back door.
    kwargs.pop("reviewed_by", None)
    kwargs.pop("author_username", None)
    modified = await contours_db.modify_contour(contour_id, db, **kwargs)
    if modified and "label_id" in kwargs:
        history_db.record_label_change(db, mask_id, user.username, contour_id,
                                       previous_label_id, kwargs["label_id"])
    return {
        "success": modified,
        "message": "Contour updated successfully." if modified else "Contour could not be updated.",
    }


@router.put("/{contour_id}")
async def replace_contour(
        contour_id: int,
        new_contour: Contour,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN, "contour_id"))
):
    """ Replace a contour with a new one. """
    await _ensure_may_edit(contour_id, user, db)
    replaced = await contours_db.replace_contour(contour_id, new_contour, db)
    return {
        "success": replaced,
        "message": "Successfully replaced contour." if replaced else "Could not replace contour.",
    }


@router.patch("/{contour_id}/label")
async def change_contour_label(
        contour_id: int,
        new_label_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_OWN, "contour_id"))
):
    """
    Edit the label of a contour, marking it reviewed by the caller where allowed.

    The automatic approval only happens for callers who actually hold
    `review.approve` and are permitted to review this contour; for everyone else
    the label change still succeeds, it just does not count as a review. Before,
    any label edit self-approved the contour, which made a mask reach "finished"
    without anyone else ever looking at it.

    Args:
        contour_id (int): The ID of the contour to edit.
        new_label_id (int): The new label ID to set for the contour.
        db (Session): The database session.
        user (AuthenticatedUser): Caller, checked for permission to edit this contour.

    Returns:
        dict: A dictionary containing the success status and whether a review was recorded.
    """
    dataset_id = ensure_permission_for(user, "contour_id", contour_id,
                                       Permission.ANNOTATION_EDIT_OWN, db)
    contour = await _ensure_may_edit(contour_id, user, db)
    previous_label_id, mask_id = contour.label_id, contour.mask_id

    # 1. Change the label_id, this checks if the new label is valid
    await contours_db.modify_contour(contour_id, label_id=new_label_id, db=db)
    history_db.record_label_change(db, mask_id, user.username, contour_id,
                                   previous_label_id, new_label_id)

    # 2. Record a review only if this caller is entitled to give one
    reviewed = False
    if user.has_permission(dataset_id, Permission.REVIEW_APPROVE):
        reviewed = await contours_db.review_contour(contour_id, user, db, strict=False)

    return {
        "success": True,
        "message": "Contour updated successfully.",
        "reviewed": reviewed,
    }


def _reviewers_of(contour_id: int, db: Session) -> list[str]:
    """Current reviewer list, so the client can render state it did not invent."""
    contour = db.query(Contours).filter_by(id=contour_id).first()
    return [reviewer.username for reviewer in contour.reviewed_by] if contour else []


@router.post("/{contour_id}/reviews/add")
async def add_contour_review(
        contour_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.REVIEW_APPROVE, "contour_id")),
):
    """ Mark a contour as reviewed by adding the current user to reviewed_by list."""
    try:
        await contours_db.review_contour(contour_id, user, db, strict=True)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return {
        "success": True,
        "message": f"Contour {contour_id} marked as reviewed successfully.",
        "reviewed_by": _reviewers_of(contour_id, db),
    }


@router.delete("/{contour_id}/reviews/remove")
async def remove_contour_review(
        contour_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.REVIEW_APPROVE, "contour_id"))
):
    """ Withdraw the caller's own approval of a contour."""
    await contours_db.remove_review(contour_id, user, db)

    return {
        "success": True,
        "message": f"User removed from reviewer of contour {contour_id}.",
        "reviewed_by": _reviewers_of(contour_id, db),
    }


@router.delete("/{contour_id}/reviews")
async def remove_all_contour_reviews(
        contour_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.REVIEW_REVOKE, "contour_id"))
):
    """ Remove everyone's approval of a contour, sending it back for review. """
    contour = db.query(Contours).filter_by(id=contour_id).first()
    if contour is None:
        raise HTTPException(status_code=404, detail="Contour not found.")

    contour.reviewed_by = []
    db.commit()

    return {
        "success": True,
        "message": f"Removed all reviewers from contour {contour_id}.",
    }


@router.delete("/{contour_id}")
async def delete_contour(
        contour_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_DELETE_OWN, "contour_id"))
):
    """
    Delete a contour and all its descendants (via CASCADE).
    Returns the list of deleted contour IDs.
    """
    contour = await _ensure_may_edit(contour_id, user, db)
    mask_id = contour.mask_id
    # Taken before the delete: the CASCADE removes the descendants too, and undo
    # has to be able to bring the whole subtree back.
    snapshot = history_db.snapshot_subtree(contour_id, db)
    await contours_db.delete_contour(contour_id, db)
    history_db.record_delete(db, mask_id, user.username, snapshot)

    # The descendants go with the CASCADE, so a caller holding its own list of
    # this mask's contours (the review queue does) needs to know which ids are
    # gone, not just that the root one is.
    deleted_ids = [entry["id"] for entry in (snapshot or {}).get("contours", [])]
    return {
        "success": True,
        "message": "Contour and descendants deleted successfully.",
        "deleted_ids": deleted_ids,
    }

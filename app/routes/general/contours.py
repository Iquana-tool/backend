from logging import getLogger

from fastapi import APIRouter
from fastapi import Depends, HTTPException
from iquana_toolbox.schemas.contours import Contour
from iquana_toolbox.schemas.user import User
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.contours import Contours
from app.services.auth import get_current_user
from app.services.database_access import contours as contours_db

router = APIRouter(prefix="/contours", tags=["contours"])
logger = getLogger(__name__)


@router.get("/{contour_id}")
async def get_contour(
        contour_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
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
        contour_id,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user),
        **kwargs
):
    """
    Edit a contour by updating its coordinates or label.

    Args:
        contour_id (int): The ID of the contour to edit.
        db (Session): The database session.
        user (User): Authentication dependency.
        **kwargs: Arbitrary keyword arguments to update the contour attributes.

    Returns:
        dict: A dictionary containing the success status, message, and the ID of the edited contour.
    """
    modified = await contours_db.modify_contour(contour_id, db, **kwargs)
    return {
        "success": modified,
        "message": "Contour updated successfully." if modified else "Contour could not be updated.",
    }


@router.put("/{contour_id}")
async def replace_contour(
        contour_id,
        new_contour: Contour,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """ Replace a contour with a new one. """
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
        user: User = Depends(get_current_user)
):
    """
    Edit the label of a contour and automatically mark it as reviewed by the current user.

    Args:
        contour_id (int): The ID of the contour to edit.
        new_label_id (int): The new label ID to set for the contour.
        db (Session): The database session.
        user (User): Authentication dependency.

    Returns:
        dict: A dictionary containing the success status, message, and the ID of the edited contour.
    """
    # 1. Change the label_id, this checks if the new label is valid
    await contours_db.modify_contour(contour_id, label_id=new_label_id, db=db)

    # 2. Add the user to the reviewed list
    # Get current reviewed_by users and add the current user if not already there
    await contours_db.review_contour(contour_id, user, db)

    # Update both label and reviewed_by
    return {
        "success": True,
        "message": "Contour updated successfully.",
    }


@router.post("/{contour_id}/reviews/add")
async def add_contour_review(
        contour_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user),
):
    """ Mark a contour as reviewed by adding the current user to reviewed_by list."""
    await contours_db.review_contour(contour_id, user, db)

    return {
        "success": True,
        "message": f"Contour {contour_id} marked as reviewed successfully."
    }


@router.delete("/{contour_id}/reviews/remove")
async def remove_contour_review(
        contour_id: int,
        user: User = Depends(get_current_user),
        db: Session = Depends(get_session)
):
    """ Mark a contour as reviewed by adding the current user to reviewed_by list."""
    await contours_db.remove_review(contour_id, user, db)

    return {
        "success": True,
        "message": f"User removed from reviewer of contour {contour_id}.",
    }


@router.delete("/{contour_id}/reviews")
async def remove_all_contour_reviews(contour_id: int,
                                     user: User = Depends(get_current_user),
                                     db: Session = Depends(get_session)):
    """ Remove all reviews. Admin feature. """
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
async def delete_contour(contour_id: int,
                         user: User = Depends(get_current_user),
                         db: Session = Depends(get_session)):
    """
    Delete a contour and all its descendants (via CASCADE).
    Returns the list of deleted contour IDs.
    """
    await contours_db.delete_contour(contour_id, db)

    return {
        "success": True,
        "message": "Contour and descendants deleted successfully.",
    }

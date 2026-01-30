from logging import getLogger

from fastapi import APIRouter
from fastapi import Depends, HTTPException
from schemas.contours import Contour
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.contours import Contours, save_contour_tree
from app.database.users import Users
from schemas.user import User
from app.services.auth import get_current_user

router = APIRouter(prefix="/contours", tags=["contours"])
logger = getLogger(__name__)


@router.get("/{contour_id}")
async def get_contour(contour_id: int, db: Session = Depends(get_session), user: User = Depends(get_current_user)):
    existing_contour = db.query(Contours).filter_by(id=contour_id).first()
    if not existing_contour:
        raise HTTPException(status_code=404, detail="Contour not found.")
    return {
        "success": True,
        "message": "Contour retrieved successfully.",
        "contour": Contour.from_db(existing_contour)
    }


@router.patch("/{contour_id}")
async def modify_contour(contour_id,
                         db: Session = Depends(get_session),
                         user: User = Depends(get_current_user),
                         **kwargs):
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
    try:
        existing_contour = db.query(Contours).filter_by(id=contour_id).first()
        if not existing_contour:
            raise HTTPException(status_code=404, detail="Contour not found.")

        for key, value in kwargs.items():
            if hasattr(existing_contour, key):
                if key == "label":
                    # Update the label - validation for parent/child label consistency can be added here if needed
                    setattr(existing_contour, key, value)
                elif key == "reviewed_by":
                    # Handle reviewed_by
                    if value is not None:
                        # Convert list of usernames to User instances
                        user_instances = []
                        for username in value:
                            user = db.query(Users).filter_by(username=username).first()
                            if user:
                                user_instances.append(user)
                            else:
                                logger.warning(f"User {username} not found when adding to reviewed_by")
                        existing_contour.reviewed_by = user_instances
                else:
                    setattr(existing_contour, key, value)

        db.commit()
        return {
            "success": True,
            "message": "Contour edited successfully.",
            "contour": Contour.from_db(existing_contour)
        }
    except Exception as e:
        logger.error(f"Error modifying contour: {e}")
        db.rollback()
        raise e


@router.put("/{contour_id}")
async def replace_contour(contour_id,
                          new_contour: Contour,
                          user: User = Depends(get_current_user),
                          db: Session = Depends(get_session)):
    """ Replace a contour with a new one. """
    new_contour.id = contour_id
    contour = db.query(Contours).filter_by(id=contour_id).first()
    db.query(Contours).filter_by(id=contour_id).delete()
    save_contour_tree(db, new_contour, contour.mask_id, contour.parent_id)
    db.commit()
    return {
        "success": True,
        "message": "Successfully replaced contour.",
    }


@router.patch("/{contour_id}/label")
async def change_contour_label(contour_id: int, new_label_id: int,
                               user: User = Depends(get_current_user),
                               db: Session = Depends(get_session)):
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
    # Get existing contour to check current reviewers
    existing_contour = db.query(Contours).filter_by(id=contour_id).first()
    if not existing_contour:
        raise HTTPException(status_code=404, detail="Contour not found.")

    # Get current reviewed_by users and add the current user if not already there
    reviewed_by_usernames = [u.username for u in existing_contour.reviewed_by]
    if user.username not in reviewed_by_usernames:
        reviewed_by_usernames.append(user.username)

    # Update both label and reviewed_by
    return await modify_contour(contour_id, label_id=new_label_id, reviewed_by=reviewed_by_usernames, db=db)


@router.post("/{contour_id}/reviews/add")
async def add_contour_review(contour_id: int,
                             user: User = Depends(get_current_user),
                             db: Session = Depends(get_session)):
    """ Mark a contour as reviewed by adding the current user to reviewed_by list."""
    contour = db.query(Contours).filter_by(id=contour_id).first()
    if contour is None:
        raise HTTPException(status_code=404, detail="Contour not found.")

    # Only add user if not already in reviewed_by list
    if user not in contour.reviewed_by:
        contour.reviewed_by.append(user)
        db.commit()

    return {
        "success": True,
        "message": f"Contour {contour_id} marked as reviewed successfully.",
        "reviewed_by": [u.username for u in contour.reviewed_by],
    }


@router.delete("/{contour_id}/reviews/remove")
async def remove_contour_review(contour_id: int,
                                user: User = Depends(get_current_user),
                                db: Session = Depends(get_session)):
    """ Mark a contour as reviewed by adding the current user to reviewed_by list."""
    contour = db.query(Contours).filter_by(id=contour_id).first()
    if contour is None:
        raise HTTPException(status_code=404, detail="Contour not found.")

    # Only add user if not already in reviewed_by list
    if user in contour.reviewed_by:
        contour.reviewed_by.remove(user)
        db.commit()

    return {
        "success": True,
        "message": f"User removed from reviewer of contour {contour_id}.",
        "reviewed_by": [u.username for u in contour.reviewed_by],
    }


@router.delete("/{contour_id}/reviews")
async def remove_all_contour_reviews(contour_id: int,
                                     user: User = Depends(get_current_user),
                                     db: Session = Depends(get_session)):
    """ Mark a contour as reviewed by adding the current user to reviewed_by list."""
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
    # Fetch the contour and all descendants in one query
    contour = (
        db.query(Contours)
        .filter_by(id=contour_id)
        .first()
    )
    if not contour:
        raise HTTPException(status_code=404, detail="Contour not found.")

    # Delete the root contour (CASCADE will handle the rest)
    db.delete(contour)
    db.commit()

    return {
        "success": True,
        "message": "Contour and descendants deleted successfully.",
    }

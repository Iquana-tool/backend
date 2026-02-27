import logging

from fastapi import APIRouter, Depends
from iquana_toolbox.schemas.contours import Contour
from iquana_toolbox.schemas.user import User
from sqlalchemy.orm import Session

from app.database import get_session
from app.services.auth import get_current_user
from app.services.database_access import masks as masks_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/masks", tags=["masks"])


@router.get("/{mask_id}")
async def get_mask(
        mask_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """ Get a mask by its ID.

    Args:
        mask_id (int): The ID of the mask.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and the mask.
    """

    return {
        "success": True,
        "mask": await masks_db.get_mask(mask_id, db)
    }


@router.get("/{mask_id}/status")
async def get_mask_annotation_status(
        mask_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """ Check the annotation status of a mask by its ID.

    Args:
        mask_id (int): The ID of the mask.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the annotation status.
    """
    return {
        "success": True,
        "message": "Mask status retrieved successfully.",
        "status": (await masks_db.get_mask(mask_id, db)).status
    }


@router.delete("/{mask_id}")
async def delete_mask(
        mask_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """ Delete a mask and all its contours by its ID.

    Args:
        mask_id (int): The ID of the mask.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    await masks_db.delete_mask(mask_id, db)
    return {
        "success": True,
        "message": "Mask deleted successfully."
    }


@router.patch("/{mask_id}/status/complete")
async def mark_as_fully_annotated(
        mask_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """ Mark a mask as finished, generate it as an image file and upload it to the AI external service.

    Args:
        mask_id (int): The ID of the mask.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and mask ID.
    """
    await masks_db.mark_mask_as_complete(mask_id, db)
    return {
        "success": True,
        "message": "Mask marked as finished successfully.",
    }


@router.patch("/{mask_id}/status/incomplete")
async def unmark_as_fully_annotated(
        mask_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """ Remove the finished status from a mask, allowing it to be edited again. This will also delete the mask image
        file and remove it from the AI external service.

    Args:
        mask_id (int): The ID of the mask.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and mask ID.
    """
    # Check if mask exists
    await masks_db.mark_mask_as_incomplete(mask_id, db)
    return {
        "success": True,
        "message": "Mask marked as not fully annotated successfully.",
    }


@router.get("/{mask_id}/contours")
async def get_contours_of_mask(
        mask_id: int,
        flattened: bool = True,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """ Export quantification data for the given mask_id and labels.

    Args:
        mask_id (int): The ID of the mask to export contours for.
        flattened (bool): Whether to flatten the hierarchical JSON structure. Defaults to True. If False, the
            hierarchical structure will be preserved, i.e. children contours will be nested under their
            parent contour.
        db (Session, optional): The database session. Defaults to Depends(get_session). This is a fastapi dependency.
        user (User): Authentication dependency.

    Returns:
        dict: A dictionary containing the success status and message if error, or a hierarchical JSON structure of
        contours for the given mask_id.
    """
    hierarchy = await masks_db.get_contour_hierarchy_of_mask(mask_id, db)
    return {
        "success": True,
        "message": f"Contours {'hierarchy' if not flattened else ''} retrieved.",
        "contours": hierarchy.model_dump() if not flattened else hierarchy.dump_contours_as_list()
    }


@router.put("/{mask_id}/contours")
async def add_contour(
        mask_id: int,
        contour_to_add: Contour,
        check_hierarchy: bool = True,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """
    Add a contour to a mask in the database.

    Args:
        mask_id (int): The ID of the mask to which the contour will be added.
        contour_to_add (Contour): The contour data to add.
        check_hierarchy (bool): Whether to check the hierarchy of the contour. Defaults to True. If true, fits the contour
            into the existing hierarchy.
        user (User): Authentication dependency.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and the ID of the added contour.
    """
    added_contour = await masks_db.add_contour_to_mask(mask_id, contour_to_add, check_hierarchy=check_hierarchy, db=db)
    return {
        "success": True,
        "message": "Contour added successfully.",
        "added_contour": added_contour.model_dump(),
    }


@router.put("/{mask_id}/contours/multi")
async def add_contours(
        mask_id: int,
        contours_to_add: list[Contour],
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """
    Add multiple contours to a mask in the database. Internally calls `add_contour` for each contour.

    Args:
        mask_id (int): The ID of the mask to which the contours will be added.
        contours_to_add (list[Contour]): A list of contour data to add.
        temporary_list (list[bool]): A list saying whether or not the contours should be temporary.
        user (User): Authentication dependency.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and lists of added and failed contour IDs.
    """
    hierarchy = await masks_db.get_contour_hierarchy_of_mask(mask_id, db)
    added = []
    for contour_to_add in contours_to_add:
        logger.info(f"Added {len(added)} / {len(contours_to_add)} contours.")
        # 1. Add to the hierarchy, ensuring it fits and respects hierarchies
        fitted_contour, changed = hierarchy.add_contour(contour_to_add)

        # 2. Add the (fitted) contour to the db; don't need to check the hierarchy here
        await masks_db.add_contour_to_mask(mask_id, fitted_contour, check_hierarchy=False, db=db)

        # 3. Add to a list for us to return
        added.append(fitted_contour)

    if len(added) < len(contours_to_add):
        return {
            "success": False,
            "message": f"Added {len(added)} contours. Failed to add all {len(contours_to_add)} contours.",
            "mask_id": mask_id,
            "added_contours": added,
        }
    else:
        return {
            "success": True,
            "message": "All contours added successfully.",
            "mask_id": mask_id,
            "added_contours": added,
        }


@router.delete("/{mask_id}/contours")
async def delete_all_contours_of_mask(
        mask_id: int,
        user: User = Depends(get_current_user),
        db: Session = Depends(get_session)
):
    """ Deletes all contours of a mask. """
    await masks_db.delete_all_contours_of_mask(mask_id, db)
    return {
        "success": True,
        "message": f"Deleted all contours of mask {mask_id}"
    }


@router.delete("/{mask_id}/contours/unreviewed")
async def delete_unreviewed_contours_of_mask(mask_id: int,
                                             user: User = Depends(get_current_user),
                                             db: Session = Depends(get_session)):
    """ Deletes all temporary contours of a mask. """
    await masks_db.delete_all_contours_of_mask(mask_id, unreviewed_only=True, db=db)
    return {
        "success": True,
        "message": f"Deleted all unreviewed contours of mask {mask_id}"
    }

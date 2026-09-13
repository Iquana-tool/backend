import logging

from fastapi import APIRouter, Depends, HTTPException, status
from iquana_toolbox.schemas.database.contours import Contour
from sqlalchemy.orm import Session

from app.database import get_session
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services import image_status
from app.services.database_access import masks as masks_db
from app.services.permissions import require

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/masks", tags=["masks"])


@router.get("/{mask_id}")
async def get_mask(
        mask_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ, "mask_id"))
):
    """ Get a mask by its ID.

    Args:
        mask_id (int): The ID of the mask.
        user (AuthenticatedUser): The current authenticated user.

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
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ, "mask_id"))
):
    """ Check the workflow status of a mask's image by the mask ID.

    Reports all three phases — ``calibrate``, ``annotate`` and ``review`` — each
    one of ``not_started``, ``in_progress`` or ``finished``, plus the combined
    ``status``, which is ``finished`` only when every phase is. A reviewer sending
    work back (an open rejection) pulls annotate and review back to ``in_progress``.

    Args:
        mask_id (int): The ID of the mask.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        dict: A dictionary containing the overall status and the per-phase breakdown.
    """
    mask = await masks_db.get_mask(mask_id, db)
    state = image_status.status_for_mask(db, mask)
    return {
        "success": True,
        "message": "Mask status retrieved successfully.",
        "status": state["status"],
        "phases": state["phases"],
    }


@router.delete("/{mask_id}")
async def delete_mask(
        mask_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.MASK_DELETE, "mask_id"))
):
    """ Delete a mask and all its contours by its ID.

    Args:
        mask_id (int): The ID of the mask.
        user (AuthenticatedUser): The current authenticated user.

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
        user: AuthenticatedUser = Depends(require(Permission.MASK_SUBMIT, "mask_id"))
):
    """ Submit a mask for review: mark it as containing every object.

    Args:
        mask_id (int): The ID of the mask.
        db (Session): The database session.
        user (AuthenticatedUser): The current authenticated user.

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
        user: AuthenticatedUser = Depends(require(Permission.MASK_REOPEN, "mask_id"))
):
    """ Reopen a submitted mask for editing.

    Requires `mask.reopen` rather than `mask.submit`: once work is in the review
    queue, pulling it back out is the reviewer's call, not the annotator's.

    Args:
        mask_id (int): The ID of the mask.
        db (Session): The database session.
        user (AuthenticatedUser): The current authenticated user.

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
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ, "mask_id"))
):
    """ Export quantification data for the given mask_id and labels.

    Args:
        mask_id (int): The ID of the mask to export contours for.
        flattened (bool): Whether to flatten the hierarchical JSON structure. Defaults to True. If False, the
            hierarchical structure will be preserved, i.e. children contours will be nested under their
            parent contour.
        db (Session, optional): The database session. Defaults to Depends(get_session). This is a fastapi dependency.
        user (AuthenticatedUser): Authentication dependency.

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
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_CREATE, "mask_id"))
):
    """
    Add a contour to a mask in the database.

    Args:
        mask_id (int): The ID of the mask to which the contour will be added.
        contour_to_add (Contour): The contour data to add.
        check_hierarchy (bool): Whether to check the hierarchy of the contour. Defaults to True. If true, fits the contour
            into the existing hierarchy.
        user (AuthenticatedUser): Authentication dependency.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and the ID of the added contour.
    """
    added_contour = await masks_db.add_contour_to_mask(mask_id, contour_to_add, check_hierarchy=check_hierarchy,
                                                       db=db, author_username=user.username)
    if added_contour is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Contour has no drawable pixels after hierarchy fitting.",
        )
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
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_CREATE, "mask_id"))
):
    """
    Add multiple contours to a mask in the database. Internally calls `add_contour` for each contour.

    Args:
        mask_id (int): The ID of the mask to which the contours will be added.
        contours_to_add (list[Contour]): A list of contour data to add.
        user (AuthenticatedUser): Authentication dependency.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and lists of added and failed contour IDs.
    """
    hierarchy = await masks_db.get_contour_hierarchy_of_mask(mask_id, db)
    added = []
    for contour_to_add in contours_to_add:
        logger.info(f"Added {len(added)} / {len(contours_to_add)} contours.")
        # 1. Add to the hierarchy, ensuring it fits and respects hierarchies
        fitted_contour = masks_db.add_contour_to_hierarchy(hierarchy, contour_to_add)
        if fitted_contour is None:
            continue

        # 2. Add the (fitted) contour to the db; don't need to check the hierarchy here
        await masks_db.add_contour_to_mask(mask_id, fitted_contour, check_hierarchy=False, db=db,
                                           author_username=user.username)

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
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_EDIT_ANY, "mask_id"))
):
    """ Deletes all contours of a mask. Wipes other people's work, so reviewer+. """
    await masks_db.delete_all_contours_of_mask(mask_id, db)
    return {
        "success": True,
        "message": f"Deleted all contours of mask {mask_id}"
    }


@router.delete("/{mask_id}/contours/unreviewed")
async def delete_unreviewed_contours_of_mask(
        mask_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.REVIEW_PURGE_UNREVIEWED, "mask_id"))
):
    """ Deletes every not-yet-approved contour of a mask. """
    await masks_db.delete_all_contours_of_mask(mask_id, unreviewed_only=True, db=db)
    return {
        "success": True,
        "message": f"Deleted all unreviewed contours of mask {mask_id}"
    }

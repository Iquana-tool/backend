import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.contours import Contours
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.schemas.contour_hierarchy import ContourHierarchy
from app.schemas.labels import LabelHierarchy
from app.schemas.user import User
from app.services.auth import get_current_user
from app.services.database_access import save_array_to_disk

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/masks", tags=["masks"])


@router.put("/create_mask/{image_id}")
async def create_mask(
    image_id: int,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """ Create a new mask for the given image ID. Only one mask can exist per image.

    Args:
        image_id (int): The ID of the image.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and mask ID.
    """
    try:
        # Check if mask already exists for the image
        existing_mask = db.query(Masks).filter_by(image_id=image_id).first()
        if existing_mask:
            return {
                "success": False,
                "message": "Mask already exists for this image.",
                "mask_id": existing_mask.id
            }
        # Create a new mask
        new_mask = Masks(image_id=image_id)
        db.add(new_mask)
        db.commit()
        return {
            "success": True,
            "message": "Mask created successfully.",
            "mask_id": new_mask.id
        }
    except Exception as e:
        logger.error(f"Error creating mask: {e}")
        raise HTTPException(status_code=500, detail="Error creating mask.")


@router.post("/mark_as_fully_annotated/{mask_id}")
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
    # Check if mask exists
    existing_mask = db.query(Masks).filter_by(id=mask_id).first()
    if not existing_mask:
        raise HTTPException(status_code=404, detail="Mask not found.")
    logger.info(f"Finishing this mask: {existing_mask}")
    # Check if the mask is already finished
    if bool(existing_mask.fully_annotated):
        return {
            "success": True,
            "message": "Mask is already marked as fully annotated.",
            "mask_id": existing_mask.id
        }
    image = db.query(Images).filter_by(id=existing_mask.image_id).first()
    # Generate the mask from contours
    contours = db.query(Contours).filter_by(mask_id=mask_id).all()
    contours_hierarchy = ContourHierarchy.from_contours(contours)
    semantic_mask = contours_hierarchy.to_semantic_mask(image.height, image.width)
 
    logging.debug(f"Generated mask with the following labels: {np.unique(semantic_mask).tolist()}")
    mask_path = save_array_to_disk(semantic_mask,
                       image.dataset_id,
                       image.scan_id,
                       is_mask=True,
                       new_filename=image.file_name)
    # Mark the mask as finished
    existing_mask.fully_annotated = True
    db.commit()
    return {
        "success": True,
        "message": "Mask marked as finished successfully.",
        "mask_id": existing_mask.id
    }


@router.post("/unmark_as_fully_annotated/{mask_id}")
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
    existing_mask = db.query(Masks).filter_by(id=mask_id).first()
    if not existing_mask:
        raise HTTPException(status_code=404, detail="Mask not found.")
    # Check if the mask is already unfinished
    if not existing_mask.fully_annotated:
        return {
            "success": True,
            "message": "Mask is not marked as fully annotated.",
            "mask_id": existing_mask.id
        }
    # Mark the mask as unfinished
    existing_mask.fully_annotated = False
    db.commit()
    return {
        "success": True,
        "message": "Mask marked as not fully annotated successfully.",
        "mask_id": existing_mask.id
    }


@router.get("/get_mask/{mask_id}")
async def get_mask(
    mask_id: int,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """ Get a mask by its ID.

    Args:
        mask_id (int): The ID of the mask.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and the mask.
    """
    mask = db.query(Masks).filter_by(id=mask_id).first()
    if mask is None:
        raise HTTPException(status_code=404, detail="Mask not found.")
    return {
        "success": True,
        "mask": mask
    }


@router.get("/get_mask_annotation_status/{mask_id}")
async def get_mask_annotation_status(
    mask_id: int,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """ Check the annotation status of a mask by its ID.

    Args:
        mask_id (int): The ID of the mask.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the annotation status.
    """
    mask = db.query(Masks).filter_by(id=mask_id).first()
    if mask is None:
        raise HTTPException(status_code=404, detail="Mask does not exist.")
    contours = db.query(Contours).filter_by(mask_id=mask.id).all()
    if len(contours) == 0:
        # Zero annotated objects means we have not started annotating yet
        return {
            "success": True,
            "message": "Mask status: Not started.",
            "status": "not_started",
        }
    elif not mask.fully_annotated:
        # The mask has not been marked as fully annotated, so annotation must still be in progress
        return {
            "success": True,
            "message": "Mask status: In progress.",
            "status": "in_progress",
        }
    elif np.any(len(contour.reviewed_by) == 0 for contour in contours):
        # Mask has been marked as fully annotated, but we still have contours without reviewers, so the mask is reviewable
        return {
            "success": True,
            "message": "Mask status: Reviewable.",
            "status": "reviewable",
        }
    else:
        # Mask marked as fully annotated and each contour has at least one reviewer, the mask is finished.
        return {
            "success": True,
            "message": "Mask status: Finished.",
            "status": "finished",
        }


@router.delete("/delete_mask/{mask_id}")
async def delete_mask(
    mask_id: int,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """ Delete a mask and all its contours by its ID.

    Args:
        mask_id (int): The ID of the mask.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    mask = db.query(Masks).filter_by(id=mask_id).first()
    if mask is None:
        return {
            "success": True,
            "message": "Mask does not exist.",
        }
    db.delete(mask)
    db.commit()
    return {
        "success": True,
        "message": "Mask deleted successfully."
    }


@router.post("/post_mask/mask_id={mask_id}&added_by={added_by}", deprecated=True)
async def post_mask(
    mask_id: int,
    added_by: str,
    mask: UploadFile = File(...),
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """
    Upload a mask to a mask id. Compute the contours for each label in the mask, build the hierarchy and add
    them to the database.

    Args:
        mask_id (int): The ID of the mask.
        added_by (str): Who added the mask.
        temporary (bool): Whether the mask is temporary.
        mask (UploadFile): The mask file.
        session (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and result.
    """
    mask_array = np.frombuffer(mask.file.read(), dtype=np.uint8)
    image_id = session.query(Masks.image_id).filter_by(id=mask_id).first()
    dataset_id = session.query(Images.dataset_id).filter_by(id=image_id).first()
    labels = session.query(Labels).filter_by(dataset_id=dataset_id)
    label_hierarchy = LabelHierarchy.from_query(labels)
 
    # Create an initial hierarchy of already added contours
    contour_hierarchy = ContourHierarchy.from_query(session.query(Contours).filter_by(mask_id=mask_id))
    # Add new contours from the mask
    contour_hierarchy = await contour_hierarchy.from_semantic_mask(
        mask_id,
        mask_array,
        label_hierarchy,
        added_by,
        session
    )
    return {
        "success": True,
        "message": "Converted mask object to contour hierarchy and added it to the database.",
        "result": contour_hierarchy.model_dump_json()
    }

import logging
import os
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from fastapi import APIRouter, HTTPException, Depends
from iquana_toolbox.schemas.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.contours import Contour
from iquana_toolbox.schemas.labels import LabelHierarchy
from iquana_toolbox.schemas.user import User
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse, FileResponse

from app.database import get_session
from app.database.contours import Contours, save_contour_tree
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.services.auth import get_current_user
from app.services.database_access import save_semantic_mask
from app.services.database_access.masks import get_contour_hierarchy_of_mask, get_size_of_mask
from app.services.util import get_mask_path_from_image_path

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


@router.get("/{mask_id}/status")
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
    return {
        "success": True,
        "message": "Mask status retrieved successfully.",
        "status": mask.status
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
    # Check if mask exists
    mask, dataset_id, width, height = db.query(
        Masks,
        Images.dataset_id,
        Images.width,
        Images.height,
    ).join(
        Images, Masks.image_id == Images.id
    ).filter_by(
        id=mask_id
    ).first()
    if not mask:
        raise HTTPException(status_code=404, detail="Mask not found.")
    logger.info(f"Finishing this mask: {mask}")
    # Check if the mask is already finished
    if mask.fully_annotated:
        return {
            "success": True,
            "message": "Mask is already marked as fully annotated.",
            "mask_id": mask.id
        }
    # Generate the mask from contours
    # Currently three separate queries, should be one
    contours = db.query(Contours).filter_by(mask_id=mask_id).all()
    contours_hierarchy = ContourHierarchy.from_query(contours, width, height)
    labels = db.query(Labels).filter_by(dataset_id=dataset_id)
    labels_hierarchy = LabelHierarchy.from_query(labels)
    semantic_mask = contours_hierarchy.to_semantic_mask(height, width, labels_hierarchy.id_to_value_map)
    await save_semantic_mask(semantic_mask, mask.file_path)

    # Mark the mask as finished
    mask.fully_annotated = True
    db.commit()
    return {
        "success": True,
        "message": "Mask marked as finished successfully.",
        "mask_id": mask.id
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


@router.get("/{mask_id}/contours")
async def get_contours_of_mask(mask_id: int,
                               flattened: bool = True,
                               db: Session = Depends(get_session),
                               user: User = Depends(get_current_user)):
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
    hierarchy = await get_contour_hierarchy_of_mask(mask_id, db)
    return {
        "success": True,
        "message": f"Contours {'hierarchy' if not flattened else ''} retrieved.",
        "contours": hierarchy.model_dump() if not flattened else hierarchy.dump_contours_as_list()
    }


@router.put("/{mask_id}/contours")
async def add_contour(mask_id: int,
                      contour_to_add: Contour,
                      check_parent: bool = False,
                      user: User = Depends(get_current_user),
                      db: Session = Depends(get_session)):
    """
    Add a contour to a mask in the database.

    Args:
        mask_id (int): The ID of the mask to which the contour will be added.
        contour_to_add (Contour): The contour data to add.
        user (User): Authentication dependency.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and the ID of the added contour.
    """
    hierarchy = await get_contour_hierarchy_of_mask(mask_id, db)
    added_contour, changed = hierarchy.add_contour(contour_to_add)
    # Add contour to the database
    entry = save_contour_tree(db, added_contour, mask_id)
    db.commit()
    added_contour.id = entry.id

    # SVG path computation for the frontend
    # Get image dimensions and compute path
    size = await get_size_of_mask(mask_id, db)
    added_contour.compute_path(
        image_width=size["width"],
        image_height=size["height"],
    )

    return {
        "success": True,
        "message": "Contour added successfully.",
        "added_contour": added_contour.model_dump(),
    }


@router.put("/{mask_id}/contours/multi")
async def add_contours(mask_id: int,
                       contours_to_add: list[Contour],
                       user: User = Depends(get_current_user),
                       db: Session = Depends(get_session)):
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
    added = []
    for contour_to_add in contours_to_add:
        logger.info(f"Added {len(added)} / {len(contours_to_add)} contours.")
        result = await add_contour(mask_id, contour_to_add, user, db)
        if result["success"]:
            added.append(result["added_contour"])
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
async def delete_all_contours_of_mask(mask_id: int,
                                      user: User = Depends(get_current_user),
                                      db: Session = Depends(get_session)):
    """ Deletes all contours of a mask. """
    db.query(Contours).filter_by(mask_id=mask_id).delete()
    mask = db.query(Masks).filter_by(id=mask_id).first()
    mask.fully_annotated = False
    db.commit()
    return {
        "success": True,
        "message": f"Deleted all contours of mask {mask_id}"
    }


@router.delete("/{mask_id}/contours/unreviewed")
async def delete_unreviewed_contours_of_mask(mask_id: int,
                                             user: User = Depends(get_current_user),
                                             db: Session = Depends(get_session)):
    """ Deletes all temporary contours of a mask. """
    try:
        contours = db.query(Contours).filter(Contours.mask_id == mask_id, ~Contours.reviewed_by.any()).delete()
        db.commit()
        return {
            "success": True,
            "message": f"Deleted all temporary contours of mask {mask_id}"
        }
    except Exception as e:
        logger.error(e)
        db.rollback()
        raise e


@router.get("/{mask_id}", deprecated=True)
async def get_mask_csv(
        mask_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """
    Download quantification data for the given mask_id as a CSV file.

    Args:
        mask_id (int): The ID of the mask.
        db (Session): The database session.
        user (User): The current authenticated user.
    """
    image_name = db.query(Images.file_name).join(Masks, Images.id == Masks.image_id).filter(Masks.id == mask_id).first()
    response = await get_contours_of_mask(mask_id, True, db)
    df = pd.DataFrame(response["contours"])
    csv_content = df.to_csv(index=False)
    response = StreamingResponse(StringIO(csv_content), media_type="text/csv")
    response.headers["Content-Disposition"] = f'attachment; filename="{image_name[0]}_quantifications.csv"'
    return response


@router.get("/get_segmentation_mask_file/{mask_id}")
async def get_segmentation_mask_file(
        mask_id: int,
        user: User = Depends(get_current_user),
        db: Session = Depends(get_session)
):
    """Download the prompted_segmentation mask file for the given mask_id."""
    mask = db.query(Masks).filter_by(id=mask_id).first()
    if not mask:
        return {
            "success": False,
            "message": "Mask not found."
        }
    elif not mask.fully_annotated:
        return {
            "success": False,
            "message": "Cannot download mask that is not fully annotated."
        }

    image = db.query(Images).filter_by(id=mask.image_id).first()
    file_path = get_mask_path_from_image_path(image.file_path) if image else None

    if not file_path or not os.path.exists(file_path):
        logger.error(f"Mask is finished but mask file not found at {file_path}.")
        raise HTTPException(status_code=404, detail="Mask file not found.")

    return FileResponse(
        file_path, media_type="image/png", filename=os.path.basename(file_path)
    )

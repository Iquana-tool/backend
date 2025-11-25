import logging
 
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from sqlalchemy.orm import Session
 
from app.database import get_session
from app.database.contours import Contours
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.routes.contours import delete_contour, add_contours
from app.routes.semantic_segmentation.upload_data import proxy_upload_file
from app.schemas.contours import ContourHierarchy
from app.schemas.labels import LabelHierarchy
from app.schemas.prompted_segmentation.segmentations import SemanticSegmentationMask
from app.services.database_access import save_array_to_disk
from app.schemas.user import User
from app.services.auth import get_current_user
 
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


@router.post("/finish_mask/{mask_id}")
async def finish_mask(
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
    print(f"Finishing this mask: {existing_mask}")
    # Check if the mask is already finished
    if bool(existing_mask.finished):
        return {
            "success": True,
            "message": "Mask is already marked as finished.",
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
    existing_mask.finished = True
    db.commit()
    # Upload the image and mask to the AI external service
    file_name, extension = image.file_name.rsplit(".", maxsplit=1)  # Get the file name without extension
    with open(image.file_path, "rb") as img_file:
        img_upload = UploadFile(file=img_file,
                                filename=file_name,
                                headers={"Content-Type": f'image/{extension}'})
        img_response = await proxy_upload_file(
            dataset_id=image.dataset_id,
            is_image=True,
            file=img_upload,
            filename=file_name
        )
 
    with open(mask_path, "rb") as mask_file:
        mask_upload = UploadFile(file=mask_file,
                                 filename=file_name,
                                 headers={"Content-Type": f'image/{extension}'})
        mask_response = await proxy_upload_file(
            dataset_id=image.dataset_id,
            is_image=False,
            file=mask_upload,
            filename=file_name
        )
    return {
        "success": True,
        "message": "Mask marked as finished successfully.",
        "mask_id": existing_mask.id
    }


@router.post("/unfinish_mask/{mask_id}")
async def unfinish_mask(
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
    # TODO: Implement deletion of mask image file and removal from AI external service.
    # Check if mask exists
    existing_mask = db.query(Masks).filter_by(id=mask_id).first()
    if not existing_mask:
        raise HTTPException(status_code=404, detail="Mask not found.")
    # Check if the mask is already unfinished
    if not existing_mask.finished:
        return {
            "success": True,
            "message": "Mask is not marked as finished.",
            "mask_id": existing_mask.id
        }
    # Mark the mask as unfinished
    existing_mask.finished = False
    db.commit()
    return {
        "success": True,
        "message": "Mask marked as unfinished successfully.",
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
        raise HTTPException(status_code=404, detail="Mask not found.")
 
    # Check if the mask is finished
    if mask.finished:
        return {
            "success": True,
            "message": "Mask is finished.",
            "status": "manually_annotated",
            "mask_id": mask.id
        }
 
    # Check if the mask is generated
    if mask.generated:
        return {
            "success": True,
            "message": "Mask is generated but not finished.",
            "status": "auto_annotated",
            "mask_id": mask.id
        }
 
    return {
        "success": True,
        "message": "Mask is neither finished nor generated.",
        "status": "missing",
        "mask_id": mask.id
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
        raise HTTPException(status_code=404, detail="Mask not found.")
    root_contours = db.query(Contours).filter_by(mask_id=mask_id, parent_id=None).all()
    for contour in root_contours:
        await delete_contour(contour.id, user, db)
    db.delete(mask)
    db.commit()
    return {"success": True, "message": "Mask deleted successfully."}


@router.post("/post_mask/mask_id={mask_id}&added_by={added_by}&temporary={temporary}", deprecated=True)
async def post_mask(
    mask_id: int,
    added_by: str,
    temporary: bool,
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
        temporary,
        session
    )
    return {
        "success": True,
        "message": "Converted mask object to contour hierarchy and added it to the database.",
        "result": contour_hierarchy.model_dump_json()
    }



async def create_masks_and_add_contours_for_images(image_ids: list[int],
                                                   mask_responses: list[SemanticSegmentationMask],
                                                   db: Session = Depends(get_session)):
    """
    Create masks for a list of image IDs and add contours to them.

    Args:
        image_ids (list[int]): List of image IDs for which to create masks.
        mask_responses (list[SemanticSegmentationMask]): List of prompted_segmentation mask responses containing contours.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and responses for each image.
    """
    if len(image_ids) != len(mask_responses):
        raise ValueError(
            f"Number of image_ids does not match number of mask_responses."
        )
    responses = []
    for image_id, mask_response in zip(image_ids, mask_responses):
        mask = db.query(Masks).filter_by(image_id=image_id).first()
        if not mask:
            response = await create_mask(image_id, db)
            mask = db.query(Masks).filter_by(image_id=image_id).first()
        responses.append(await add_contours(mask.id, mask_response.contours, None, db))
        mask.generated = True
        db.commit()
    return {
        "success": True,
        "message": f"Created and added masks for {len(image_ids)} images.",
        "responses": responses
    }

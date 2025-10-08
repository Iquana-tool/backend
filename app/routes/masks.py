import json
import logging
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, File
from starlette.datastructures import UploadFile
from sqlalchemy.orm import Session
from app.routes.contours import delete_contour, add_contours
from app.database import get_session
from app.database.images import Images
from app.database.masks import Masks
from app.database.contours import Contours
from app.routes.prompted_segmentation.util import get_masks_responses
from app.schemas.segmentation.segmentations import SegmentationMaskModel
from app.services.mask_generation import generate_mask
from app.services.database_access import save_array_to_disk
from app.services.labels import get_hierarchical_label_name
from app.routes.semantic_segmentation.upload_data import proxy_upload_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/masks", tags=["masks"])


@router.put("/create_mask/{image_id}")
async def create_mask(image_id: int, db: Session = Depends(get_session)):
    """ Create a new mask for the given image ID. Only one mask can exist per image. """
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
async def finish_mask(mask_id: int, db: Session = Depends(get_session)):
    """ Mark a mask as finished, generate it as an image file and upload it to the AI external service. """
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
    mask_image = generate_mask(mask_id)
    logging.debug(f"Generated mask with the following labels: {np.unique(mask_image).tolist()}")
    mask_path = save_array_to_disk(mask_image,
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
async def unfinish_mask(mask_id: int, db: Session = Depends(get_session)):
    """ Remove the finished status from a mask, allowing it to be edited again. This will also delete the mask image
        file and remove it from the AI external service. """
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
async def get_mask(mask_id: int, db: Session = Depends(get_session)):
    """ Get a mask by its ID. """
    mask = db.query(Masks).filter_by(id=mask_id).first()
    if mask is None:
        raise HTTPException(status_code=404, detail="Mask not found.")
    return {
        "success": True,
        "mask": mask
    }


@router.get("/get_mask_annotation_status/{mask_id}")
async def get_mask_annotation_status(mask_id: int, db: Session = Depends(get_session)):
    """ Check the annotation status of a mask by its ID. """
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
async def delete_mask(mask_id: int, db: Session = Depends(get_session)):
    """ Delete a mask and all its contours by its ID. """
    mask = db.query(Masks).filter_by(id=mask_id).first()
    if mask is None:
        raise HTTPException(status_code=404, detail="Mask not found.")
    root_contours = db.query(Contours).filter_by(mask_id=mask_id, parent_id=None).all()
    for contour in root_contours:
        await delete_contour(contour.id, db)
    db.delete(mask)
    db.commit()
    return {"success": True, "message": "Mask deleted successfully."}


@router.post("/post_mask/{mask_id}")
async def post_mask(mask_id: int, mask: UploadFile =File(...), session: Session = Depends(get_session)):
    mask_array = np.frombuffer(mask.file.read(), dtype=np.uint8)
    mask_responses = await get_masks_responses([mask_array], [1], only_return_one=False)
    image_id = session.query(Masks.image_id).filter_by(id=mask_id).first()
    return await create_masks_and_add_contours_for_images([image_id], mask_array)



async def create_masks_and_add_contours_for_images(image_ids: list[int],
                                                   mask_responses: list[SegmentationMaskModel],
                                                   db: Session = Depends(get_session)):
    """
    Create masks for a list of image IDs and add contours to them.

    Args:
        image_ids (list[int]): List of image IDs for which to create masks.
        mask_responses (list[SegmentationMaskModel]): List of segmentation mask responses containing contours.
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

import json
import logging
import os.path

import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from iquana_toolbox.schemas.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.image import Image
from iquana_toolbox.schemas.labels import LabelHierarchy
from iquana_toolbox.schemas.user import User
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.services.auth import get_current_user
from app.services.database_access.images import process_and_save_image
from app.services.database_access.masks import create_new_mask

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/images", tags=["images"])


@router.post("/upload")
async def upload_image(dataset_id: int, file: UploadFile = File(...), db: Session = Depends(get_session)):
    dataset = db.query(Datasets).get(dataset_id)
    image_id = await process_and_save_image(file, dataset_id, dataset.folder_path, db)
    db.commit()  # Commit once
    return {"success": True, "image_id": image_id}


@router.post("/upload_multi")
async def upload_images(dataset_id: int, files: list[UploadFile] = File(...), db: Session = Depends(get_session)):
    dataset = db.query(Datasets).get(dataset_id)
    image_ids = []

    for file in files:
        # Now we only query the dataset folder ONCE at the top
        image_id = await process_and_save_image(file, dataset_id, dataset.folder_path, db)
        image_ids.append(image_id)

    db.commit()  # One big commit for all images (Atomic!)
    return {"success": True, "image_ids": image_ids}


@router.delete("/{image_id}")
async def delete_image(
        image_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """Delete an image and its associated masks.

    Args:
        image_id: ID of the image to delete.
        db: Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A dictionary indicating success and a message.
    """
    image = db.query(Images).filter_by(id=image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    if os.path.exists(image.file_path):
        os.remove(image.file_path)  # Remove the original image file
    if os.path.exists(image.thumbnail_file_path):
        os.remove(image.thumbnail_file_path)  # Remove the thumbnail
    db.delete(image)
    db.commit()
    return {"success": True,
            "message": f"Deleted image {image_id}."}



@router.get("/{image_id}/b64")
async def get_base64_image(
        image_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """Get images via ids.

    Args:
        image_id (int): Image ID to retrieve.
        low_res (bool): Whether to return low resolution images (thumbnails). Defaults to False.
        db (Session): Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A dict mapping from image ID to base64 encoded image.
    """
    image_query = db.query(Images).filter_by(id=image_id).first()
    image = Image.from_db(image_query)
    b64_str = image.load_image(as_base64=True)
    return {
        "success": True,
        "message": f"Successfully retrieved image {image_id}.",
        image_id: b64_str
    }


@router.get("/{image_id}/thumbnail")
async def get_base64_thumbnail(
        image_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """Get images via ids.

    Args:
        image_id (int): Image ID to retrieve.
        low_res (bool): Whether to return low resolution images (thumbnails). Defaults to False.
        db (Session): Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A dict mapping from image ID to base64 encoded image.
    """
    image_query = db.query(Images).filter_by(id=image_id).first()
    image = Image.from_db(image_query)
    b64_str = image.load_thumbnail(as_base64=True)
    return {
        "success": True,
        "message": f"Successfully retrieved image {image_id}.",
        image_id: b64_str
    }


@router.get("/ids/b64")
async def get_base64_images(
        image_ids: str,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """Get images via a list of image IDs. This gets the images in batches to avoid sending too many requests at once.

    Args:
        image_ids (str): JSON string containing a list of image IDs to retrieve.
        low_res (bool): Whether to return low resolution images (thumbnails). Defaults to False.
        db (Session): Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A dictionary mapping from image ID to base64 encoded image.
    """
    # Parse image_ids from JSON string
    image_ids = json.loads(image_ids)
    if not isinstance(image_ids, list):
        raise HTTPException(status_code=400, detail="image_ids must be a list")

    response = {}
    images_query = db.query(Images).filter(Images.id.in_(image_ids)).all()
    images = [Image.from_db(img) for img in images_query]
    for img in images:
        response[img.id] = img.load_image(as_base64=True)
    return {
        "success": True,
        "message": f"Successfully retrieved {len(image_ids)} images.",
        "images": response
    }


@router.get("/ids/thumbnails")
async def get_base64_thumbnails(
        image_ids: str,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """Get images via a list of image IDs. This gets the images in batches to avoid sending too many requests at once.

    Args:
        image_ids (str): JSON string containing a list of image IDs to retrieve.
        low_res (bool): Whether to return low resolution images (thumbnails). Defaults to False.
        db (Session): Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A dictionary mapping from image ID to base64 encoded image.
    """
    # Parse image_ids from JSON string
    image_ids = json.loads(image_ids)
    if not isinstance(image_ids, list):
        raise HTTPException(status_code=400, detail="image_ids must be a list")

    response = {}
    images_query = db.query(Images).filter(Images.id.in_(image_ids)).all()
    images = [Image.from_db(img) for img in images_query]
    for img in images:
        response[img.id] = img.load_thumbnail(as_base64=True)
    return {
        "success": True,
        "message": f"Successfully retrieved {len(image_ids)} images.",
        "images": response
    }


@router.post("/{image_id}/masks/create", deprecated=True)
async def create_new_mask_for_image(
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
    raise NotImplementedError("This method is not implemented.")


@router.get("/{image_id}/masks")
async def get_mask_for_image(image_id: int,
                             db: Session = Depends(get_session),
                             user: User = Depends(get_current_user)):
    """ Get the mask image for a given image. """
    masks = db.query(Masks).filter_by(image_id=image_id).all()
    if masks is None:
        raise HTTPException(status_code=404, detail=f"No mask for image {image_id} found.")
    return {
        "success": True,
        "masks": masks
    }


@router.post("/{image_id}/masks/upload/semantic_mask")
async def post_semantic_mask_to_image(
        image_id: int,
        mask: UploadFile = File(...),
        session: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """
    Upload a mask to a mask id. Compute the contours for each label in the mask, build the hierarchy and add
    them to the database.

    Args:
        image_id (int): The ID of the image.
        mask (UploadFile): The mask file.
        session (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and result.
    """
    mask_array = np.frombuffer(mask.file.read(), dtype=np.uint8)
    image = session.query(Images, Masks.id).join(Masks, Images.id == Masks.image_id).filter(Images.id == image_id).first()
    labels = session.query(Labels).filter_by(dataset_id=image.dataset_id)
    label_hierarchy = LabelHierarchy.from_query(labels)

    # Create an initial hierarchy of already added contours
    contour_hierarchy = ContourHierarchy.from_query(
        session.query(Contours).filter_by(mask_id=image.mask_id).all(),
        height=image.height,
        width=image.width,
    )
    # Add new contours from the mask
    contour_hierarchy = contour_hierarchy.from_semantic_mask(
        mask_array,
        label_hierarchy,
        user.username,
    )
    return {
        "success": True,
        "message": "Converted mask object to contour hierarchy and added it to the database.",
        "result": contour_hierarchy.model_dump_json()
    }

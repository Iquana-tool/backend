import json
import logging

from fastapi import APIRouter, UploadFile, File, Depends
from iquana_toolbox.schemas.user import User
from sqlalchemy.orm import Session

from app.database import get_session
from app.services.auth import get_current_user
from app.services.database_access import datasets as datasets_db
from app.services.database_access import images as images_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/images", tags=["images"])


@router.post("/upload")
async def upload_image(
        dataset_id: int,
        file: UploadFile = File(...),
        db: Session = Depends(get_session),
):
    dataset = await datasets_db.get_dataset(dataset_id, db=db)
    image_id = await images_db.process_and_save_image(file, dataset_id, dataset.folder_path, db=db)
    return {
        "success": True,
        "message": f"Uploaded image {image_id}.",
        "image_id": image_id
    }


@router.post("/upload_multi")
async def upload_images(
        dataset_id: int,
        files: list[UploadFile] = File(...),
        db: Session = Depends(get_session),
):
    dataset = await datasets_db.get_dataset(dataset_id, db=db)
    image_ids = []

    for file in files:
        # Now we only query the dataset folder ONCE at the top
        image_id = await images_db.process_and_save_image(file, dataset_id, dataset.folder_path, db=db)
        image_ids.append(image_id)

    return {
        "success": True,
        "message": f"Uploaded {len(image_ids)} images.",
        "image_ids": image_ids
    }


@router.delete("/{image_id}")
async def delete_image(
        image_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """Delete an image and its associated masks.

    Args:
        image_id: ID of the image to delete.
        user (User): The current authenticated user.

    Returns:
        A dictionary indicating success and a message.
    """
    await images_db.delete_image(image_id, db=db)
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
    return {
        "success": True,
        "message": f"Successfully retrieved image {image_id}.",
        image_id: await images_db.get_image_data(image_id, as_thumbnail=False, as_base64=True, db=db)
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
    return {
        "success": True,
        "message": f"Successfully retrieved image {image_id}.",
        image_id: await images_db.get_image_data(image_id, as_thumbnail=True, as_base64=True, db=db)
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
    image_ids = json.loads(image_ids)
    return {
        "success": True,
        "message": f"Successfully retrieved {len(image_ids)} images.",
        "images": await images_db.get_images_data(image_ids, as_thumbnail=False, as_base64=True, db=db)
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
        db (Session): Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A dictionary mapping from image ID to base64 encoded image.
    """
    image_ids = json.loads(image_ids)
    return {
        "success": True,
        "message": f"Successfully retrieved {len(image_ids)} images.",
        "images": await images_db.get_images_data(image_ids, as_thumbnail=True, as_base64=True, db=db)
    }


@router.get("/{image_id}/masks")
async def get_mask_for_image(
        image_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """ Get the mask image for a given image. """
    return {
        "success": True,
        "masks": await images_db.get_masks_of_image(image_id, db=db)
    }


@router.post("/{image_id}/masks/upload/semantic_mask")
async def post_semantic_mask_to_image(
        image_id: int,
        mask: UploadFile = File(...),
        db: Session = Depends(get_session),
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
    raise NotImplementedError

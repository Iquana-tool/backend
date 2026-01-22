import os
import shutil
from collections import defaultdict
from logging import getLogger
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from schemas.image import Image
from schemas.labels import LabelHierarchy
from schemas.user import User
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.routes.general.images import router, logger
from app.routes.general.labels import router
from app.routes.general.masks import get_mask_annotation_status
from app.services.auth import get_current_user
from paths import Paths

# Create a router for the export functionality
router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = getLogger(__name__)

@router.post("/create")
async def create_dataset(name: str,
                         description: str,
                         dataset_type: Literal["image", "scan", "DICOM"],
                         db: Session = Depends(get_session),
                         current_user=Depends(get_current_user)):
    """Create a new dataset.

    Args:
        name (str): The name of the dataset.
        description (str): A brief description of the dataset.
        dataset_type (Literal["image", "scan", "DICOM"]): The type of dataset.
        db (Session): The database session.
        current_user (Users): Auth bearer token.

    Returns:
        dict: A dictionary containing the success status and message, or error details.
    """
    try:
        # Check if dataset with the same name already exists
        existing_dataset = db.query(Datasets).filter_by(name=name.strip()).first()
        if existing_dataset:
            return {"success": False,
                    "message": f"Dataset with name '{name.strip()}' already exists.",
                    "error": "Duplicate dataset name"}
        
        dataset_path = os.path.join(Paths.datasets_dir, name.strip())
        # Use exist_ok=True to avoid FileExistsError if directory already exists
        os.makedirs(dataset_path, exist_ok=True)
        
        new_dataset = Datasets(
            name=name.strip(),
            description=description.strip(),
            folder_path=dataset_path,
            dataset_type=dataset_type,
            created_by=current_user.username,
        )
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)
        return {"success": True,
                "message": "Dataset created successfully.",
                "dataset_id": new_dataset.id
                }
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        db.rollback()
        return {"success": False,
                "message": "Error creating dataset.",
                "error": str(e)}

@router.post("/{dataset_id}/share")
async def share_dataset(
    dataset_id: int,
    share_with_username: str,
    db: Session = Depends(get_session),
    user: "User" = Depends(get_current_user)
):
    """Share a dataset with another user by username.

    Args:
        dataset_id (int): The ID of the dataset to share.
        share_with_username (str): The username to share with.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.created_by != user.id:
        raise HTTPException(status_code=403, detail="Only the owner can share this dataset")
    user_to_share = db.query(Users).filter_by(name=share_with_username).first()
    if not user_to_share:
        raise HTTPException(status_code=404, detail="User to share with not found")
    if user_to_share in dataset.shared_with:
        return {"success": False, "message": "User already has access"}
    dataset.shared_with.append(user_to_share)
    db.commit()
    return {"success": True, "message": f"Dataset shared with {share_with_username}"}


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_session),
    user: "User" = Depends(get_current_user)
):
    """Get dataset information.

    Args:
        dataset_id (int): The ID of the dataset.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and dataset information.
    """
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        return {"success": False, "message": "Dataset not found."}
    return {"success": True, "message": "Dataset found.", "dataset": dataset}


@router.get("/{dataset_id}/images/count")
async def get_number_of_images(
    dataset_id: int,
    db: Session = Depends(get_session),
    user: "User" = Depends(get_current_user)
):
    """Get the number of images in a dataset.

    Args:
        dataset_id (int): The ID of the dataset.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the number of images.
    """
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        return {"success": False, "message": "Dataset not found."}
    number_of_images = db.query(Images).filter_by(dataset_id=dataset_id).count()
    return {"success": True, "number_of_images": number_of_images}


@router.get("/{dataset_id}/progress")
async def get_annotation_progress(dataset_id: int,
                                  user: User = Depends(get_current_user),
                                  db: Session = Depends(get_session)):
    """Get the annotation progress of a dataset.

    Args:
        dataset_id (int): The ID of the dataset to check.
        user (Users): Authentication bearer token.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the annotation progress details. The dict contains:
            - success (bool): Indicates if the operation was successful.
            - message (str): A message indicating the result of the operation.
            - manually_annotated (int): Number of images manually annotated.
            - auto_annotated_reviewed (int): Number of images auto-annotated with review.
            - auto_annotated_without_review (int): Number of images auto-annotated without review.
            - missing (int): Number of images missing annotations.
            - total_images (int): Total number of images in the dataset.
    """
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        return {"success": False, "message": "Dataset not found."}
    images = db.query(Images).filter_by(dataset_id=dataset_id).all()
    status_dict = defaultdict(lambda: 0)
    for image in images:
        mask = db.query(Masks).filter_by(image_id=image.id).first()
        status_response = await get_mask_annotation_status(mask.id, db, user)
        status_dict[status_response["status"]] += 1
    return {
        "success": True,
        "message": "Annotation progress retrieved successfully.",
        "total_images": len(images),
        "num_masks_with_status": status_dict,
    }


@router.get("/all")
async def get_all_datasets(
    db: Session = Depends(get_session),
    user: "User" = Depends(get_current_user)
):
    """Get all datasets owned by or shared with the current user.

    Args:
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and the list of datasets.
    """
    available_datasets = user.available_datasets
    datasets = db.query(Datasets).filter(
        Datasets.id.in_(available_datasets)
    ).all()
    return {"success": True, "datasets": [
        {
            "id": ds.id,
            "name": ds.name,
            "description": ds.description,
            "dataset_type": ds.dataset_type,
            "folder_path": ds.folder_path,
            "created_by": ds.created_by,
            "shared_with": [u.id for u in ds.shared_with]
        }
        for ds in datasets
    ]}


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_session),
    user: "User" = Depends(get_current_user)
):
    """Delete a dataset.

    Args:
        dataset_id (int): The ID of the dataset to delete.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    if dataset_id not in user.owned_datasets:
        return {"success": False, "message": "Dataset not owned by user. Only the owner can delete this dataset."}
    try:
        dataset = db.query(Datasets).filter_by(id=dataset_id).first()
        if not dataset:
            return {"success": False, "message": "Dataset not found."}
        # Delete disk directory, removes all image files.
        shutil.rmtree(dataset.folder_path, ignore_errors=True)
        # Delete the dataset
        db.delete(dataset)
        db.commit()
        return {"success": True, "message": "Dataset deleted successfully."}
    except Exception as e:
        logger.error(e)
        raise e
        return {"success": False, "message": "Error deleting dataset.", "error": str(e)}


@router.get("/{dataset_id}/images")
async def list_images(
    dataset_id: int,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """List all uploaded image ids in an image dataset.

    Args:
        dataset_id: ID of the dataset to retrieve images from.
        db: Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A dictionary containing the success status and the list of images.
    """
    images = (
        db.query(Images, Masks)
        .join(Masks, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
        .all()
    )
    image_response = []
    for entry in images:
        image = entry[0]
        mask = entry[1]
        image_response.append({
            **image.__dict__,
            "status": await get_mask_annotation_status(mask.id, db, user)
        })
    return {
        "success": True,
        "images": image_response
    }


@router.get("/{dataset_id}/images?status={status}")
async def list_images_with_annotation_status(
    dataset_id: int,
    status: Literal["not_started", "in_progress", "reviewable", "finished"],
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """List all images with masks of certain status for a given image ID.

    Args:
        dataset_id: Dataset ID to retrieve images from.
        status: The status of the masks to filter by.
        db: Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A list of image IDs.
    """
    match status:
        case "not_started":
            images = db.query(Images.id).filter_by(dataset_id=dataset_id).filter(~Masks.contours.any()).all()
        case "in_progress":
            images = db.query(Images.id).filter_by(dataset_id=dataset_id).filter(Masks.contours.any(), ~Masks.fully_annotated).all()
        case "reviewable":
            images = db.query(Images.id).filter_by(dataset_id=dataset_id).filter(
                            Masks.fully_annotated, Masks.contours.any(~Contours.reviewed_by.any())).all()
        case "finished":
            images = db.query(Images.id).filter_by(dataset_id=dataset_id).filter(
                Masks.fully_annotated, ~Masks.contours.any(~Contours.reviewed_by.any())
            )
        case _:
            raise HTTPException(status_code=403, detail="Unknown status.")
    return {
        "success": True,
        "message": "Retrieved images with status successfully.",
        "image_ids": [img.id for img in images]
    }


@router.get("/{dataset_id}/images/b64")
async def get_base64_images_of_dataset(
    dataset_id: int,
    limit: int = None,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """Get all images of a dataset.

    Args:
        dataset_id: ID of the dataset to retrieve images from.
        low_res: Whether to return low resolution images (thumbnails).
        limit: Optional limit on the number of images to return. If not provided, all images will be returned.
        db: Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A dict mapping from image ID to base64 encoded image.
    """
    try:
        response = {}
        images_query = db.query(Images).filter_by(dataset_id=dataset_id).limit(limit).all()
        images = [Image.from_db(img) for img in images_query]
        for img in images:
            response[id] = img.load_image(as_base64=True)
        return {
            "success": True,
            "message": f"Successfully retrieved {len(images)} images from dataset {dataset_id}.",
            "images": response
        }
    except Exception as e:
        logger.error(f"Get all images of dataset error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/thumbnails/b64")
async def get_base64_thumbnails_of_dataset(
    dataset_id: int,
    limit: int = None,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """Get all images of a dataset.

    Args:
        dataset_id: ID of the dataset to retrieve images from.
        low_res: Whether to return low resolution images (thumbnails).
        limit: Optional limit on the number of images to return. If not provided, all images will be returned.
        db: Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A dict mapping from image ID to base64 encoded image.
    """
    try:
        response = {}
        images_query = db.query(Images).filter_by(dataset_id=dataset_id).limit(limit).all()
        images = [Image.from_db(img) for img in images_query]
        for img in images:
            response[id] = img.load_image(as_base64=True)
        return {
            "success": True,
            "message": f"Successfully retrieved {len(images)} images from dataset {dataset_id}.",
            "images": response
        }
    except Exception as e:
        logger.error(f"Get all images of dataset error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get_labels/{dataset_id}")
async def get_labels(
    dataset_id: int,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """Retrieve all labels for a given dataset.

    Args:
        dataset_id (int): The ID of the dataset.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and the labels hierarchy.
    """
    labels = db.query(Labels).filter_by(dataset_id=dataset_id)
    labels_hierarchy = LabelHierarchy.from_query(labels)
    return {
        "success": True,
        "message": f"Retrieved {len(labels_hierarchy)} labels for dataset {dataset_id}.",
        "labels": labels_hierarchy.model_dump()
    }

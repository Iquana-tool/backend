import shutil
from app.services.auth import get_current_user
from logging import getLogger

from paths import Paths
from fastapi import APIRouter, Depends, HTTPException
import os
from typing import Literal
from app.database import get_session
from sqlalchemy.orm import Session
from app.database.images import Images
from app.database.datasets import Datasets
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.routes.images import delete_image
from app.routes.labels import delete_label

# Create a router for the export functionality
router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = getLogger(__name__)

@router.post("/create_dataset")
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
        dataset_path = os.path.join(Paths.datasets_dir, name)
        os.makedirs(dataset_path)
        new_dataset = Datasets(
            name=name.strip(),
            description=description.strip(),
            folder_path=dataset_path,
            dataset_type=dataset_type,
            created_by=current_user.id
        )
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)
        return {"success": True,
                "message": "Dataset created successfully.",
                "dataset_id": new_dataset.id
                }
    except Exception as e:
        return {"success": False,
                "message": "Error creating dataset.",
                "error": str(e)}

@router.post("/share_dataset")
async def share_dataset(
    dataset_id: int,
    share_with_username: str,
    db: Session = Depends(get_session),
    current_user=Depends(get_current_user)
):
    """Share a dataset with another user by username."""
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.created_by != current_user.id:
        raise HTTPException(status_code=403, detail="Only the owner can share this dataset")
    user_to_share = db.query(Users).filter_by(name=share_with_username).first()
    if not user_to_share:
        raise HTTPException(status_code=404, detail="User to share with not found")
    if user_to_share in dataset.shared_with:
        return {"success": False, "message": "User already has access"}
    dataset.shared_with.append(user_to_share)
    db.commit()
    return {"success": True, "message": f"Dataset shared with {share_with_username}"}


@router.get("/get_dataset/{dataset_id}")
async def get_dataset(dataset_id: int,
                     db: Session = Depends(get_session),
                     current_user=Depends(get_current_user)):
    """Get dataset information."""
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        return {"success": False, "message": "Dataset not found."}
    return {"success": True, "message": "Dataset found.", "dataset": dataset}


@router.get("/get_number_of_images/{dataset_id}")
async def get_number_of_images(dataset_id: int, db: Session = Depends(get_session)):
    """Get the number of images in a dataset."""
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        return {"success": False, "message": "Dataset not found."}
    number_of_images = db.query(Images).filter_by(dataset_id=dataset_id).count()
    return {"success": True, "number_of_images": number_of_images}


@router.get("/get_annotation_progress/{dataset_id}")
async def get_annotation_progress(dataset_id: int, db: Session = Depends(get_session)):
    """Get the annotation progress of a dataset.

    Args:
        dataset_id (int): The ID of the dataset to check.
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
    manually_annotated = 0
    auto_annotated_with_review = 0
    auto_annotated_without_review = 0
    missing = 0
    for image in images:
        mask = db.query(Masks).filter_by(image_id=image.id, finished=True).first()
        if mask:
            if mask.finished:
                manually_annotated += 1
            elif mask.generated:
                if mask.reviewed:
                    auto_annotated_with_review += 1
                else:
                    auto_annotated_without_review += 1
            else:
                missing += 1

    return {
        "success": True,
        "message": "Annotation progress retrieved successfully.",
        "manually_annotated": manually_annotated,
        "auto_annotated_reviewed": auto_annotated_with_review,
        "auto_annotated_without_review": auto_annotated_without_review,
        "missing": len(images) - (manually_annotated + auto_annotated_with_review + auto_annotated_without_review),
        "total_images": len(images),
    }


@router.get("/get_datasets")
async def get_datasets(db: Session = Depends(get_session), current_user=Depends(get_current_user)):
    """Get all datasets owned by or shared with the current user."""
    datasets = db.query(Datasets).filter(
        (Datasets.created_by == current_user.id) |
        (Datasets.shared_with.any(id=current_user.id))
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


@router.delete("/delete_dataset/{dataset_id}")
async def delete_dataset(dataset_id: int, db: Session = Depends(get_session)):
    """Delete a dataset."""
    try:
        dataset = db.query(Datasets).filter_by(id=dataset_id).first()
        if not dataset:
            return {"success": False, "message": "Dataset not found."}
        images = db.query(Images).filter_by(dataset_id=dataset_id).all()
        for image in images:
            await delete_image(image.id, db)
        # Delete disk directory
        shutil.rmtree(dataset.folder_path, ignore_errors=True)
        # Delete associated labels
        labels = db.query(Labels).filter_by(dataset_id=dataset_id).all()
        for label in labels:
            await delete_label(label.id, db)
        # Delete the dataset
        db.delete(dataset)
        db.commit()
        return {"success": True, "message": "Dataset deleted successfully."}
    except Exception as e:
        logger.error(e)
        raise e
        return {"success": False, "message": "Error deleting dataset.", "error": str(e)}

import shutil
from paths import Paths
from fastapi import APIRouter, Depends
import os
from typing import Literal
from app.database import get_session
from sqlalchemy.orm import Session
from app.database.images import Images
from app.database.datasets import Datasets, Labels
from app.database.mask_generation import Masks

# Create a router for the export functionality
router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/create_dataset")
async def create_dataset(name: str,
                         description: str,
                         dataset_type: Literal["image", "scan", "DICOM"],
                         db: Session = Depends(get_session)):
    """Create a new dataset."""
    try:
        dataset_path = os.path.join(Paths.datasets_dir, name)
        os.makedirs(dataset_path)
        new_dataset = Datasets(name=name.strip(),
                               description=description.strip(),
                               folder_path=dataset_path,
                               dataset_type=dataset_type)
        db.add(new_dataset)
        db.commit()
        return {"success": True,
                "message": "Dataset created successfully.",
                "dataset_id": new_dataset.id}
    except Exception as e:
        return {"success": False,
                "message": "Error creating dataset.",
                "error": str(e)}


@router.get("/get_dataset/{dataset_id}")
def get_dataset(dataset_id: int, db: Session = Depends(get_session)):
    """Get dataset information."""
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        return {"success": False, "message": "Dataset not found."}
    return {"success": True, "message": "Dataset found.", "dataset": dataset}


@router.get("/get_number_of_images/{dataset_id}")
def get_number_of_images(dataset_id: int, db: Session = Depends(get_session)):
    """Get the number of images in a dataset."""
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        return {"success": False, "message": "Dataset not found."}
    number_of_images = db.query(Images).filter_by(dataset_id=dataset_id).count()
    return {"success": True, "number_of_images": number_of_images}


@router.get("/get_annotation_progress/{dataset_id}")
async def get_annotation_progress(dataset_id: int, db: Session = Depends(get_session)):
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        return {"success": False, "message": "Dataset not found."}
    images = db.query(Images).filter_by(dataset_id=dataset_id).all()
    manually_annotated = 0
    auto_annotated_with_review = 0
    auto_annotated_without_review = 0
    for image in images:
        mask = db.query(Masks).filter_by(image_id=image.id, finished=True).first()
        if mask:
            if mask.generated:
                if mask.reviewed:
                    auto_annotated_with_review += 1
                else:
                    auto_annotated_without_review += 1
            else:
                manually_annotated += 1
    n_images = get_number_of_images(dataset_id, db)["number_of_images"]

    return {
        "success": True,
        "message": "Annotation progress retrieved successfully.",
        "manually_annotated": manually_annotated,
        "auto_annotated_reviewed": auto_annotated_with_review,
        "auto_annotated_without_review": auto_annotated_without_review,
        "total_images": n_images,
    }


@router.get("/get_datasets")
def get_datasets(db: Session = Depends(get_session)):
    """Get all datasets."""
    datasets = db.query(Datasets).all()
    return {"success": True, "datasets": datasets}


@router.delete("/delete_dataset/{dataset_id}")
async def delete_dataset(dataset_id: int, db: Session = Depends(get_session)):
    """Delete a dataset."""
    try:
        dataset = db.query(Datasets).filter_by(id=dataset_id).first()
        if not dataset:
            return {"success": False, "message": "Dataset not found."}
        # Delete disk directory
        shutil.rmtree(dataset.folder_path, ignore_errors=True)
        # Delete associated labels
        db.query(Labels).filter_by(dataset_id=dataset_id).delete()
        # Delete the dataset
        db.delete(dataset)
        db.commit()
        return {"success": True, "message": "Dataset deleted successfully."}
    except Exception as e:
        return {"success": False, "message": "Error deleting dataset.", "error": str(e)}

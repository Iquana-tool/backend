from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, StreamingResponse
import pandas as pd
import os
from io import StringIO
from app.database import get_session
from sqlalchemy.orm import Session
from app.database.images import Images
from app.database.datasets import Datasets, Labels
from app.database.mask_generation import Masks

# Create a router for the export functionality
router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/create_dataset")
async def create_dataset(name: str, description: str, db: Session = Depends(get_session)):
    """Create a new dataset."""
    try:
        new_dataset = Datasets(name=name, description=description)
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

        # Delete associated labels
        db.query(Labels).filter_by(dataset_id=dataset_id).delete()

        # Delete the dataset
        db.delete(dataset)
        db.commit()
        return {"success": True, "message": "Dataset deleted successfully."}
    except Exception as e:
        return {"success": False, "message": "Error deleting dataset.", "error": str(e)}

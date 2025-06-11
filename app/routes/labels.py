import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import Field
from sqlalchemy.orm import Session
from app.database import get_session
from app.database.datasets import Labels


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/labels", tags=["labels"])


@router.get("/get_labels/{dataset_id}")
def get_labels(dataset_id: int, db: Session = Depends(get_session)):
    classes = db.query(Labels).filter_by(dataset_id=dataset_id).all()
    return classes


@router.post("/create_label")
async def create_label(label_name: str,
                       dataset_id: int,
                       parent_label_id: int = None,
                       label_value: int = None,
                       db: Session = Depends(get_session)):
    try:
        # Check if class already exists
        existing_class = db.query(Labels).filter_by(dataset_id=dataset_id, name=label_name).first()
        if existing_class:
            raise HTTPException(status_code=400, detail="Label already exists.")
        if parent_label_id:
            # Check if parent class exists
            parent_label = db.query(Labels).filter_by(id=parent_label_id).first()
            if not parent_label:
                raise HTTPException(status_code=404, detail="Parent label not found.")
        if not label_value:
            label_value = db.query(Labels).filter_by(dataset_id=dataset_id).count() + 1  # Default value
        # Create a new class
        new_label = Labels(dataset_id=dataset_id,
                           name=label_name,
                           parent_id=parent_label_id,
                           value=label_value)
        db.add(new_label)
        db.commit()
        return {
            "success": True,
            "message": "Label created successfully.",
            "class_id": new_label.id
        }
    except Exception as e:
        logger.error(f"Error creating class: {e}")
        raise HTTPException(status_code=500, detail="Error creating class.")


@router.delete("/delete_label/label={label_id}")
async def delete_label(label_id: int, db: Session = Depends(get_session)):
    try:
        # Check if class exists
        existing_label = db.query(Labels).filter_by(id=label_id).first()
        if not existing_label:
            raise HTTPException(status_code=404, detail="Class not found.")

        # Delete the class
        db.delete(existing_label)
        db.commit()
        return {
            "success": True,
            "message": "Class deleted successfully."
        }
    except Exception as e:
        logger.error(f"Error deleting class: {e}")
        raise HTTPException(status_code=500, detail="Error deleting class.")
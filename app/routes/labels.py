import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import Field
from sqlalchemy.orm import Session
from app.database import get_session
from app.database.labels import Labels
from app.database.contours import Contours
from app.schemas.labels import LabelHierarchy

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/labels", tags=["labels"])


@router.get("/get_labels/{dataset_id}")
async def get_labels(dataset_id: int, db: Session = Depends(get_session)):
    """Retrieve all labels for a given dataset."""
    labels = db.query(Labels).filter_by(dataset_id=dataset_id)
    labels_hierarchy = LabelHierarchy.from_query(labels)
    return {
        "success": True,
        "message": f"Retrieved {len(labels)} labels for dataset {dataset_id}.",
        "labels": labels_hierarchy.model_dump()
    }


@router.post("/create_label")
async def create_label(label_name: str,
                       dataset_id: int,
                       parent_label_id: int = None,
                       label_value: int = None,
                       db: Session = Depends(get_session)):
    """Create a new label for a dataset.

    Args:
        label_name (str): The name of the label to create.
        dataset_id (int): The ID of the dataset to which the label belongs.
        parent_label_id (int, optional): The ID of the parent label if this is a child label. Defaults to None.
        label_value (int, optional): The value of the label. If not provided, it will be set to the next available value.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and class ID if created successfully.
    """
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
    """
    Delete a label, its children and all associated contours.

    Args:
        label_id (int): The ID of the label to delete.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    try:
        # Check if class exists
        existing_label = db.query(Labels).filter_by(id=label_id).first()
        if not existing_label:
            # If class does not exist, return success with message
            # Remark: Not sure if this is the desired behavior, but it is consistent.
            return {
                "success": True,
                "message": "Class never existed."
            }

        # Check if class has children
        children = db.query(Labels).filter_by(parent_id=label_id).all()
        for child in children:
            await delete_label(child.id, db)

        # Delete all contours with this label
        contours = db.query(Contours).filter_by(label=label_id).all()
        for contour in contours:
            db.delete(contour)

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
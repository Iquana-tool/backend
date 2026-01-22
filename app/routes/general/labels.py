import logging

from fastapi import APIRouter, HTTPException, Depends
from schemas.labels import Label
from schemas.user import User
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.labels import Labels
from app.services.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/labels", tags=["labels"])


@router.post("/create")
async def create_label(label_name: str,
                       dataset_id: int,
                       parent_label_id: int = None,
                       label_value: int = None,
                       user: User = Depends(get_current_user),
                       db: Session = Depends(get_session)):
    """Create a new label for a dataset.

    Args:
        label_name (str): The name of the label to create.
        dataset_id (int): The ID of the dataset to which the label belongs.
        parent_label_id (int, optional): The ID of the parent label if this is a child label. Defaults to None.
        label_value (int, optional): The value of the label. If not provided, it will be set to the next available value.
        user (User): The current authenticated user. Defaults to Depends(get_current_user).
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and class ID if created successfully.
    """
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


@router.get("/{label_id}")
async def get_label(label_id: int,
                    user: User = Depends(get_current_user),
                    db: Session = Depends(get_session)):
    """Create a new label for a dataset.

    Args:
        label_id (int): The ID of the label to get.
        user (User): The current authenticated user. Defaults to Depends(get_current_user).
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and class ID if created successfully.
    """
    # Check if class already exists
    existing_class = db.query(Labels).filter_by(id=label_id).first()
    return {
        "success": True,
        "message": "Label retrieved successfully.",
        "class_id": Label.from_db(existing_class),
    }


@router.patch("/{label_id}")
async def modify_label(label_id: int,
                       updates: dict = None,
                       user: User = Depends(get_current_user),
                       db: Session = Depends(get_session)):
    """Create a new label for a dataset.

    Args:
        label_id (int): The ID of the label to get.
        user (User): The current authenticated user. Defaults to Depends(get_current_user).
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and class ID if created successfully.
    """
    # Check if class already exists
    existing_class = db.query(Labels).filter_by(id=label_id).first()
    for k, v in updates.items():
        setattr(existing_class, k, v)
    db.commit()
    return {
        "success": True,
        "message": "Label updated successfully.",
        "class_id": Label.from_db(existing_class),
    }


@router.put("/{label_id}")
async def replace_label(label_id: int,
                        new_label: Label,
                        user: User = Depends(get_current_user),
                        db: Session = Depends(get_session)):
    """Create a new label for a dataset.

    Args:
        label_id (int): The ID of the label to get.
        user (User): The current authenticated user. Defaults to Depends(get_current_user).
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and class ID if created successfully.
    """
    # Check if class already exists
    existing_class = db.query(Labels).filter_by(id=label_id).first()
    parent_id = existing_class.parent_id
    db.delete(existing_class)
    new_label.id = label_id
    new_label.parent = parent_id
    new_label_db = Labels.from_schema(new_label)
    db.add(new_label_db)
    db.commit()
    return {
        "success": True,
        "message": "Label replaced successfully.",
    }


@router.delete("/{label_id}")
async def delete_label(label_id: int,
                       user: User = Depends(get_current_user),
                       db: Session = Depends(get_session)):
    """
    Delete a label, its children and all associated contours.

    Args:
        label_id (int): The ID of the label to delete.
        user (User): The current authenticated user.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    # Check if class exists
    existing_label = db.query(Labels).filter_by(id=label_id).first()
    if not existing_label:
        # If class does not exist, return success with message
        # Remark: Not sure if this is the desired behavior, but it is consistent.
        return {
            "success": True,
            "message": "Class never existed."
        }
    # Delete the class
    db.delete(existing_label)
    db.commit()
    return {
        "success": True,
        "message": "Class deleted successfully."
    }

import logging

from fastapi import APIRouter, Depends, HTTPException
from iquana_toolbox.schemas.database.labels import Label
from sqlalchemy.orm import Session

from app.database import get_session
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.label_space import LabelSpaceDraft
from app.schemas.permissions import Permission
from app.services.database_access import labels as labels_db
from app.services.permissions import require

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/labels", tags=["labels"])


@router.post("/create")
async def create_label(
        label_name: str,
        dataset_id: int,
        parent_label_id: int = None,
        label_value: int = None,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_MANAGE))
):
    """Create a new label for a dataset.

    Args:
        label_name (str): The name of the label to create.
        dataset_id (int): The ID of the dataset to which the label belongs.
        parent_label_id (int, optional): The ID of the parent label if this is a child label. Defaults to None.
        label_value (int, optional): The value of the label. If not provided, it will be set to the next available value.
        user (AuthenticatedUser): The current authenticated user.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and class ID if created successfully.
    """
    new_label = await labels_db.create_label(label_name, dataset_id, db, parent_label_id, label_value)
    return {
        "success": True,
        "message": "Label created successfully.",
        "class_id": new_label.id
    }


@router.post("/bulk_create")
async def bulk_create_labels(
        dataset_id: int,
        draft: LabelSpaceDraft,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_MANAGE)),
):
    """Persist an approved draft label hierarchy for a dataset in one transaction.

    Used by the "Describe your label space" assistant to apply a reviewed draft.
    Label names must be unique across the dataset; on a conflict nothing is
    created and a 400 is returned so the user can resolve it in the review step.

    Args:
        dataset_id (int): The dataset the labels belong to.
        draft (LabelSpaceDraft): The nested label hierarchy to create.
        db (Session): The database session.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        dict: success status, message and the number of labels created.
    """
    try:
        created_count = await labels_db.bulk_create_labels(dataset_id, draft.labels, db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "success": True,
        "message": f"Created {created_count} labels.",
        "created_count": created_count,
    }


@router.get("/{label_id}")
async def get_label(
        label_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_READ, "label_id")),
):
    """Get a single label.

    Args:
        label_id (int): The ID of the label to get.
        user (AuthenticatedUser): The current authenticated user.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status, message, and the label.
    """

    return {
        "success": True,
        "message": "Label retrieved successfully.",
        "class_id": await labels_db.get_label(label_id, db),
    }


@router.patch("/{label_id}")
async def modify_label(
        label_id: int,
        updates: dict = None,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_MANAGE, "label_id")),
):
    """Update fields on a label.

    Args:
        label_id (int): The ID of the label to update.
        user (AuthenticatedUser): The current authenticated user.
        updates (dict): A dictionary containing the updated label data. Defaults to None.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    # Check if class already exists
    await labels_db.update_label(label_id, updates, db)
    return {
        "success": True,
        "message": "Label updated successfully.",
    }


@router.put("/{label_id}")
async def replace_label(
        label_id: int,
        new_label: Label,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_MANAGE, "label_id"))
):
    """Replace a label wholesale.

    Args:
        label_id (int): The ID of the label to replace.
        user (AuthenticatedUser): The current authenticated user.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    await labels_db.replace_label(label_id, new_label, db)
    return {
        "success": True,
        "message": "Label replaced successfully.",
    }


@router.delete("/{label_id}")
async def delete_label(
        label_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_MANAGE, "label_id"))
):
    """
    Delete a label, its children and all associated contours.

    Deleting a label cascades into every contour carrying it, which is why this
    needs `label.manage` rather than ordinary annotation rights.

    Args:
        label_id (int): The ID of the label to delete.
        user (AuthenticatedUser): The current authenticated user.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    await labels_db.delete_label(label_id, db)
    return {
        "success": True,
        "message": "Class deleted successfully."
    }

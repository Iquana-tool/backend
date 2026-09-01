import logging

from fastapi import APIRouter, Depends, HTTPException
from iquana_toolbox.schemas.database.labels import Label
from sqlalchemy.orm import Session

from app.database import get_session
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.label_space import LabelSpaceDraft
from app.schemas.labels import LabelMoveRequest, LabelUpdate
from app.schemas.permissions import Permission
from app.services.database_access import label_moves, labels as labels_db
from app.services.database_access.label_moves import LabelMoveBlocked, LabelMoveError
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
        updates: LabelUpdate,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_MANAGE, "label_id")),
):
    """Rename a label.

    Only the name is patchable: re-parenting goes through ``/labels/{id}/move`` because
    it can invalidate existing annotations, and ``value`` is what mask encodings were
    written against.

    Args:
        label_id (int): The ID of the label to update.
        updates (LabelUpdate): The fields to change.
        user (AuthenticatedUser): The current authenticated user.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    await labels_db.update_label(label_id, updates.model_dump(exclude_unset=True), db)
    return {
        "success": True,
        "message": "Label updated successfully.",
    }


@router.post("/{label_id}/move")
async def move_label(
        label_id: int,
        request: LabelMoveRequest,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_MANAGE, "label_id")),
):
    """Move a label under a different parent, or to the top level.

    Nesting means part-of, and annotation enforces it: an object may only carry a label
    that is a direct part of the label on the object containing it. A move that would
    strand already-annotated objects is therefore refused with a 409 describing what it
    would break; repeating it with ``detach_affected`` demotes those objects to root
    level, keeping their label and dropping only the containment link.

    Args:
        label_id (int): The ID of the label to move.
        request (LabelMoveRequest): The destination, and whether to accept detaching.
        user (AuthenticatedUser): The current authenticated user.
        db (Session): The database session.

    Returns:
        dict: Success status, message, and how many objects were detached.
    """
    try:
        impact = await label_moves.move_label(
            db, label_id, request.new_parent_id, request.detach_affected
        )
    except LabelMoveError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LabelMoveBlocked as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": str(exc),
                "affected_count": exc.impact.count,
                # Capped: the dialog needs a sense of scale and a few examples to open,
                # not every contour in a fully annotated dataset.
                "affected_objects": [
                    {"contour_id": affected.contour_id, "image_id": affected.image_id}
                    for affected in exc.impact.affected[:20]
                ],
            },
        )

    return {
        "success": True,
        "message": "Label moved successfully.",
        "detached_count": impact.count,
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

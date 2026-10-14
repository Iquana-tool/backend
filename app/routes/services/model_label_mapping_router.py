"""Map a model's label space onto a dataset's labels.

``GET`` shows every class of the model with what it currently means in the dataset
(a saved choice, the same-name label it falls back to, or nothing). ``PUT`` changes
classes: map to a label, discard, add as a new label, or clear a saved choice.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from starlette import status as http_status

from app.database import get_session
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.model_label_mapping import ModelLabelMappingRead, ModelLabelMappingWrite
from app.schemas.permissions import Permission
from app.services import model_label_mapping
from app.services.auth import get_current_user
from app.services.permissions import ensure_permission

router = APIRouter(prefix="/model-label-mappings", tags=["model_label_mappings"])


@router.get("/{dataset_id}/{model_registry_key}", response_model=ModelLabelMappingRead)
def get_model_label_mapping(
    dataset_id: int,
    model_registry_key: str,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """The model's classes and what each one means in the dataset."""
    if not (
        user.has_permission(dataset_id, Permission.AI_BATCH_INFER)
        or user.has_permission(dataset_id, Permission.AI_INTERACTIVE)
    ):
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail=f"Missing permission to view model label mappings on dataset {dataset_id}.",
        )
    return model_label_mapping.get_mapping(db, dataset_id, model_registry_key)


@router.put("/{dataset_id}/{model_registry_key}", response_model=ModelLabelMappingRead)
def update_model_label_mapping(
    dataset_id: int,
    model_registry_key: str,
    body: ModelLabelMappingWrite,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Change what some of the model's classes mean in the dataset."""
    ensure_permission(user, dataset_id, Permission.AI_BATCH_INFER)
    if any(change.action == "create" for change in body.classes):
        ensure_permission(user, dataset_id, Permission.LABEL_MANAGE)
    return model_label_mapping.save_mapping(
        db, dataset_id, model_registry_key, body.classes, user.username
    )

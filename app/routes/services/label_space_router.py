import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_session
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.label_space import (
    GenerateLabelSpaceRequest,
    GenerateLabelSpaceResponse,
    LabelSpaceConfigResponse,
    RefineLabelSpaceRequest,
)
from app.schemas.permissions import Permission
from app.services import credentials as credentials_service
from app.services.ai_services.label_space import LabelSpaceService
from app.services.auth import get_current_user
from app.services.credentials import LlmConfig
from app.services.permissions import ensure_permission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/label_space", tags=["Label Space"])
service = LabelSpaceService()


def _llm_for(user: AuthenticatedUser, dataset_id: int | None, db: Session) -> LlmConfig:
    """The key and model for this caller's generation, or a 503 saying where to add one.

    With a dataset, its organisation's key applies -- so drafting for a dataset
    needs the right to change its labels.
    """
    if dataset_id is not None:
        ensure_permission(user, dataset_id, Permission.LABEL_MANAGE)
    config = credentials_service.resolve_llm(user, db, dataset_id)
    if config is None:
        raise HTTPException(
            status_code=503,
            detail="Label-space generation has no LLM key to use. Add a personal key to your "
                   "account, ask your organisation's admins for one, or an admin can set the "
                   "instance's under Admin -> Settings.",
        )
    return config


@router.get("/config", response_model=LabelSpaceConfigResponse)
async def get_config(dataset_id: int | None = None,
                     db: Session = Depends(get_session),
                     user: AuthenticatedUser = Depends(get_current_user)):
    """Report whether LLM-assisted generation is available to the caller, and whose key it uses."""
    if dataset_id is not None:
        ensure_permission(user, dataset_id, Permission.DATASET_READ)
    config = credentials_service.resolve_llm(user, db, dataset_id)
    if config is None:
        return LabelSpaceConfigResponse(enabled=False)
    return LabelSpaceConfigResponse(enabled=True, model=config.model, source=config.source)


@router.post("/generate", response_model=GenerateLabelSpaceResponse)
async def generate_label_space(
        request: GenerateLabelSpaceRequest,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(get_current_user),
):
    """Turn a plain-language description into a draft label hierarchy.

    This does not touch the labels -- the returned draft is reviewed and edited by
    the user, then persisted via ``POST /labels/bulk_create``.
    """
    config = _llm_for(user, request.dataset_id, db)
    draft = service.generate(
        config,
        description=request.description,
        max_depth=request.max_depth,
        max_labels=request.max_labels,
        model=request.model,
    )
    credentials_service.mark_used(config, db)
    return GenerateLabelSpaceResponse(draft=draft)


@router.post("/refine", response_model=GenerateLabelSpaceResponse)
async def refine_label_space(
        request: RefineLabelSpaceRequest,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(get_current_user),
):
    """Revise an existing draft according to a follow-up instruction."""
    config = _llm_for(user, request.dataset_id, db)
    draft = service.refine(
        config,
        current_draft=request.current_draft,
        message=request.message,
        description=request.description,
        max_depth=request.max_depth,
        max_labels=request.max_labels,
        model=request.model,
    )
    credentials_service.mark_used(config, db)
    return GenerateLabelSpaceResponse(draft=draft)

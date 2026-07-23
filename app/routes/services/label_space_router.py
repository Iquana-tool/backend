import logging

from fastapi import APIRouter, Depends
from iquana_toolbox.schemas.user import User

from app.schemas.label_space import (
    GenerateLabelSpaceRequest,
    GenerateLabelSpaceResponse,
    LabelSpaceConfigResponse,
    RefineLabelSpaceRequest,
)
from app.services.ai_services.label_space import LabelSpaceService
from app.services.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/label_space", tags=["Label Space"])
service = LabelSpaceService()


@router.get("/config", response_model=LabelSpaceConfigResponse)
async def get_config(user: User = Depends(get_current_user)):
    """Report whether LLM-assisted label-space generation is available."""
    enabled = service.is_enabled()
    return LabelSpaceConfigResponse(enabled=enabled, model=service.model() if enabled else None)


@router.post("/generate", response_model=GenerateLabelSpaceResponse)
async def generate_label_space(
        request: GenerateLabelSpaceRequest,
        user: User = Depends(get_current_user),
):
    """Turn a plain-language description into a draft label hierarchy.

    This does not touch the database — the returned draft is reviewed and edited
    by the user, then persisted via ``POST /labels/bulk_create``.
    """
    draft = service.generate(
        description=request.description,
        max_depth=request.max_depth,
        max_labels=request.max_labels,
        model=request.model,
    )
    return GenerateLabelSpaceResponse(draft=draft)


@router.post("/refine", response_model=GenerateLabelSpaceResponse)
async def refine_label_space(
        request: RefineLabelSpaceRequest,
        user: User = Depends(get_current_user),
):
    """Revise an existing draft according to a follow-up instruction."""
    draft = service.refine(
        current_draft=request.current_draft,
        message=request.message,
        description=request.description,
        max_depth=request.max_depth,
        max_labels=request.max_labels,
        model=request.model,
    )
    return GenerateLabelSpaceResponse(draft=draft)

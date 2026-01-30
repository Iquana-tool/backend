import os
from logging import getLogger

import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.params import Depends
from schemas.service_requests import SemanticSegmentationRequest
from schemas.training import SemanticTrainingRequest
from schemas.user import User
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse, StreamingResponse

from app.database import get_session
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.services.ai_services.semantic_segmentation import segment_image_with_semantic_model, \
    SemanticSegmentationService
from app.services.auth import get_current_user
from app.services.util import get_mask_path_from_image_path
from paths import SEMANTIC_SEGMENTATION_BACKEND_URL as BASE_URL

logger = getLogger(__name__)
router = APIRouter(prefix="/semantic_segmentation", tags=["semantic_segmentation"])
service: SemanticSegmentationService = SemanticSegmentationService()


@router.post("/run")
async def run_inference(
        request: SemanticSegmentationRequest,
        user: User = Depends(get_current_user),
        db: Session = Depends(get_session)):
    """  Run inference on a single image. """
    return await service.inference(request)


@router.get("/models")
async def get_models(
        user: User = Depends(get_current_user)
):
    """Retrieve all available models for this dataset."""
    return await service.get_models()


@router.delete("/models/{model_registry_key}")
async def delete_model(model_registry_key: str,
                       user: User = Depends(get_current_user)):
    """ Delete a model based on its id. """
    return await service.delete_model(model_registry_key)


@router.get("/training/{task_id}")
async def get_training_status(
        model_registry_key: str,
        user: User = Depends(get_current_user)
):
    """ Get the status of a training job by its ID. """
    raise NotImplementedError("This is currently not implemented.")


@router.get("/training/{task_id}/stream")
async def get_training_status_stream(
        task_id: int,
        user: User = Depends(get_current_user)
):
    """ Get live updates of the status of a training job by its ID. """
    raise NotImplementedError("This is currently not implemented.")


@router.delete("/training/{task_id}")
async def cancel_training_of_model(
        model_id: str,
        user: User = Depends(get_current_user)
):
    """ Cancel a training job by its ID."""
    raise NotImplementedError("This is currently not implemented.")


@router.post("/training/start")
async def start_training(
        request: SemanticTrainingRequest,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """ Start training a model for automatic prompted_segmentation.
    This endpoint prepares the request with necessary parameters and forwards it to the Automatic Segmentation Service.
    Args:
        request (TrainingRequest): The training request containing dataset_id and other parameters.
        db (Session): The database session for querying labels.
        user (User): The current authenticated user.

    Returns:
        JSONResponse: The response from the Automatic Segmentation Service.
    """
    return await service.start_training(request)

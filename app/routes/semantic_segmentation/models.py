import json

from fastapi import Depends, APIRouter
from fastapi.responses import JSONResponse
import httpx
from logging import getLogger
from sqlalchemy.orm import Session
from app.database import get_session
from paths import AUTOMATIC_SEGMENTATION_BACKEND_URL as BASE_URL


logger = getLogger(__name__)
router = APIRouter(prefix="/semantic_segmentation", tags=["semantic_segmentation"])


@router.get("/get_available_base_models")
async def get_available_base_models():
    """Retrieve all available base models for training."""
    url = f"{BASE_URL}/models/get_trainable_base_models"
    logger.info(url)
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return JSONResponse(resp.json())


@router.get("/get_trained_models_of_dataset/{dataset_id}")
async def get_trained_models_of_dataset(dataset_id: int):
    """Retrieve all trained models for a specific dataset."""
    url = f"{BASE_URL}/models/get_trained_models_of_dataset/{dataset_id}"  # Gets all already trained models
    logger.info(url)
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return JSONResponse(resp.json())


@router.get("/get_training_models")
async def get_training_models(dataset_id: int):
    """Retrieve all models that are currently being trained."""
    raise NotImplementedError("This endpoint is not implemented yet. "
                              "It should return models that are currently being trained.")


@router.get("/get_model_metadata/{model_id}")
async def get_model_metadata(model_id: int):
    """Retrieve metadata for a specific model. Metadata includes training status and training info, as well as general
    model information."""
    logger.debug(f"Fetching metadata for model ID {model_id}.")
    url = f"{BASE_URL}/models/get_model_metadata/{model_id}"  # Gets all already trained models
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return JSONResponse(resp.json())

@router.delete("/delete_model/{model_id}")
async def delete_model(model_id: int):
    """ Delete a model based on its id. """
    logger.debug(f"Deleting model with id: {model_id}")
    url = f"{BASE_URL}/models/delete_model/{model_id}"
    async with httpx.AsyncClient() as client:
        resp = await client.delete(url)
        resp.raise_for_status()
        return JSONResponse(resp.json())
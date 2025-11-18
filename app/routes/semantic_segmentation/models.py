from logging import getLogger

import httpx
from fastapi import Depends, APIRouter
from fastapi.responses import JSONResponse

from app.schemas.user import User
from app.services.auth import get_current_user
from paths import AUTOMATIC_SEGMENTATION_BACKEND_URL as BASE_URL

logger = getLogger(__name__)
router = APIRouter(prefix="/semantic_segmentation", tags=["semantic_segmentation"])


@router.get("/get_models/dataset={dataset_id}")
async def get_models(dataset_id: int,
                     user: User = Depends(get_current_user)):
    """Retrieve all available models for this dataset."""
    url = f"{BASE_URL}/models/get_models/type={'all'}&available={True}&dataset_id={dataset_id}"
    logger.info(url)
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return JSONResponse(resp.json())


@router.delete("/delete_model/{model_id}")
async def delete_model(model_id: int,
                       user: User = Depends(get_current_user)):
    """ Delete a model based on its id. """
    logger.debug(f"Deleting model with id: {model_id}")
    url = f"{BASE_URL}/models/delete_model/{model_id}"
    async with httpx.AsyncClient() as client:
        resp = await client.delete(url)
        resp.raise_for_status()
        return JSONResponse(resp.json())

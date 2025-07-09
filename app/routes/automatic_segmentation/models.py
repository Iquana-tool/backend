import json

from fastapi import Depends, APIRouter
from fastapi.responses import JSONResponse
import httpx
from logging import getLogger
from sqlalchemy.orm import Session
from app.database import get_session
from app.database.models import Models
from paths import AUTOMATIC_SEGMENTATION_BACKEND_URL as BASE_URL


logger = getLogger(__name__)
router = APIRouter(prefix="/automatic_segmentation", tags=["automatic_segmentation"])


@router.get("/available_base_models")
async def get_available_base_models():
    url = f"{BASE_URL}/models/get_trainable_base_models"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return JSONResponse(resp.json())


@router.get("/trained_models_of_dataset")
async def get_trained_models_of_dataset():
    url = f"{BASE_URL}/models/get_trained_models"  # Gets all already trained models
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return JSONResponse(resp.json())


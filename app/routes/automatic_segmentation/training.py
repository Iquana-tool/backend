from paths import AUTOMATIC_SEGMENTATION_BACKEND_URL as BASE_URL
from app.schemas.automatic_segmentation.training import TrainingRequest
import httpx
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from logging import getLogger


logger = getLogger(__name__)
router = APIRouter(prefix="/automatic_segmentation", tags=["automatic_segmentation"])


@router.post("/start_training")
async def start_training(request: TrainingRequest):
    result = await send_start_training_request(request)
    return JSONResponse(result)

async def send_start_training_request(request: TrainingRequest):
    """ Forwards to Automatic Segmentation Service."""
    url = f"{BASE_URL}/training/start_training"
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=request.model_dump())
        resp.raise_for_status()
        return resp.json()

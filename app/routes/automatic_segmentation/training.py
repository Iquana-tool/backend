from . import router, logger
from paths import AUTOMATIC_SEGMENTATION_BACKEND_URL as BASE_URL
from app.schemas.automatic_segmentation.training import TrainingRequest
import httpx


@router.post("/start_training")
async def start_training(request: TrainingRequest):
    response = await send_start_training_request(request)
    pass


async def send_start_training_request(request: TrainingRequest):
    url = f"{BASE_URL}/start_training"
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, data=request)
        resp.raise_for_status()
        return resp.json()

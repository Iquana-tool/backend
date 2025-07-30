from fastapi.params import Depends
from sqlalchemy.orm import Session
from app.database import get_session
from app.database.labels import Labels
from paths import AUTOMATIC_SEGMENTATION_BACKEND_URL as BASE_URL
from app.schemas.automatic_segmentation.training import TrainingRequest
import httpx
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from logging import getLogger


logger = getLogger(__name__)
router = APIRouter(prefix="/automatic_segmentation", tags=["automatic_segmentation"])


@router.get("/get_training_status/{job_id}")
async def get_training_status(job_id: str):
    url = f"{BASE_URL}/training/get_job_status/{job_id}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


@router.get("/cancel_job/{job_id}")
async def cancel_training_status(job_id: str):
    try:
        url = f"{BASE_URL}/training/cancel_job/{job_id}"
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        return {
            "success": False,
            "message": f"Could not cancel job {job_id}, probably due to computational overload."
        }


@router.post("/start_training")
async def start_training(request: TrainingRequest, db: Session = Depends(get_session)):
    # Get the request as a dict, to add values to it
    request_dict = request.model_dump()

    # Find out the number of classes present. + 1 for background
    # Caution: This will get all labels that the user created, even if they did not annotate them anywhere.
    num_classes = db.query(Labels).filter_by(dataset_id=request.dataset_id).count() + 1
    request_dict["num_classes"] = num_classes

    # Abstract this from the frontend, user should not be concerned with this.
    request_dict["in_channels"] = 3
    request_dict["batch_size"] = 32
    request_dict["lr"] = 0.0001
    result = await send_start_training_request(request_dict)
    return JSONResponse(result)

async def send_start_training_request(request: dict):
    """ Forwards to Automatic Segmentation Service."""
    logger.info(f"Start training request: {request}")
    url = f"{BASE_URL}/training/start_training"
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=request)
        if not resp.is_success:
            logger.error(f"Training request failed: {resp.text}")
        resp.raise_for_status()
        return resp.json()

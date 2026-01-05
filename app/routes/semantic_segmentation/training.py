from logging import getLogger

import httpx
from fastapi import APIRouter
from fastapi.params import Depends
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.labels import Labels
from app.schemas.semantic_segmentation.training import TrainingRequest
from app.schemas.user import User
from app.services.auth import get_current_user
from paths import SEMANTIC_SEGMENTATION_BACKEND_URL as BASE_URL

logger = getLogger(__name__)
router = APIRouter(prefix="/semantic_segmentation", tags=["semantic_segmentation"])


@router.get("/get_training_status/{model_registry_key}")
async def get_training_status(
        model_registry_key: str,
        user: User = Depends(get_current_user)
):
    """ Get the status of a training job by its ID. """
    url = f"{BASE_URL}/training/get_job_status/{model_registry_key}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


@router.get("/get_training_status_stream/{model_registry_key}")
async def get_training_status_stream(
        model_registry_key: str,
        user: User = Depends(get_current_user)
):
    """ Get live updates of the status of a training job by its ID. """
    url = f"{BASE_URL}/training/get_status_stream/{model_registry_key}"
    async with httpx.AsyncClient() as client:
        # Use `stream=True` to get a streaming response
        async with client.stream("GET", url) as response:
            # Forward the stream as-is
            return StreamingResponse(
                response.aiter_bytes(),  # or response.aiter_text() for text streams
                media_type=response.headers.get("content-type", "application/octet-stream"),
                headers=dict(response.headers)
            )


@router.get("/cancel_training_of_model/{model_id}")
async def cancel_training_of_model(
        model_id: str,
        user: User = Depends(get_current_user)
):
    """ Cancel a training job by its ID."""
    try:
        url = f"{BASE_URL}/training/cancel_job/{model_id}"
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        return {
            "success": False,
            "message": f"Could not cancel job {model_id}, probably due to computational overload."
        }


@router.post("/start_training")
async def start_training(
        request: TrainingRequest,
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
    """ Forwards TrainingRequest to Automatic Segmentation Service."""
    logger.info(f"Start training request: {request}")
    url = f"{BASE_URL}/training/start_training"
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=request)
        if not resp.is_success:
            logger.error(f"Training request failed: {resp.text}")
        resp.raise_for_status()
        return resp.json()

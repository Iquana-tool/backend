from logging import getLogger

from fastapi import APIRouter, Depends
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.user import User

from app.services.auth import get_current_user

logger = getLogger(__name__)
router = APIRouter(prefix="/semantic_segmentation", tags=["semantic_segmentation"])

# NOTE: Instance segmentation is not implemented yet. Every endpoint below is a stub that
# raises NotImplementedError until the service is built out. Keep the routes registered so
# the API surface is visible and the wiring stays in place.


@router.post("/run")
async def run_inference(
        request: InstanceSegmentationRequest,
        user: User = Depends(get_current_user),
):
    """ Run inference on a single image. """
    raise NotImplementedError("Instance segmentation inference is not implemented yet.")


@router.get("/models")
async def get_models(
        user: User = Depends(get_current_user),
):
    """Retrieve available instance segmentation models."""
    raise NotImplementedError("Listing instance segmentation models is not implemented yet.")


@router.delete("/models/{model_registry_key}")
async def delete_model(
        model_registry_key: str,
        user: User = Depends(get_current_user),
):
    """ Delete a model based on its id. """
    raise NotImplementedError("Deleting an instance segmentation model is not implemented yet.")


@router.get("/training/{task_id}")
async def get_training_status(
        task_id: str,
        user: User = Depends(get_current_user),
):
    """ Get the status of a training job by its ID. """
    raise NotImplementedError("Instance segmentation training status is not implemented yet.")


@router.get("/training/{task_id}/stream")
async def get_training_status_stream(
        task_id: str,
        user: User = Depends(get_current_user),
):
    """ Stream status updates for a training job. """
    raise NotImplementedError("Instance segmentation training stream is not implemented yet.")


@router.delete("/training/{task_id}")
async def cancel_training_of_model(
        task_id: str,
        user: User = Depends(get_current_user),
):
    """ Cancel a training job by its ID. """
    raise NotImplementedError("Cancelling instance segmentation training is not implemented yet.")


@router.post("/training/start")
async def start_training(
        model_registry_key: str,
        dataset_id: int | str,
        user: User = Depends(get_current_user),
):
    """ Start training an instance segmentation model. """
    raise NotImplementedError("Starting instance segmentation training is not implemented yet.")

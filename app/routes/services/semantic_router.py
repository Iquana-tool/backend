import json
from logging import getLogger

from celery.result import AsyncResult
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from iquana_toolbox.schemas.service_requests import SemanticSegmentationRequest
from iquana_toolbox.schemas.training import SemanticTrainingRequest, TrainingProgress
from iquana_toolbox.schemas.user import User
from redis.asyncio import Redis
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

from app.database import get_session
from app.services.ai_services.semantic_segmentation import SemanticSegmentationService
from app.services.auth import get_current_user
from app.services.redis import get_redis

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


@router.get("/training/{task_id}",
            deprecated=True,
            description="Queries the celery backend to get an update on the training status. This means overhead for "
                        "celery, please use the stream endpoint instead. It subscribes to a redis publisher.")
async def get_training_status(
        task_id: str,
        user: User = Depends(get_current_user)
):
    """ Get the status of a training job by its ID. Accesses the celery queue. """
    result = AsyncResult(task_id)

    return {
        "success": True,
        "message": "Successfully fetched training status.",
        "result": {
            "task_id": task_id,
            "status": result.status,  # PENDING, STARTED, SUCCESS, FAILURE
            "info": result.info if not isinstance(result.info, Exception) else str(result.info)
        }
    }


@router.get("/training/{task_id}/stream")
async def get_training_status_stream(
        task_id: int,
        user: User = Depends(get_current_user),
        redis_client: Redis = Depends(get_redis)
):
    """
    Streams status updates by subscribing to a Redis channel.
    Channel name: task_progress_{task_id}
    """
    channel_name = f"task_progress_{task_id}"

    async def event_generator():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel_name)

        try:
            # You might want to send an initial "connected" message
            yield f"data: {json.dumps({'status': 'connected', 'channel': channel_name})}\n\n"

            async for message in pubsub.listen():
                if message["type"] == "message":
                    progress = TrainingProgress.model_validate(message["data"])
                    yield progress.model_dump_json() + "\n"

                    if progress.status != "PROGRESS":
                        break

        finally:
            await pubsub.unsubscribe(channel_name)
            await pubsub.close()

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@router.delete("/training/{task_id}")
async def cancel_training_of_model(
        task_id: str,
        user: User = Depends(get_current_user)
):
    """ Cancel a training job by its ID."""
    result = AsyncResult(task_id)

    # terminate=True sends a SIGTERM to the worker process
    result.revoke(terminate=True)

    return {"message": f"Task {task_id} has been revoked."}


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

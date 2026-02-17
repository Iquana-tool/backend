import json
import os
from logging import getLogger

from celery.result import AsyncResult
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from iquana_toolbox.schemas.labels import LabelHierarchy
from iquana_toolbox.schemas.service_requests import SemanticSegmentationRequest
from iquana_toolbox.schemas.training import SemanticTrainingRequest, TrainingProgress, SemanticTrainingConfig
from iquana_toolbox.schemas.user import User
from redis.asyncio import Redis
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.services.celery_app import celery_app
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
            description="Queries the celery backend to get an update on the training status. For continuous updates, please "
                        "use the /training/{task_id}/stream endpoint.")
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
            "progress": result.info
        }
    }


@router.get("/training/{task_id}/stream")
async def get_training_status_stream(
        task_id: str,
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
        model_registry_key: str,
        dataset_id: int | str,
        training_config: SemanticTrainingConfig,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """ Start training a model for automatic prompted_segmentation.
    This endpoint prepares the request with necessary parameters and forwards it to the Automatic Segmentation Service.
    """
    # Build the training request, which consists of the training_data and training_config
    # First get all images and masks urls of the dataset
    file_urls = (db.query(
        Images.file_path.label("image_url"),
        Masks.file_path.label("mask_url")
    ).join(
        Masks, Masks.image_id == Images.id
    ).filter(
        Images.dataset_id == dataset_id,
        Masks.fully_annotated == True,
    ).all())
    labels = db.query(Labels).filter_by(dataset_id=dataset_id)
    label_hierarchy = LabelHierarchy.from_query(labels)

    if len(file_urls) == 0:
        return {
            "success": False,
            "message": f"No data to train on for dataset {dataset_id}!"
        }

    request = SemanticTrainingRequest(
        model_registry_key=model_registry_key,
        image_urls=[row.image_url for row in file_urls],
        mask_urls=[row.mask_url for row in file_urls],
        label_hierarchy=label_hierarchy,
        **training_config.model_dump(),
    )
    task = celery_app.send_task(
        "semantic_segmentation.train_model",
        args=[request.model_dump_json()],
        queue="semantic_queue"
    )
    return {
        "success": True,
        "message": "Training task enqueued.",
        "result": {"task_id": task.task_id, "state": task.state, "data": task.info}
    }

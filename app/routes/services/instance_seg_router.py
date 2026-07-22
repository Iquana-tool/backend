import asyncio
import json
import os
from logging import getLogger
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest
from iquana_toolbox.schemas.user import User
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette import status as http_status

from app.database import get_session
from app.database.images import Images
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services.ai_services.instance_segmentation import InstanceSegmentationService
from app.services.auth import get_current_user
from app.services.database_access import datasets as datasets_db
from app.services.database_access import labels as labels_db
from app.services.database_access.datasets import export_dataset_contours_to_coco
from app.services.model_registry import MODEL_REGISTRY, list_available_models
from app.services.permissions import ensure_permission, require

logger = getLogger(__name__)
router = APIRouter(prefix="/instance_segmentation", tags=["instance_segmentation"])
service = InstanceSegmentationService()

# Default fine-tuning model. Only one instance-segmentation model exists for now.
DEFAULT_MODEL_REGISTRY_KEY = "mask2former"

# Must match instance-segmentation-service/app/tasks.py:TRAINING_EXPERIMENT. The
# worker tags each training run with ``celery_task_id`` so we can map a task id to
# its MLflow run.
TRAINING_EXPERIMENT = "instance-segmentation-training"

# Map MLflow run statuses to the coarse state the frontend renders.
_STATE_BY_MLFLOW_STATUS = {
    "FINISHED": "SUCCESS",
    "FAILED": "FAILED",
    "KILLED": "FAILED",
}


class StartTrainingBody(BaseModel):
    """Training configuration sent by the Model Training page."""
    dataset_id: int = Field(..., description="Dataset to train on.")
    label_ids: list[int] = Field(
        default_factory=list,
        description="Labels (classes) to train on. Empty means all labels in the dataset hierarchy (multiclass).",
    )
    model_registry_key: str = Field(DEFAULT_MODEL_REGISTRY_KEY, description="Base model to fine-tune.")
    # Model-declared hyperparameters (keys match the model's training_parameters),
    # forwarded as-is to the training request.
    hyper_parameter: dict = Field(default_factory=dict, description="Hyperparameter overrides keyed by model param key.")


@router.get("/models")
async def get_models(user: User = Depends(get_current_user)):
    """Retrieve available instance segmentation models directly from MLflow."""
    return list_available_models("instance-segmentation")


@router.post("/training/start")
async def start_training(
        body: StartTrainingBody,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user),
):
    """Start training an instance segmentation model on a dataset.

    Exports the dataset's annotations to a COCO file on the shared data volume,
    then dispatches a training job to the instance-segmentation service (which
    delegates to a Celery worker). Returns the task id, which is also the MLflow
    run id used to poll progress.
    """
    # The dataset id only exists after the body is parsed, so this is the
    # imperative form of the require() dependency used elsewhere.
    ensure_permission(user, body.dataset_id, Permission.AI_TRAIN)

    dataset = await datasets_db.get_dataset(body.dataset_id, db=db)
    if not dataset:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Dataset not found.")

    # The COCO file_name is a basename, so the worker needs the directory that
    # actually holds the image files. Derive it from a stored image path rather than
    # assuming it equals the dataset root (images live under e.g. <root>/images/).
    sample_image = db.query(Images).filter_by(dataset_id=body.dataset_id).first()
    if sample_image is None:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST,
                            detail="The dataset has no images to train on.")
    image_folder_path = os.path.dirname(str(sample_image.file_path))

    # Resolve the labels to train on. Empty selection -> every label in the dataset
    # hierarchy (multiclass, "predict everything"); the single-class case is just a
    # one-element selection.
    hierarchy = await labels_db.get_label_hierarchy(body.dataset_id, db=db)
    if body.label_ids:
        missing = [lid for lid in body.label_ids if lid not in hierarchy.id_to_label_object]
        if missing:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail=f"Labels not found in dataset: {missing}.")
        labels = [hierarchy.id_to_label_object[lid] for lid in body.label_ids]
    else:
        labels = list(hierarchy.id_to_label_object.values())
    if not labels:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST,
                            detail="The dataset has no labels to train on.")

    # Write the COCO annotation file to disk so the worker can read it from the
    # shared data volume. contour_selection="all" emits every hierarchy level so the
    # model sees training examples for every class (parents overlap their children).
    export = await export_dataset_contours_to_coco(
        body.dataset_id, db, contour_selection="all", write_to_disk=True
    )
    if not export.get("success"):
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST,
                            detail=export.get("message", "Failed to export annotations."))
    if export.get("num_annotations", 0) == 0:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST,
                            detail="Dataset has no reviewed annotations to train on.")

    request = InstanceSegmentationTrainingRequest(
        dataset_id=body.dataset_id,
        image_folder_path=image_folder_path,
        model_registry_key=body.model_registry_key,
        user_id=user.username,
        labels=labels,
        annotation_file_url=export["output_file_path"],
        hyper_parameter=dict(body.hyper_parameter),
    )

    try:
        result = await service.start_training(request)
    except Exception as exc:
        logger.exception("Failed to start instance segmentation training.")
        raise HTTPException(status_code=http_status.HTTP_502_BAD_GATEWAY,
                            detail=f"Could not start training: {exc}")

    return {"success": True, "message": "Training started.", "task_id": result.get("task_id")}


def _find_training_run(task_id: str):
    """Find the MLflow run a Celery training task logged to, via its tag.

    The worker can't force the run id to equal the task id, so it tags the run
    with ``celery_task_id``. Returns the most recent matching run (handles task
    retries, which create a fresh run under the same tag) or ``None`` if the task
    hasn't started a run yet.
    """
    client = MODEL_REGISTRY.client
    experiment = client.get_experiment_by_name(TRAINING_EXPERIMENT)
    if experiment is None:
        return None
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.celery_task_id = '{task_id}'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    return runs[0] if runs else None


def _parse_label_ids(raw) -> list[int]:
    """Parse the stringified ``label_ids`` tag (e.g. "[1, 2]") back to a list."""
    if not raw:
        return []
    try:
        import ast
        parsed = ast.literal_eval(raw)
        return [int(v) for v in parsed]
    except (ValueError, SyntaxError, TypeError):
        return []


def _snapshot_from_run(client, run, task_id: Optional[str] = None) -> dict:
    """Build a progress snapshot dict from an MLflow run."""
    run_id = run.info.run_id
    mlflow_status = run.info.status
    state = _STATE_BY_MLFLOW_STATUS.get(mlflow_status, "PROGRESS")

    try:
        loss_history = client.get_metric_history(run_id, "loss")
    except Exception:
        loss_history = []
    loss = [{"epoch": int(m.step), "value": m.value}
            for m in sorted(loss_history, key=lambda m: m.step)]

    total_epochs = run.data.params.get("epochs")
    total_epochs = int(total_epochs) if total_epochs is not None else None

    epoch_metric = run.data.metrics.get("epoch")
    epoch = int(epoch_metric) if epoch_metric is not None else (loss[-1]["epoch"] if loss else 0)

    return {
        "task_id": task_id if task_id is not None else run.data.tags.get("celery_task_id"),
        "run_id": run_id,
        "state": state,
        "mlflow_status": mlflow_status,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "loss": loss,
        "label_ids": _parse_label_ids(run.data.tags.get("label_ids")),
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
    }


def _read_training_snapshot(task_id: str) -> dict:
    """Read a progress snapshot for a Celery ``task_id`` straight from MLflow.

    Returns a ``"starting"`` snapshot while no run exists yet (task queued but not
    picked up by a worker).
    """
    run = _find_training_run(task_id)
    if run is None:
        return {"task_id": task_id, "run_id": None, "state": "starting", "mlflow_status": None,
                "epoch": 0, "total_epochs": None, "loss": [], "label_ids": []}
    return _snapshot_from_run(MODEL_REGISTRY.client, run, task_id)


def _read_run_snapshot(run_id: str) -> dict:
    """Read a progress snapshot for a specific MLflow ``run_id`` (a past run)."""
    client = MODEL_REGISTRY.client
    run = client.get_run(run_id)
    return _snapshot_from_run(client, run)


def _list_training_runs(dataset_id: int) -> list[dict]:
    """List training runs for a dataset (newest first) as lightweight summaries."""
    client = MODEL_REGISTRY.client
    experiment = client.get_experiment_by_name(TRAINING_EXPERIMENT)
    if experiment is None:
        return []
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.dataset_id = '{dataset_id}'",
        order_by=["attributes.start_time DESC"],
        max_results=100,
    )
    return [_snapshot_from_run(client, run) for run in runs]


@router.get("/training/runs")
async def list_training_runs(dataset_id: int,
                             user: AuthenticatedUser = Depends(require(Permission.AI_TRAIN))):
    """List past + active training runs for a dataset (for the run-history list)."""
    return {"success": True, "runs": await asyncio.to_thread(_list_training_runs, dataset_id)}


@router.get("/training/runs/{run_id}")
async def get_run_snapshot(run_id: str, user: User = Depends(get_current_user)):
    """Return a progress snapshot for a specific (e.g. past) MLflow run."""
    return await asyncio.to_thread(_read_run_snapshot, run_id)


@router.get("/training/{task_id}")
async def get_training_status(task_id: str, user: User = Depends(get_current_user)):
    """Return a single MLflow-backed progress snapshot for a training job."""
    return await asyncio.to_thread(_read_training_snapshot, task_id)


@router.get("/training/{task_id}/stream")
async def get_training_status_stream(task_id: str, user: User = Depends(get_current_user)):
    """Stream MLflow-backed progress for a training job as Server-Sent Events.

    Polls the MLflow run every couple of seconds and emits one ``data:`` event per
    tick until the run reaches a terminal state (FINISHED/FAILED/KILLED).
    """

    async def event_generator():
        while True:
            snapshot = await asyncio.to_thread(_read_training_snapshot, task_id)
            yield f"data: {json.dumps(snapshot)}\n\n"
            if snapshot["state"] in ("SUCCESS", "FAILED"):
                return
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/training/{task_id}")
async def cancel_training_of_model(task_id: str, user: User = Depends(get_current_user)):
    """Cancel (revoke) a running training job by its task id."""
    try:
        return await service.cancel_training(task_id)
    except Exception as exc:
        logger.exception("Failed to cancel instance segmentation training.")
        raise HTTPException(status_code=http_status.HTTP_502_BAD_GATEWAY,
                            detail=f"Could not cancel training: {exc}")


@router.post("/run")
async def run_inference(
        request: InstanceSegmentationRequest,
        user: AuthenticatedUser = Depends(get_current_user),
):
    """Run inference on a single image.

    NOTE: this endpoint cannot be dataset-scoped as it stands. The request carries
    an `image_url` rather than an `image_id`, so there is nothing to resolve a
    dataset from, and it is authenticated-only. Adding `image_id` to
    `InstanceSegmentationRequest` in the toolbox would let this take
    `require(Permission.AI_BATCH_INFER, "image_id")` like the rest.
    """
    return await service.inference(request)

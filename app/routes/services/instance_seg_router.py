import asyncio
import json
import os
import re
from logging import getLogger
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest
from iquana_toolbox.schemas.user import User
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette import status as http_status

from app.database import get_session
from app.database.contours import Contours
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

# Canonical-to-public lifecycle state mapping.
_CANONICAL_TO_PUBLIC_STATE = {
    "starting": "STARTING",
    "running": "PROGRESS",
    "completed": "SUCCESS",
    "failed": "FAILED",
    "cancelled": "CANCELLED",
    "timed_out": "TIMED_OUT",
}

# Legacy MLflow status fallback mapping when training_state tag is absent.
_LEGACY_MLFLOW_STATUS_TO_PUBLIC_STATE = {
    "RUNNING": "PROGRESS",
    "FINISHED": "SUCCESS",
    "FAILED": "FAILED",
    "KILLED": "CANCELLED",
}

_TERMINAL_STATES = {"SUCCESS", "FAILED", "CANCELLED", "TIMED_OUT"}
_VALIDATION_METRIC_PATTERN = re.compile(
    r"^val_mask_(iou|f1|precision|recall)_label_(\d+)$"
)


def _parse_timestamp(raw_value) -> float | None:
    """Safely parse optional numeric queue timestamps (e.g. Unix seconds)."""
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (ValueError, TypeError):
        return None


def _map_training_state_to_public_state(
    training_state: Optional[str], mlflow_status: Optional[str] = None
) -> str:
    """Map canonical/durable training state to public state with legacy MLflow fallback."""
    canonical = str(training_state).lower().strip() if training_state else None

    # If MLflow status is terminal (FINISHED, FAILED, KILLED) but the training_state tag is still
    # non-terminal (starting, running), the terminal MLflow status takes precedence over the stale tag.
    if mlflow_status in _LEGACY_MLFLOW_STATUS_TO_PUBLIC_STATE and mlflow_status != "RUNNING":
        legacy_state = _LEGACY_MLFLOW_STATUS_TO_PUBLIC_STATE[mlflow_status]
        if (
            canonical not in _CANONICAL_TO_PUBLIC_STATE
            or _CANONICAL_TO_PUBLIC_STATE[canonical] not in _TERMINAL_STATES
        ):
            return legacy_state

    if canonical and canonical in _CANONICAL_TO_PUBLIC_STATE:
        return _CANONICAL_TO_PUBLIC_STATE[canonical]
    if mlflow_status:
        return _LEGACY_MLFLOW_STATUS_TO_PUBLIC_STATE.get(mlflow_status, "PROGRESS")
    return "STARTING"



def _empty_snapshot(task_id: str, ai_status: Optional[dict] = None) -> dict:
    """Build an initial/fallback snapshot when no MLflow run exists yet."""
    ai_status = ai_status or {}
    raw_training_state = ai_status.get("training_state")
    celery_or_effective_state = ai_status.get("state")

    if raw_training_state:
        training_state = str(raw_training_state).lower().strip()
        state = _map_training_state_to_public_state(training_state)
    elif celery_or_effective_state == "REVOKED":
        state = "CANCELLED"
        training_state = "cancelled"
    elif celery_or_effective_state == "FAILURE":
        state = "FAILED"
        training_state = "failed"
    elif celery_or_effective_state == "SUCCESS":
        state = "SUCCESS"
        training_state = "completed"
    else:
        state = "STARTING"
        training_state = "starting"

    mlflow_status = None
    if state in {"CANCELLED", "TIMED_OUT"}:
        mlflow_status = "KILLED"
    elif state == "FAILED":
        mlflow_status = "FAILED"
    elif state == "SUCCESS":
        mlflow_status = "FINISHED"

    return {
        "task_id": task_id,
        "run_id": ai_status.get("run_id"),
        "state": state,
        "training_state": training_state,
        "message": ai_status.get("message"),
        "queued_at": _parse_timestamp(ai_status.get("queued_at")),
        "start_deadline": _parse_timestamp(ai_status.get("start_deadline")),
        "started_at": ai_status.get("started_at"),
        "mlflow_status": mlflow_status,
        "epoch": 0,
        "total_epochs": None,
        "training_parameters": {},
        "loss": [],
        "validation_metrics": None,
        "validation_metrics_unavailable": None,
        "label_ids": [],
        "run_name": None,
        "start_time": None,
        "end_time": None,
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
    # Optional human-readable label for the run (e.g. "Cells-FineTuned-v1").  Stored
    # as an MLflow tag so it appears in the run history and can be searched.  Kept
    # optional with no default so the worker's run_name still reads as a useful
    # auto-generated fallback when the field is absent.
    model_run_name: Optional[str] = Field(
        default=None,
        max_length=80,
        description="Optional human-readable name/alias for this training run.",
        pattern=r"^[\w\-\s]{1,80}$",
    )


@router.get("/models")
async def get_models(user: User = Depends(get_current_user)):
    """Retrieve available instance segmentation models directly from MLflow."""
    return await asyncio.to_thread(list_available_models, "instance-segmentation")


@router.get("/training/label-annotation-counts")
async def get_label_annotation_counts(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.AI_TRAIN)),
):
    """Return the number of reviewed, fully annotated annotations per label.

    Used by the training UI to show a pre-flight annotation count beside each
    class selector and disable "Start Training" when no reviewed annotations exist.
    A contour is considered training-ready when at least one user has reviewed it
    and its mask is marked fully annotated, matching the default COCO export.
    """
    from sqlalchemy import func

    from app.database.masks import Masks

    # Contours → Masks → Images to reach dataset_id. These filters match the
    # default COCO export: only fully annotated masks and reviewed contours.
    rows = (
        db.query(Contours.label_id, func.count(Contours.id).label("count"))
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
        .filter(Masks.fully_annotated.is_(True))
        .filter(Contours.reviewed_by.any())
        .group_by(Contours.label_id)
        .all()
    )
    counts = {row.label_id: row.count for row in rows}
    return {"success": True, "reviewed_annotation_counts": counts}


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
        result = await service.start_training(
            request,
            model_run_name=body.model_run_name,
            dataset_name=dataset.name,
        )
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


def _validation_metrics_from_run(run) -> dict | None:
    """Expose final held-out quality metrics in a UI-friendly shape."""
    per_label: dict[int, dict[str, float | int]] = {}
    for metric_name, value in run.data.metrics.items():
        match = _VALIDATION_METRIC_PATTERN.fullmatch(metric_name)
        if match is None:
            continue
        metric, raw_label_id = match.groups()
        label_id = int(raw_label_id)
        per_label.setdefault(label_id, {"label_id": label_id})[metric] = float(value)

    macro_iou = run.data.metrics.get("val_mask_iou_macro")
    macro_f1 = run.data.metrics.get("val_mask_f1_macro")
    macro_precision = run.data.metrics.get("val_mask_precision_macro")
    macro_recall = run.data.metrics.get("val_mask_recall_macro")
    ap = run.data.metrics.get("val_mask_ap")
    ap50 = run.data.metrics.get("val_mask_ap50")
    ap75 = run.data.metrics.get("val_mask_ap75")
    if not per_label and all(
        metric is None
        for metric in (
            macro_iou,
            macro_f1,
            macro_precision,
            macro_recall,
            ap,
            ap50,
            ap75,
        )
    ):
        return None

    return {
        "ap": float(ap) if ap is not None else None,
        "ap50": float(ap50) if ap50 is not None else None,
        "ap75": float(ap75) if ap75 is not None else None,
        "macro_iou": float(macro_iou) if macro_iou is not None else None,
        "macro_f1": float(macro_f1) if macro_f1 is not None else None,
        "macro_precision": (
            float(macro_precision) if macro_precision is not None else None
        ),
        "macro_recall": float(macro_recall) if macro_recall is not None else None,
        "per_label": [per_label[label_id] for label_id in sorted(per_label)],
    }


def _snapshot_from_run(
    client, run, task_id: Optional[str] = None, ai_status: Optional[dict] = None
) -> dict:
    """Build a progress snapshot dict from an MLflow run."""
    run_id = run.info.run_id
    mlflow_status = run.info.status
    tags = getattr(run.data, "tags", {}) or {}
    ai_status = ai_status or {}

    raw_training_state = tags.get("training_state") or ai_status.get("training_state")
    training_state = (
        str(raw_training_state).lower().strip() if raw_training_state else None
    )
    state = _map_training_state_to_public_state(training_state, mlflow_status)

    loss_history = []
    if client is not None:
        try:
            loss_history = client.get_metric_history(run_id, "loss")
        except Exception:
            loss_history = []
    loss = [
        {"epoch": int(m.step), "value": m.value}
        for m in sorted(loss_history, key=lambda m: m.step)
    ]

    params = getattr(run.data, "params", {}) or {}
    total_epochs = params.get("epochs")
    total_epochs = int(total_epochs) if total_epochs is not None else None
    training_parameters = {
        key: value
        for key, value in params.items()
        if key not in {"dataset_id", "selected_database_label_ids"}
    }

    metrics = getattr(run.data, "metrics", {}) or {}
    epoch_metric = metrics.get("epoch")
    epoch = int(epoch_metric) if epoch_metric is not None else (loss[-1]["epoch"] if loss else 0)
    validation_metrics = _validation_metrics_from_run(run)

    message = (
        tags.get("status_message")
        or tags.get("message")
        or ai_status.get("message")
    )
    queued_at = _parse_timestamp(
        tags.get("queued_at") or ai_status.get("queued_at")
    )
    start_deadline = _parse_timestamp(
        tags.get("start_deadline") or ai_status.get("start_deadline")
    )
    started_at = (
        tags.get("started_at")
        or ai_status.get("started_at")
    )

    resolved_task_id = task_id if task_id is not None else tags.get("celery_task_id")

    return {
        "task_id": resolved_task_id,
        "run_id": run_id,
        "state": state,
        "training_state": training_state,
        "message": message,
        "queued_at": queued_at,
        "start_deadline": start_deadline,
        "started_at": started_at,
        "mlflow_status": mlflow_status,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "training_parameters": training_parameters,
        "loss": loss,
        "validation_metrics": validation_metrics,
        "validation_metrics_unavailable": tags.get("validation_metrics_unavailable"),
        "label_ids": _parse_label_ids(tags.get("label_ids")),
        "run_name": tags.get("run_name"),  # user-supplied alias; None when not set
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
    }


def _set_run_terminated(
    client, run_id: str, mlflow_status: str, training_state: Optional[str] = None
):
    """Set a verified terminal MLflow state and optional tag, and return the fresh run record."""
    if training_state:
        try:
            client.set_tag(run_id, "training_state", training_state)
        except Exception:
            pass
    client.set_terminated(run_id, status=mlflow_status)
    return client.get_run(run_id)


async def _reconcile_run_with_celery(run):
    """Close an orphaned MLflow run only when Celery proves it is terminal.

    A worker can be terminated while its MLflow ``start_run`` context is open.
    In that case MLflow keeps the run as RUNNING even though the task has
    stopped. We deliberately do not infer this from elapsed time: expired or
    unavailable Celery results leave the run unchanged.
    """
    if run.info.status != "RUNNING":
        return run

    tags = getattr(run.data, "tags", {}) or {}
    task_id = tags.get("celery_task_id")
    if not task_id:
        return run

    try:
        ai_status = await service.get_training_task_status(task_id)
    except Exception:
        logger.warning("Could not read Celery/AI state for training task %s.", task_id)
        return run

    # If the AI call updated shared MLflow tags, refetch the run before creating the snapshot
    try:
        refreshed_run = await asyncio.to_thread(MODEL_REGISTRY.client.get_run, run.info.run_id)
        if refreshed_run is not None:
            run = refreshed_run
    except Exception:
        pass

    if run.info.status != "RUNNING":
        return run

    training_state = (ai_status.get("training_state") or "").lower().strip()
    celery_state = ai_status.get("state")

    target_mlflow_status = None
    target_training_state = None
    if training_state == "completed" or celery_state == "SUCCESS":
        target_mlflow_status = "FINISHED"
        target_training_state = "completed"
    elif training_state == "failed" or celery_state == "FAILURE":
        target_mlflow_status = "FAILED"
        target_training_state = "failed"
    elif training_state in {"cancelled", "timed_out"} or celery_state == "REVOKED":
        target_mlflow_status = "KILLED"
        target_training_state = "timed_out" if training_state == "timed_out" else "cancelled"

    if target_mlflow_status is None:
        return run

    return await asyncio.to_thread(
        _set_run_terminated,
        MODEL_REGISTRY.client,
        run.info.run_id,
        target_mlflow_status,
        target_training_state,
    )



async def _read_training_snapshot(task_id: str) -> dict:
    """Read a progress snapshot for a Celery ``task_id`` straight from MLflow.

    Returns an initial snapshot while no MLflow run exists yet.
    """
    run = await asyncio.to_thread(_find_training_run, task_id)
    if run is None:
        ai_status = None
        try:
            ai_status = await service.get_training_task_status(task_id)
        except Exception:
            logger.warning("Could not read training task status from AI service for task %s.", task_id)
            ai_status = None

        if ai_status and ai_status.get("run_id"):
            try:
                run = await asyncio.to_thread(MODEL_REGISTRY.client.get_run, ai_status["run_id"])
            except Exception:
                run = None

        if run is None:
            return _empty_snapshot(task_id, ai_status)

    run = await _reconcile_run_with_celery(run)
    return await asyncio.to_thread(_snapshot_from_run, MODEL_REGISTRY.client, run, task_id)


async def _read_run_snapshot(run_id: str) -> dict:
    """Read a progress snapshot for a specific MLflow ``run_id`` (a past run)."""
    client = MODEL_REGISTRY.client
    run = await asyncio.to_thread(client.get_run, run_id)
    run = await _reconcile_run_with_celery(run)
    return await asyncio.to_thread(_snapshot_from_run, client, run)


def _find_training_runs(dataset_id: int):
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
    return runs


async def _list_training_runs(dataset_id: int) -> list[dict]:
    """Return snapshots, reconciling only runs with a verified terminal task."""
    runs = await asyncio.to_thread(_find_training_runs, dataset_id)

    async def snapshot(run):
        run = await _reconcile_run_with_celery(run)
        return await asyncio.to_thread(_snapshot_from_run, MODEL_REGISTRY.client, run)

    return await asyncio.gather(*(snapshot(run) for run in runs))


@router.get("/training/runs")
async def list_training_runs(dataset_id: int,
                             user: AuthenticatedUser = Depends(require(Permission.AI_TRAIN))):
    """List past + active training runs for a dataset (for the run-history list)."""
    return {"success": True, "runs": await _list_training_runs(dataset_id)}


@router.get("/training/runs/{run_id}")
async def get_run_snapshot(run_id: str, user: User = Depends(get_current_user)):
    """Return a progress snapshot for a specific (e.g. past) MLflow run."""
    return await _read_run_snapshot(run_id)


@router.get("/training/{task_id}")
async def get_training_status(task_id: str, user: User = Depends(get_current_user)):
    """Return a single MLflow-backed progress snapshot for a training job."""
    return await _read_training_snapshot(task_id)


@router.get("/training/{task_id}/stream")
async def get_training_status_stream(task_id: str, request: Request,
                                     user: User = Depends(get_current_user)):
    """Stream MLflow-backed progress for a training job as Server-Sent Events.

    Polls the MLflow run every couple of seconds and emits one ``data:`` event per
    tick until the run reaches a terminal state (SUCCESS/FAILED/CANCELLED/TIMED_OUT).
    """

    async def event_generator():
        while True:
            if await request.is_disconnected():
                return
            snapshot = await _read_training_snapshot(task_id)
            yield f"data: {json.dumps(snapshot)}\n\n"
            if snapshot["state"] in _TERMINAL_STATES:
                return
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/training/{task_id}")
async def cancel_training_of_model(task_id: str, user: User = Depends(get_current_user)):
    """Cancel a task and close its matching MLflow run as cancelled."""
    ai_status = None
    try:
        ai_status = await service.cancel_training(task_id)
    except Exception as cancel_exc:
        # Check if the task/run is already terminal before failing
        run = await asyncio.to_thread(_find_training_run, task_id)
        if run is not None and run.info.status in {"FINISHED", "FAILED", "KILLED"}:
            return await asyncio.to_thread(
                _snapshot_from_run, MODEL_REGISTRY.client, run, task_id
            )

        logger.exception(
            "Failed to cancel instance segmentation training on AI service for %s.", task_id
        )
        raise HTTPException(
            status_code=http_status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not cancel training: {cancel_exc}",
        )

    run = await asyncio.to_thread(_find_training_run, task_id)
    if run is not None and run.info.status == "RUNNING":
        run = await asyncio.to_thread(
            _set_run_terminated, MODEL_REGISTRY.client, run.info.run_id, "KILLED", "cancelled"
        )

    if run is None:
        cancel_payload = ai_status or {
            "training_state": "cancelled",
            "state": "REVOKED",
            "message": "Training cancelled by user.",
        }
        return _empty_snapshot(task_id, cancel_payload)
    return await asyncio.to_thread(
        _snapshot_from_run, MODEL_REGISTRY.client, run, task_id, ai_status
    )




# The POST /run inference endpoint was removed: nothing called it, and its request
# body carried a raw filesystem path (`image_url`) that was handed straight to
# cv2.imread, which made it both unauthorizable — there is no dataset to resolve
# from a path — and an arbitrary-file-read on the shared volume. Interactive
# inference goes through the annotation-session WebSocket, which resolves the path
# server-side from the image id. If a direct inference API is ever needed, it
# should take an `image_id` so it can be permission-checked.

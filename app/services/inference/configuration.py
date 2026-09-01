"""Business logic for persisting, updating, and resolving dataset model routing policies."""
from __future__ import annotations

from datetime import datetime, timezone
from logging import getLogger
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.database.datasets import Datasets
from app.database.dataset_model_routing_configs import DatasetModelRoutingConfigs
from app.database.labels import Labels
from app.schemas.inference import (
    MODEL_ROUTING_TASKS,
    DatasetModelRoutingRead,
    DatasetModelRoutingWrite,
    InferenceOptions,
    ModelRoutingBinding,
    ModelRoutingSuggestResult,
    ModelRoutingTask,
    ResolvedStep,
    WriteMode,
)
from app.services.inference import planning
from app.services.inference.input_validator import validate_and_normalize_inputs

logger = getLogger(__name__)

SUPPORTED_SUGGEST_TASKS: tuple[str, ...] = ("cross-image-suggestion",)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_read_schema(row: DatasetModelRoutingConfigs) -> DatasetModelRoutingRead:
    """Convert an ORM policy row into a canonical DatasetModelRoutingRead schema."""
    bindings = [
        ModelRoutingBinding.model_validate(b)
        for b in (row.bindings or [])
    ]
    return DatasetModelRoutingRead(
        dataset_id=row.dataset_id,
        bindings=bindings,
        updated_by=row.updated_by,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def get_routing_policy(
    db: Session,
    dataset_id: int,
) -> Optional[DatasetModelRoutingRead]:
    """Retrieve the model routing policy for a dataset.

    Returns stored bindings as-is (including bindings whose models may currently be
    unavailable or incompatible) so client UIs can display repair state.
    Returns None when no routing policy has been configured for the dataset.
    """
    row = (
        db.query(DatasetModelRoutingConfigs)
        .filter(DatasetModelRoutingConfigs.dataset_id == dataset_id)
        .first()
    )
    if row is None:
        return None
    return _to_read_schema(row)


def upsert_routing_policy(
    db: Session,
    dataset_id: int,
    username: Optional[str],
    body: DatasetModelRoutingWrite,
) -> DatasetModelRoutingRead:
    """Save or replace the model routing policy for a dataset atomically.

    Validates:
    1. Request body dataset ID matches path parameter dataset ID.
    2. Dataset exists in the database.
    3. Each binding targets a valid canonical routing task.
    4. Each label-specific override targets a label belonging to the dataset hierarchy.
    5. Each model is registered, ready, and advertises the binding's task.
    6. For class-aware models with declared label IDs, the target label is supported.
    7. Model inputs are validated and normalized according to the model's contract.

    Commits one complete replacement atomically.
    """
    if body.dataset_id != dataset_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset ID in request body ({body.dataset_id}) does not match path parameter ({dataset_id}).",
        )

    dataset_exists = (
        db.query(Datasets.id).filter(Datasets.id == dataset_id).first() is not None
    )
    if not dataset_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found.",
        )

    levels = planning.label_levels(db, dataset_id)
    dataset_labels = {
        label.id: label
        for label in db.query(Labels).filter(Labels.dataset_id == dataset_id).all()
    }
    catalog = {
        (option.registry_key, option.task): option
        for option in planning.model_catalog(db, dataset_id, tasks=MODEL_ROUTING_TASKS).models
    }

    canonical_bindings: list[dict] = []
    for binding in body.bindings:
        task_str = binding.task.value if isinstance(binding.task, ModelRoutingTask) else str(binding.task)
        if task_str not in MODEL_ROUTING_TASKS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported routing task '{task_str}'. Supported tasks: {MODEL_ROUTING_TASKS}.",
            )

        label_name = None
        if binding.label_id is not None:
            label = dataset_labels.get(binding.label_id)
            if label is None or binding.label_id not in levels:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Label {binding.label_id} is not part of dataset {dataset_id}'s hierarchy.",
                )
            label_name = label.name

        option = catalog.get((binding.model_registry_key, task_str))
        if option is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No ready '{task_str}' model named {binding.model_registry_key!r}.",
            )

        if binding.label_id is not None and option.label_ids:
            if binding.label_id not in option.label_ids:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Model {option.name!r} does not predict label {label_name!r}; "
                        f"it predicts label ids {sorted(option.label_ids)}."
                    ),
                )

        normalized_inputs = None
        if binding.inputs is not None:
            if option.input_contract is not None:
                try:
                    normalized_inputs = validate_and_normalize_inputs(
                        option.input_contract, binding.inputs
                    )
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid inference inputs for model {option.name!r} (task '{task_str}'): {exc}",
                    ) from exc
            else:
                normalized_inputs = binding.inputs

        canonical_bindings.append({
            "task": task_str,
            "label_id": binding.label_id,
            "model_registry_key": binding.model_registry_key,
            "inputs": normalized_inputs,
        })

    now = _utcnow()
    row = (
        db.query(DatasetModelRoutingConfigs)
        .filter(DatasetModelRoutingConfigs.dataset_id == dataset_id)
        .first()
    )

    if row is None:
        row = DatasetModelRoutingConfigs(
            dataset_id=dataset_id,
            bindings=canonical_bindings,
            updated_by=username,
            created_at=now,
            updated_at=now,
        )
        db.add(row)
    else:
        row.bindings = canonical_bindings
        row.updated_by = username
        row.updated_at = now

    db.commit()
    db.refresh(row)
    return _to_read_schema(row)


def clear_routing_policy(
    db: Session,
    dataset_id: int,
) -> bool:
    """Explicitly delete the model routing policy for a dataset.

    Returns True if a policy was deleted, or False if none existed.
    """
    row = (
        db.query(DatasetModelRoutingConfigs)
        .filter(DatasetModelRoutingConfigs.dataset_id == dataset_id)
        .first()
    )
    if row is None:
        return False

    db.delete(row)
    db.commit()
    return True


def resolve_routing_binding(
    db: Session,
    dataset_id: int,
    task: str | ModelRoutingTask,
    label_id: Optional[int] = None,
) -> Optional[dict]:
    """Resolve the effective model binding for a given task and optional label.

    Resolution order:
    1. Exact dataset override for (task, label_id) if label_id is provided.
    2. Dataset default binding for (task, None).
    3. None (no dataset override exists; consumers fall back to user favorites / defaults).
    """
    task_str = task.value if isinstance(task, ModelRoutingTask) else str(task)
    row = (
        db.query(DatasetModelRoutingConfigs)
        .filter(DatasetModelRoutingConfigs.dataset_id == dataset_id)
        .first()
    )
    if row is None or not row.bindings:
        return None

    # 1. Exact label-specific override
    if label_id is not None:
        for binding in row.bindings:
            if binding.get("task") == task_str and binding.get("label_id") == label_id:
                return binding

    # 2. Task-level default
    for binding in row.bindings:
        if binding.get("task") == task_str and binding.get("label_id") is None:
            return binding

    # 3. No dataset route
    return None


def _reject_if_batch_active(db: Session, dataset_id: int) -> None:
    """Reject interactive operations for every non-terminal batch lifecycle state."""
    job = planning.active_job(db, dataset_id)
    if job is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Inference job {job.id} is currently {job.status} on this dataset. "
                "Wait for it to finish or cancel it first."
            ),
        )


def execute_suggest_step(
    db: Session,
    dataset_id: int,
    image_id: int,
    label_id: int,
    username: str,
    task: str | ModelRoutingTask = ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
    mask_id: Optional[int] = None,
    # Transitional kwargs for backwards compatibility
    config_id: Optional[int] = None,
    name: Optional[str] = None,
) -> ModelRoutingSuggestResult:
    """Execute one routed model step on a single image with patch semantics.

    Resolves (task, label_id) against the dataset routing policy, validates against
    the live model catalog and dataset hierarchy, and executes on the target image mask.
    """
    from app.database.images import Images
    from app.database.masks import Masks
    from app.services.inference import execution

    task_str = task.value if isinstance(task, ModelRoutingTask) else str(task)
    if task_str not in SUPPORTED_SUGGEST_TASKS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Interactive suggestion is not supported for task '{task_str}'. "
                f"Supported suggestion tasks: {list(SUPPORTED_SUGGEST_TASKS)}."
            ),
        )

    _reject_if_batch_active(db, dataset_id)

    image = db.get(Images, image_id)
    if image is None or image.dataset_id != dataset_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} does not belong to dataset {dataset_id}.",
        )

    if mask_id is not None:
        mask = db.get(Masks, mask_id)
        if mask is None or mask.image_id != image.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Mask {mask_id} does not belong to image {image_id}.",
            )
    else:
        masks = db.query(Masks).filter(Masks.image_id == image.id).all()
        if not masks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image {image_id} has no mask to write contours to.",
            )
        if len(masks) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image {image_id} has {len(masks)} masks; mask_id must be explicitly specified.",
            )
        mask = masks[0]

    binding = resolve_routing_binding(db, dataset_id, task=task_str, label_id=label_id)
    if binding is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No routing configured for task '{task_str}' and label {label_id} in dataset {dataset_id}.",
        )

    model_key = binding["model_registry_key"]
    catalog = {
        (option.registry_key, option.task): option
        for option in planning.model_catalog(db, dataset_id, tasks=MODEL_ROUTING_TASKS).models
    }
    option = catalog.get((model_key, task_str))
    if option is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configured model {model_key!r} for task '{task_str}' is unavailable.",
        )

    levels = planning.label_levels(db, dataset_id)
    label = db.get(Labels, label_id)
    if label is None or label.dataset_id != dataset_id or label_id not in levels:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Label {label_id} is not part of dataset {dataset_id}'s hierarchy.",
        )

    if option.label_ids and label_id not in option.label_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {option.name!r} does not predict label {label.name!r}.",
        )

    raw_inputs = binding.get("inputs") or {"conditioning": {}, "parameters": {}}
    try:
        normalized_inputs = (
            validate_and_normalize_inputs(option.input_contract, raw_inputs)
            if option.input_contract is not None
            else raw_inputs
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid inference inputs for model {option.name!r}: {exc}",
        ) from exc

    resolved_step = ResolvedStep(
        label_id=label_id,
        model_registry_key=model_key,
        task=task_str,
        level=levels[label_id],
        parent_label_id=label.parent_id,
        label_name=label.name,
        model_name=option.name,
        model_label_ids=list(option.label_ids),
        inputs=normalized_inputs,
        input_contract=option.input_contract,
        provenance=option.provenance,
    )

    options = InferenceOptions(
        write_mode=WriteMode.PATCH,
        preserve_reviewed=True,
    )

    try:
        unit_result = execution.run_unit(
            db=db,
            step=resolved_step,
            image=image,
            options=options,
            username=username,
            mask=mask,
        )
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except execution.InferenceUnitError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        db.rollback()
        logger.exception(
            "Unexpected error executing inference suggestion for image %s, label %s, task %s: %s",
            image_id,
            label_id,
            task_str,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute inference suggestion.",
        ) from exc

    return ModelRoutingSuggestResult(
        dataset_id=dataset_id,
        image_id=image_id,
        label_id=label_id,
        task=task_str,
        contours_created=unit_result.created,
        contours_suppressed=unit_result.suppressed,
        contours_unparented=unit_result.unparented,
        contour_ids=unit_result.contour_ids,
    )


# Transitional aliases for backwards compatibility with existing route imports
get_config = get_routing_policy
upsert_config = upsert_routing_policy
clear_config = lambda db, dataset_id, **kwargs: clear_routing_policy(db, dataset_id)

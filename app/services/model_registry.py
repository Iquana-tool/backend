"""Direct MLflow access for model discovery.

The backend reads available models straight from the shared MLflow registry
instead of HTTP-hopping through each AI service's ``/models`` endpoint. MLflow
is the source of truth, so this removes one network round-trip per request.
"""
import json
from logging import getLogger

import mlflow
from iquana_toolbox.mlflow import MLFlowModelRegistry

from app.database import get_context_session
from app.database.datasets import Datasets
from app.database.labels import Labels
from config import MLFLOW_URL

logger = getLogger(__name__)

# Client only; no connection is made until a query runs.
MODEL_REGISTRY = MLFlowModelRegistry(MLFLOW_URL)


def _model_tag_value(model_info: dict, key: str):
    tags = model_info.get("tags")
    if isinstance(tags, dict):
        return tags.get(key)
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict) and tag.get("key") == key:
                return tag.get("value")
    return None


def _parse_id_list(value) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = value.split(",")
    if not isinstance(value, (list, tuple)):
        return []

    ids = []
    for item in value:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return ids


def _set_model_tag(model_info: dict, key: str, value: str):
    tags = model_info.get("tags")
    if isinstance(tags, dict):
        tags[key] = value
        return
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict) and tag.get("key") == key:
                tag["value"] = value
                return
        tags.append({"key": key, "value": value})
        return
    model_info["tags"] = {key: value}


def _enrich_model_provenance(model_info: dict) -> dict:
    """Add names for legacy trained models that only stored database IDs."""
    dataset_id_value = (
        _model_tag_value(model_info, "trained_on_dataset_id")
        or _model_tag_value(model_info, "dataset_id")
        or model_info.get("dataset_id")
    )
    try:
        dataset_id = int(dataset_id_value)
    except (TypeError, ValueError):
        return model_info

    label_ids = _parse_id_list(model_info.get("label_ids"))
    if not label_ids:
        label_ids = _parse_id_list(_model_tag_value(model_info, "selected_label_ids"))

    needs_dataset_name = not _model_tag_value(model_info, "trained_on_dataset_name")
    needs_label_names = bool(label_ids) and not _model_tag_value(model_info, "trained_label_names")
    if not needs_dataset_name and not needs_label_names:
        return model_info

    try:
        with get_context_session() as db:
            dataset = db.query(Datasets).filter(Datasets.id == dataset_id).first()
            if dataset and needs_dataset_name:
                _set_model_tag(model_info, "trained_on_dataset_id", str(dataset.id))
                _set_model_tag(model_info, "trained_on_dataset_name", dataset.name)

            if dataset and needs_label_names:
                labels = (
                    db.query(Labels)
                    .filter(Labels.dataset_id == dataset_id, Labels.id.in_(label_ids))
                    .all()
                )
                names_by_id = {label.id: label.name for label in labels}
                names = [names_by_id.get(label_id, "") for label_id in label_ids]
                _set_model_tag(
                    model_info,
                    "trained_label_names",
                    json.dumps(names, ensure_ascii=False),
                )
    except Exception:
        logger.exception("Failed to resolve provenance for trained model metadata.")

    return model_info


def _search_registered_models_by_tags(tags: dict):
    """Return registered models matching ``tags`` (hits carry a ``.name``).

    We search MLflow directly instead of going through the toolbox's
    ``get_model_infos_via_tags``. That helper eagerly rebuilds a ``ModelInfo``
    from each registered model's tags, which fails when the tag set lacks the
    (now required) ``ModelInfo`` fields like ``registry_key`` / ``name`` /
    ``description`` / ``usage_tip`` -- tags only carry the filterable subset
    (task/status/...). We only need the names here; the full info is read from
    artifact metadata by ``_full_model_info``, which is the source of truth.
    """
    filter_string = " AND ".join(f"tags.{key} = '{value}'" for key, value in tags.items())
    return MODEL_REGISTRY.client.search_registered_models(filter_string=filter_string)


def _registry_key(model) -> str:
    """Pull the registry key from a tag-search hit (dict or ModelInfo)."""
    if isinstance(model, dict):
        return model["name"]
    return getattr(model, "registry_key", None) or model.name


def _models_for_task(task: str):
    """Registered, ready-to-serve models advertising ``task``.

    The unified ai-service stamps a filter-safe per-task boolean tag
    (``task_<name>`` == "true") for every task a model serves, so a multi-task
    model (e.g. SAM 3, which does both instance suggestion and prompted
    segmentation) is found under each of its tasks -- not only its primary
    ``task`` tag. We union that with a search on the legacy single ``task`` tag
    so models registered before the merge still appear during the transition.
    """
    task_tag = "task_" + task.replace("-", "_")
    by_name: dict[str, object] = {}
    for tags in ({task_tag: "true", "status": "ready"},
                 {"task": task, "status": "ready"}):
        for model in _search_registered_models_by_tags(tags):
            by_name[model.name] = model
    return list(by_name.values())


def _full_model_info(registry_key: str) -> dict:
    """Return a model's complete ``model_info`` from its artifact metadata.

    ``register_model`` stores the full ``ModelInfo.model_dump()`` as the logged
    model's ``metadata`` (recorded in the MLmodel file, not the weights), so
    reading it back is lossless and cheap -- ~50ms regardless of model size, and
    independent of toolbox version. This is preferable to rebuilding the info
    from the registered-model tags, which only carry the filterable subset
    (task/status/...) and drop free-text fields like description and usage_tip.

    Falls back to a minimal stub if an older artifact carries no metadata.
    """
    try:
        info = mlflow.models.get_model_info(f"models:/{registry_key}/latest")
        if info.metadata:
            metadata = dict(info.metadata)
            if isinstance(metadata.get("tags"), dict):
                metadata["tags"] = dict(metadata["tags"])
            elif isinstance(metadata.get("tags"), list):
                metadata["tags"] = [
                    dict(tag) if isinstance(tag, dict) else tag
                    for tag in metadata["tags"]
                ]
            return _enrich_model_provenance(metadata)
        logger.warning("Model '%s' has no artifact metadata; returning stub.", registry_key)
    except Exception:
        logger.exception("Failed to read artifact metadata for model '%s'.", registry_key)
    return {"registry_key": registry_key, "name": registry_key}


def list_available_models(task: str) -> dict:
    """Return ready-to-serve models for ``task`` directly from MLflow.

    Each result is the model's full ``model_info`` (description, usage_tip,
    badges, status, trainable, and task-specific fields), read from the artifact
    metadata. Tags are used only to filter the candidate set.

    Args:
        task: The model ``task`` tag, e.g. ``"prompted-segmentation"``,
            ``"instance-suggestion"`` or ``"instance-segmentation"``.
    """
    mlflow.set_tracking_uri(MLFLOW_URL)
    matched = _models_for_task(task)
    models = [_full_model_info(_registry_key(m)) for m in matched]
    return {
        "success": True,
        "message": f"Retrieved {len(models)} available models.",
        "result": models,
    }

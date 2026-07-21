"""Direct MLflow access for model discovery.

The backend reads available models straight from the shared MLflow registry
instead of HTTP-hopping through each AI service's ``/models`` endpoint. MLflow
is the source of truth, so this removes one network round-trip per request.
"""
from logging import getLogger

import mlflow
from iquana_toolbox.mlflow import MLFlowModelRegistry

from config import MLFLOW_URL

logger = getLogger(__name__)

# Client only; no connection is made until a query runs.
MODEL_REGISTRY = MLFlowModelRegistry(MLFLOW_URL)


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
        info = mlflow.models.get_model_info(f"models:/{registry_key}@latest")
        if info.metadata:
            return info.metadata
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
    matched = _search_registered_models_by_tags({"task": task, "status": "ready"})
    models = [_full_model_info(_registry_key(m)) for m in matched]
    return {
        "success": True,
        "message": f"Retrieved {len(models)} available models.",
        "result": models,
    }

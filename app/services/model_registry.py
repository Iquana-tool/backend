"""Direct MLflow access for model discovery.

The backend reads available models straight from the shared MLflow registry
instead of HTTP-hopping through each AI service's ``/models`` endpoint. MLflow
is the source of truth, so this removes one network round-trip per request.
"""
from logging import getLogger

from iquana_toolbox.mlflow import MLFlowModelRegistry

from config import MLFLOW_URL

logger = getLogger(__name__)

# Client only; no connection is made until a query runs.
MODEL_REGISTRY = MLFlowModelRegistry(MLFLOW_URL)


def _model_infos_via_tags(tags: dict):
    """Look up models by tags across toolbox versions.

    The toolbox renamed ``get_models_via_tags`` -> ``get_model_infos_via_tags``.
    The backend and the AI services may not pin the same toolbox revision, so
    resolve whichever exists rather than committing to one name.
    """
    getter = getattr(MODEL_REGISTRY, "get_model_infos_via_tags", None) or getattr(
        MODEL_REGISTRY, "get_models_via_tags"
    )
    return getter(tags=tags)


def list_available_models(task: str) -> dict:
    """Return ready-to-serve models for ``task`` directly from MLflow.

    Args:
        task: The model ``task`` tag, e.g. ``"prompted-segmentation"``,
            ``"instance-discovery"`` or ``"instance-segmentation"``.
    """
    models = _model_infos_via_tags({"task": task, "status": "ready"})
    return {
        "success": True,
        "message": f"Retrieved {len(models)} available models.",
        "result": models,
    }

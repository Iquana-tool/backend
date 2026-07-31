from logging import getLogger

import httpx
from iquana_toolbox.schemas.networking.http.services import InstanceSegmentationRequest
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest

from app.services.ai_services.base_service import BaseService
from config import INSTANCE_SEGMENTATION_BACKEND_URL as BASE_URL

logger = getLogger(__name__)


class InstanceSegmentationService(BaseService):
    def __init__(self):
        super().__init__(BASE_URL)

    async def inference(self, request: InstanceSegmentationRequest):
        """Segment an image using 2D prompts.
        Args:
            request (InstanceSegmentationRequest): Request object.
        Returns:
            dict: A response dict
        """

        # Send the request to the backend
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/annotation_session/run"
            response = await client.post(url, json=request.model_dump())

            response.raise_for_status()

            return response.json()

    async def start_training(self, request: InstanceSegmentationTrainingRequest) -> dict:
        """Dispatch a training job to the instance-segmentation service.

        The service hands the job to a Celery worker and returns ``{"task_id": ...}``.
        The task id doubles as the MLflow run id, which the gateway later polls for
        progress.
        """
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/train"
            response = await client.post(url, json=request.model_dump())
            response.raise_for_status()
            return response.json()

    async def cancel_training(self, task_id: str) -> dict:
        """Revoke a running training job by its task id."""
        async with httpx.AsyncClient(timeout=30) as client:
            url = f"{self.backend_url}/train/{task_id}"
            response = await client.delete(url)
            response.raise_for_status()
            return response.json()

    async def get_training_task_state(self, task_id: str) -> str:
        """Read the authoritative Celery state for a training task."""
        async with httpx.AsyncClient(timeout=10) as client:
            url = f"{self.backend_url}/train/{task_id}"
            response = await client.get(url)
            response.raise_for_status()
            return response.json().get("state", "PENDING")

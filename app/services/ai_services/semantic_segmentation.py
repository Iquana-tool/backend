from logging import getLogger

import httpx
from iquana_toolbox.schemas.networking.http.services import SemanticSegmentationRequest
from iquana_toolbox.schemas.training import SemanticTrainingRequest

from app.services.ai_services.base_service import BaseService
from config import SEMANTIC_SEGMENTATION_BACKEND_URL as BASE_URL

logger = getLogger(__name__)


class SemanticSegmentationService(BaseService):
    def __init__(self):
        super().__init__(BASE_URL)

    async def get_models(self):
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/models/all"
            response = await client.get(url)

            response.raise_for_status()

            return response.json()

    async def get_model(self, model_registry_key: str):
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/model/{model_registry_key}"
            response = await client.get(url)

            response.raise_for_status()

            return response.json()

    async def delete_model(self, model_registry_key: str):
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/models/{model_registry_key}"
            response = await client.delete(url)

            response.raise_for_status()

            return response.json()

    async def inference(self, request: SemanticSegmentationRequest):
        """Segment an image using 2D prompts.
        Args:
            request (PromptedSegmentationWebsocketRequest): Request object.
        Returns:
            dict: A response dict
        """

        # Send the request to the backend
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/annotation_session/run"
            response = await client.post(url, json=request.model_dump())

            response.raise_for_status()

            return response.json()

    async def start_training(self, request: SemanticTrainingRequest):
        # Send the request to the backend
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/training/start"
            response = await client.post(url, json=request.model_dump())

            response.raise_for_status()

            return response.json()

from logging import getLogger

import httpx
from fastapi.encoders import jsonable_encoder
from schemas.completion_segmentation.inference import CompletionServiceRequest

from app.services.ai_services.base_service import BaseService
from paths import COMPLETION_SEGMENTATION_BACKEND_URL as BASE_URL

logger = getLogger(__name__)


class CompletionService(BaseService):
    def __init__(self):
        super().__init__(BASE_URL)

    async def inference(self, request: CompletionServiceRequest):
        """Segment an image using 2D prompts.
        Args:
            request (PromptedSegmentationWebsocketRequest): Request object.
        Returns:
            dict: A response dict
        """

        # Send the request to the backend
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/annotation_session/infer_instances"
            response = await client.post(url, json=jsonable_encoder(request.model_dump(exclude_none=True)))

            response.raise_for_status()

            return response.json()

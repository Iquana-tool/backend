from logging import getLogger

import httpx
from schemas.service_requests import PromptedSegmentationRequest

from app.schemas.contours import Contour
from app.services.ai_services.base_service import BaseService
from app.services.util import extract_mask_from_response
from paths import PROMPTED_SEGMENTATION_BACKEND_URL as BASE_URL

logger = getLogger(__name__)


class PromptedSegmentationService(BaseService):
    def __init__(self):
        super().__init__(BASE_URL)

    async def inference(self, request: PromptedSegmentationRequest):
        """Segment an image using 2D prompts.
        Args:
            request (PromptedSegmentationWebsocketRequest): Request object.
        Returns:
            dict: A response dict
        """
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/annotation_session/prompted_segmentation"
            # Only send the prompts in the body
            response = await client.post(url, json=request.model_dump(exclude_none=True))

            response.raise_for_status()
            return response.json()

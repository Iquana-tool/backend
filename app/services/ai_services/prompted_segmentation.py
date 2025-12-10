import json
from logging import getLogger

import httpx

from app.database import get_context_session
from app.database.contours import Contours
from app.database.images import Images
from app.schemas.prompted_segmentation.segmentations import PromptedSegmentationWebsocketRequest
from app.services.ai_services.base_service import BaseService
from app.services.util import extract_mask_from_response
from paths import PROMPTED_SEGMENTATION_BACKEND_URL as BASE_URL

logger = getLogger(__name__)


class PromptedSegmentationService(BaseService):
    def __init__(self):
        super().__init__(BASE_URL)

    async def inference(self, request: PromptedSegmentationWebsocketRequest):
        """Segment an image using 2D prompts.
        Args:
            request (PromptedSegmentationWebsocketRequest): Request object.
        Returns:
            dict: A response dict
        """

        # Send the request to the backend
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/annotation_session/segment_image_with_prompts"
            # Only send the prompts in the body
            response = await client.post(url, json=request.model_dump(exclude_none=True))

            response.raise_for_status()
            mask, shape, dtype, score = extract_mask_from_response(response)

            return {
                "success": True,
                "message": f"Successfully computed mask from prompts with confidence of {score:.1%}",
                "mask": mask,
                "score": score,
                "shape": shape,
                "dtype": dtype,
            }

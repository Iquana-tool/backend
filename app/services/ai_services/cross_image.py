"""HTTP client for the ai-service ``cross-image-suggestion`` surface.

Called from the orchestration route (an async FastAPI handler), so -- like the other
per-task clients -- it is async. Import-safe before the toolbox pin is bumped: the
``CrossImageSuggestionRequest`` type appears only in an annotation (a string under
``from __future__ import annotations``), never imported at module load.
"""
from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import httpx

from app.services.ai_services.base_service import BaseService
from config import CROSS_IMAGE_BACKEND_URL as BASE_URL

if TYPE_CHECKING:
    from iquana_toolbox.schemas.networking.http.services import CrossImageSuggestionRequest

logger = getLogger(__name__)


class CrossImageService(BaseService):
    def __init__(self):
        super().__init__(BASE_URL)

    async def inference(self, request: "CrossImageSuggestionRequest") -> dict:
        """POST a cross-image suggestion request; return the ai-service response envelope."""
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/annotation_session/run"
            response = await client.post(url, json=request.model_dump())
            response.raise_for_status()
            return response.json()

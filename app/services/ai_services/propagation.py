"""HTTP client for the ai-service ``propagation`` surface (#121).

Import-safe before the toolbox pin includes ``PropagationRequest``: the type only
appears in an annotation, never imported at module load.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, AsyncIterator

import httpx

from app.services.ai_services.base_service import BaseService
from config import PROPAGATION_BACKEND_URL as BASE_URL

if TYPE_CHECKING:
    from iquana_toolbox.schemas.networking.http.services import PropagationRequest

#: A long stack with a large model takes minutes; the request waits for all of it.
TIMEOUT_SECONDS = 900


class PropagationService(BaseService):
    def __init__(self):
        super().__init__(BASE_URL)

    async def inference(self, request: "PropagationRequest") -> dict:
        """POST a propagation request; return the ai-service response envelope."""
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(f"{self.backend_url}/inference", json=request.model_dump())
            response.raise_for_status()
            return response.json()

    async def stream(self, request: "PropagationRequest") -> AsyncIterator[dict]:
        """POST to ``/stream`` and yield each frame's result as the service sends it.

        Leaving the loop early closes the connection, which stops the service after
        the frame it is working on.
        """
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            async with client.stream("POST", f"{self.backend_url}/stream",
                                     json=request.model_dump()) as response:
                if response.status_code >= 400:
                    await response.aread()
                    response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        yield json.loads(line)

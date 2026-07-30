"""HTTP client for the ai-service ``embed`` surface.

Unlike the other ai_services clients (async, called from FastAPI request handlers), embedding
is background work -- run from a Celery task or the backfill script, both synchronous -- so this
is a plain synchronous ``httpx`` client. It POSTs an :class:`EmbedRequest` to the embed surface
and returns the parsed :class:`EmbeddingVector` list from the response envelope.
"""
from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import httpx

from config import EMBED_BACKEND_URL

if TYPE_CHECKING:  # imports only for type checkers; not evaluated at runtime
    from iquana_toolbox.schemas.networking.http.services import EmbedRequest, EmbeddingVector

logger = getLogger(__name__)


class EmbeddingService:
    def __init__(self, backend_url: str = EMBED_BACKEND_URL, timeout: float = 120.0):
        self.backend_url = backend_url
        self.timeout = timeout

    def request_embeddings(self, request: "EmbedRequest") -> "list[EmbeddingVector]":
        """POST an embed request and return the computed vectors (empty on an empty result).

        ``EmbeddingVector`` is imported lazily: the embed schemas live in the toolbox and are
        only present once its pin is bumped, but this module sits on the app import path, so a
        module-level import would crash startup before the bump. Nothing calls this method until
        the embedding feature is actually used, by which point the toolbox must be current.
        """
        from iquana_toolbox.schemas.networking.http.services import EmbeddingVector

        url = f"{self.backend_url}/inference"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=request.model_dump())
            response.raise_for_status()
            payload = response.json()
        return [EmbeddingVector(**item) for item in payload.get("result", [])]

"""Wire format for telemetry ingest and for the config endpoint."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.services.telemetry.config import TelemetryComponent


class TelemetryEvent(BaseModel):
    """One captured user action, as sent by the frontend client.

    `username` is not accepted from the client: the server stamps it from the
    bearer token, so a batch cannot attribute events to someone else.
    """

    event_id: str = Field(max_length=64)
    ts: datetime
    component: TelemetryComponent
    event_type: str = Field(max_length=128)
    session_id: str | None = Field(default=None, max_length=64)
    dataset_id: int | None = None
    image_id: int | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    payload: dict[str, Any] | None = None
    client: str | None = Field(default=None, max_length=255)

    @field_validator("event_type")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("event_type must not be empty")
        return value


class TelemetryBatch(BaseModel):
    """A flush from one client. Size is capped server-side by USER_EVENTS_MAX_BATCH."""

    events: list[TelemetryEvent] = Field(default_factory=list)


class TelemetryIngestResponse(BaseModel):
    #: Events that passed validation and component gating and were queued.
    accepted: int
    #: Events dropped because their component is switched off, or capture is.
    dropped: int


class TelemetryConfigUpdate(BaseModel):
    """Admin-supplied runtime override. Omitted fields are left unchanged."""

    capture_enabled: bool | None = None
    #: Full replacement list of enabled components, e.g. ["annotation", "ai"].
    components: list[TelemetryComponent] | None = None

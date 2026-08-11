"""Persisted runtime overrides for telemetry capture.

Exactly one row (``id == 1``) ever exists: the environment supplies the boot
defaults, and this row remembers whatever an admin last chose over
``PUT /telemetry/config`` so the choice survives a restart. The row can only
narrow or widen capture *within* a deployment that has ``USER_EVENTS_ENABLED``
set -- it can never switch on a deployment whose environment says off.
"""
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Integer, String

from app.database import database

SETTINGS_ROW_ID = 1


class TelemetrySettings(database):
    __tablename__ = "telemetry_settings"

    id = Column(Integer, primary_key=True, default=SETTINGS_ROW_ID)
    capture_enabled = Column(Boolean, nullable=False, default=False)
    #: Comma-separated component names, mirroring the USER_EVENTS_COMPONENTS env var.
    components = Column(String(255), nullable=False, default="")
    updated_at = Column(DateTime, nullable=False,
                        default=lambda: datetime.now(timezone.utc))
    #: Who last changed it, for the audit trail a study needs.
    updated_by = Column(String, nullable=True)

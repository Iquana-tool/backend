"""User telemetry and study logging.

Layout:
  * `config`   -- environment lock plus persisted runtime overrides.
  * `schemas`  -- ingest/config wire format.
  * `recorder` -- bounded queue and background writer (`recorder.record(...)`).
  * `emit`     -- convenience helpers for the backend's own emit sites.
  * `middleware` -- request/response capture for the `api` component.
  * `export`   -- JSONL/CSV streaming of stored events.

Emit sites should call `app.services.telemetry.emit` rather than the recorder
directly; the helpers there already know which component an event belongs to.

The `recorder` singleton is deliberately *not* re-exported here: binding that name
on the package would shadow the `app.services.telemetry.recorder` submodule, so
`import app.services.telemetry.recorder` would hand back the instance instead of
the module. Import it from its own module (`from ...recorder import recorder`).
"""
from app.services.telemetry.config import (
    TelemetryComponent,
    TelemetryConfig,
    get_config,
    invalidate_cache,
)

__all__ = [
    "TelemetryComponent",
    "TelemetryConfig",
    "get_config",
    "invalidate_cache",
]

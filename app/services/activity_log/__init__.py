"""Activity log: user event capture for studies.

Layout:
  * `config`   -- environment lock plus persisted runtime overrides.
  * `schemas`  -- ingest/config wire format.
  * `recorder` -- bounded queue and background writer (`recorder.record(...)`).
  * `emit`     -- convenience helpers for the backend's own emit sites.
  * `middleware` -- request/response capture for the `api` component.
  * `export`   -- JSONL/CSV streaming of stored events.

Emit sites should call `app.services.activity_log.emit` rather than the recorder
directly; the helpers there already know which component an event belongs to.

The `recorder` singleton is deliberately *not* re-exported here: binding that name
on the package would shadow the `app.services.activity_log.recorder` submodule, so
`import app.services.activity_log.recorder` would hand back the instance instead of
the module. Import it from its own module (`from ...recorder import recorder`).
"""
from app.services.activity_log.config import (
    ActivityComponent,
    ActivityLogConfig,
    get_config,
    invalidate_cache,
)

__all__ = [
    "ActivityComponent",
    "ActivityLogConfig",
    "get_config",
    "invalidate_cache",
]

"""Effective activity log configuration: environment lock plus runtime overrides.

Two layers, and the order between them is the whole point:

1. ``USER_EVENTS_ENABLED`` (environment, read once at boot) is a **lock**. When it
   is false, capture is off and nothing -- no admin, no stored row -- can turn it
   on. A deployment that must not collect data is therefore configured once, at
   deploy time, and cannot be changed by whoever holds an admin account.
2. Everything else (study logging, the per-component switches) starts from the
   environment and may then be overridden at runtime by an admin. The override
   lives in `activity_log_settings` so it survives a restart.

`get_config()` returns the resolved view and is cheap enough to call per event:
the stored row is cached in process and invalidated on write.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from logging import getLogger
from threading import Lock

from sqlalchemy.orm import Session

import config as app_config
from app.database.activity_log_settings import SETTINGS_ROW_ID, ActivityLogSettings

logger = getLogger(__name__)


class ActivityComponent(StrEnum):
    """The independently switchable capture areas.

    Adding one here and gating the emit site on it is all a new capture area needs;
    the config endpoint, the client and the admin toggle pick it up automatically.
    """

    #: Tool/mode switches, prompt placement, contour create/edit/delete, labelling,
    #: undo/redo, zoom/pan, image open/close.
    ANNOTATION = "annotation"
    #: AI assistance: invocation, latency, result counts, accept/reject of suggestions.
    AI = "ai"
    #: Login/logout, route changes and dwell time, tab visibility, idle/active.
    NAVIGATION = "navigation"
    #: HTTP request and WebSocket message timings, statuses and errors.
    API = "api"


ALL_COMPONENTS: frozenset[ActivityComponent] = frozenset(ActivityComponent)


def parse_components(raw: str | None) -> frozenset[ActivityComponent]:
    """Parse a comma-separated component list, ignoring unknown names.

    Unknown names are dropped with a warning rather than raising: a stored row
    written by a newer build must not stop an older one from booting.
    """
    if not raw:
        return frozenset()
    selected: set[ActivityComponent] = set()
    for name in raw.split(","):
        name = name.strip().lower()
        if not name:
            continue
        try:
            selected.add(ActivityComponent(name))
        except ValueError:
            logger.warning("Ignoring unknown activity log component %r.", name)
    return frozenset(selected)


def format_components(components: frozenset[ActivityComponent]) -> str:
    """Stable, sorted serialisation for storage and for the config response."""
    return ",".join(sorted(component.value for component in components))


@dataclass(frozen=True)
class ActivityLogConfig:
    """The resolved answer to "what should be captured right now?"."""

    #: The environment lock. False means every other field is inert.
    enabled: bool
    capture_enabled: bool
    components: frozenset[ActivityComponent] = field(default_factory=frozenset)
    flush_interval_ms: int = 5000
    batch_size: int = 50
    max_batch: int = 200
    max_payload_bytes: int = 4096
    #: No input on an open image for this long counts as idle in `image.visit`.
    idle_threshold_ms: int = 60000
    #: True when the stored row (rather than the environment) supplied the values.
    from_runtime_override: bool = False

    def captures(self, component: ActivityComponent) -> bool:
        """Whether an event in `component` should be recorded at all."""
        return self.enabled and self.capture_enabled and component in self.components

    def as_response(self) -> dict:
        """Shape served to the frontend client and to the admin endpoint."""
        return {
            "enabled": self.enabled,
            "capture_enabled": self.capture_enabled,
            "components": {
                component.value: self.captures(component)
                for component in ActivityComponent
            },
            "flush_interval_ms": self.flush_interval_ms,
            "batch_size": self.batch_size,
            "max_batch": self.max_batch,
            "idle_threshold_ms": self.idle_threshold_ms,
            "from_runtime_override": self.from_runtime_override,
        }


def env_defaults() -> ActivityLogConfig:
    """Boot configuration, straight from the environment."""
    return ActivityLogConfig(
        enabled=app_config.USER_EVENTS_ENABLED,
        capture_enabled=app_config.USER_EVENTS_CAPTURE,
        components=parse_components(app_config.USER_EVENTS_COMPONENTS),
        flush_interval_ms=app_config.USER_EVENTS_FLUSH_INTERVAL_MS,
        batch_size=app_config.USER_EVENTS_BATCH_SIZE,
        max_batch=app_config.USER_EVENTS_MAX_BATCH,
        max_payload_bytes=app_config.USER_EVENTS_MAX_PAYLOAD_BYTES,
        idle_threshold_ms=app_config.USER_EVENTS_IDLE_THRESHOLD_MS,
    )


# -- Cached runtime override ----------------------------------------------

_cache_lock = Lock()
_cached: ActivityLogConfig | None = None


def _resolve(db: Session | None) -> ActivityLogConfig:
    defaults = env_defaults()
    # A locked-off deployment never reads the table: the answer cannot change.
    if not defaults.enabled or db is None:
        return defaults

    row = db.query(ActivityLogSettings).filter_by(id=SETTINGS_ROW_ID).first()
    if row is None:
        return defaults
    return ActivityLogConfig(
        enabled=True,
        capture_enabled=bool(row.capture_enabled),
        components=parse_components(row.components),
        flush_interval_ms=defaults.flush_interval_ms,
        batch_size=defaults.batch_size,
        max_batch=defaults.max_batch,
        max_payload_bytes=defaults.max_payload_bytes,
        idle_threshold_ms=row.idle_threshold_ms or defaults.idle_threshold_ms,
        from_runtime_override=True,
    )


def get_config(db: Session | None = None) -> ActivityLogConfig:
    """Return the effective config, reading the stored override at most once.

    Callers on the hot path (middleware, emit helpers) pass no session and get the
    cached value. Endpoints that already hold a session pass it so a first call
    after boot can populate the cache.
    """
    global _cached
    with _cache_lock:
        if _cached is not None:
            return _cached
    resolved = _resolve(db)
    with _cache_lock:
        # Only cache once a session has been available to consult the override;
        # otherwise the very first (session-less) call would pin the env defaults.
        if resolved.from_runtime_override or not resolved.enabled or db is not None:
            _cached = resolved
    return resolved


def invalidate_cache() -> None:
    """Drop the cached config so the next `get_config` re-reads the stored row."""
    global _cached
    with _cache_lock:
        _cached = None


def save_config(db: Session,
                *,
                capture_enabled: bool | None,
                components: frozenset[ActivityComponent] | None,
                idle_threshold_ms: int | None = None,
                updated_by: str | None) -> ActivityLogConfig:
    """Persist a runtime override and return the new effective config.

    Only meaningful when the environment lock is open; the caller is expected to
    have rejected the request otherwise.
    """
    current = get_config(db)
    row = db.query(ActivityLogSettings).filter_by(id=SETTINGS_ROW_ID).first()
    if row is None:
        row = ActivityLogSettings(id=SETTINGS_ROW_ID)
        db.add(row)

    row.capture_enabled = (
        current.capture_enabled if capture_enabled is None else capture_enabled)
    row.components = format_components(
        current.components if components is None else components)
    if idle_threshold_ms is not None:
        row.idle_threshold_ms = idle_threshold_ms
    row.updated_at = datetime.now(timezone.utc)
    row.updated_by = updated_by
    db.commit()

    invalidate_cache()
    return get_config(db)

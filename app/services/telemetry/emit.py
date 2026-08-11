"""Convenience emit helpers for the backend's own capture sites.

Each helper already knows its component, so a call site stays one line and can be
made unconditionally -- gating happens inside the recorder. `track_duration` is
the context manager for anything worth timing.
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any

from app.services.telemetry.config import TelemetryComponent
from app.services.telemetry.recorder import recorder


def emit_api(event_type: str, **kwargs: Any) -> None:
    """HTTP/WebSocket transport events: path, status, duration, errors."""
    recorder.record(TelemetryComponent.API, event_type, **kwargs)


def emit_ai(event_type: str, **kwargs: Any) -> None:
    """AI assistance events: invocation, latency, result counts, failures."""
    recorder.record(TelemetryComponent.AI, event_type, **kwargs)


def emit_annotation(event_type: str, **kwargs: Any) -> None:
    """Annotation-work events originating server-side (persisted contours, masks)."""
    recorder.record(TelemetryComponent.ANNOTATION, event_type, **kwargs)


def emit_navigation(event_type: str, **kwargs: Any) -> None:
    """Session lifecycle events the server is the authority on (login, logout)."""
    recorder.record(TelemetryComponent.NAVIGATION, event_type, **kwargs)


_EMITTERS = {
    TelemetryComponent.API: emit_api,
    TelemetryComponent.AI: emit_ai,
    TelemetryComponent.ANNOTATION: emit_annotation,
    TelemetryComponent.NAVIGATION: emit_navigation,
}


@contextmanager
def track_duration(component: TelemetryComponent, event_type: str, **kwargs: Any):
    """Time a block and emit one event with its duration and outcome.

    The event is emitted whether the block succeeded or raised, so a failed
    inference is as visible in the study data as a successful one::

        with track_duration(TelemetryComponent.AI, "ai.prompted.invoke",
                            username=user, image_id=image_id) as span:
            result = await service.infer(...)
            span["object_count"] = len(result)
    """
    payload: dict[str, Any] = dict(kwargs.pop("payload", None) or {})
    started = perf_counter()
    error: BaseException | None = None
    try:
        yield payload
    except BaseException as exc:  # re-raised below; recorded either way
        error = exc
        raise
    finally:
        payload["ok"] = error is None
        if error is not None:
            payload["error"] = type(error).__name__
        _EMITTERS[component](
            event_type,
            duration_ms=int((perf_counter() - started) * 1000),
            payload=payload,
            **kwargs,
        )

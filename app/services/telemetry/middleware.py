"""Request-level capture for the `api` component.

Installed only when ``USER_EVENTS_ENABLED`` is set, so a locked-off deployment pays
nothing at all -- not even a per-request branch.

Records the matched *route template* rather than the concrete path, so
``/images/4021`` and ``/images/4022`` aggregate into one row group instead of
thousands of unique paths. The concrete ids that matter for a study (dataset,
image) are lifted out of the path parameters into their own columns.
"""
from __future__ import annotations

from logging import getLogger
from time import perf_counter

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.services.auth import _username_from_token
from app.services.telemetry.config import TelemetryComponent, get_config
from app.services.telemetry.recorder import recorder

logger = getLogger(__name__)

#: Never instrumented: the telemetry endpoints themselves (a flush would record
#: its own arrival, forever), and the noisy liveness paths.
_SKIP_PREFIXES = ("/telemetry", "/docs", "/redoc", "/openapi.json", "/status")


class TelemetryMiddleware(BaseHTTPMiddleware):
    """Emit one `api.request` event per HTTP request."""

    async def dispatch(self, request: Request, call_next):
        # CORS preflights are issued by the browser, not by the participant. They
        # carry neither the bearer token nor `X-Telemetry-Session` -- a preflight
        # is precisely the request that *asks* whether those headers may be sent --
        # so they can never be attributed, and recording them both doubled the
        # `api` row count and produced a phantom session with no id.
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if any(path.startswith(prefix) for prefix in _SKIP_PREFIXES):
            return await call_next(request)
        # Cheap early out; the recorder would gate this anyway, but not before
        # the timing work below.
        if not get_config().captures(TelemetryComponent.API):
            return await call_next(request)

        started = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            self._record(request, status_code, perf_counter() - started)

    def _record(self, request: Request, status_code: int, elapsed: float) -> None:
        try:
            params = request.scope.get("path_params") or {}
            recorder.record(
                TelemetryComponent.API,
                "api.request",
                username=_username_from_request(request),
                session_id=request.headers.get("x-telemetry-session"),
                dataset_id=_as_int(params.get("dataset_id")),
                image_id=_as_int(params.get("image_id")),
                duration_ms=int(elapsed * 1000),
                payload={
                    "method": request.method,
                    "route": _route_template(request),
                    "path": request.url.path,
                    "status": status_code,
                },
            )
        except Exception:
            # Instrumentation must not be able to fail a request it only observes.
            logger.debug("Telemetry middleware failed to record a request.",
                         exc_info=True)


def _route_template(request: Request) -> str:
    """The matched route pattern, e.g. `/images/{image_id}`; falls back to the path."""
    route = request.scope.get("route")
    return getattr(route, "path", None) or request.url.path


def _username_from_request(request: Request) -> str | None:
    """Read the caller from the bearer token without hitting the database.

    The token's subject is enough to attribute an event; loading the user would
    add a query to every instrumented request for no extra information.
    """
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    return _username_from_token(header[len("bearer "):].strip())


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

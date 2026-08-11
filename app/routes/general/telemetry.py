"""Telemetry ingest, runtime configuration and study-data export.

The router is only mounted when ``USER_EVENTS_ENABLED`` is set, so a deployment
that must not collect data does not merely refuse these calls -- it does not
expose them at all.

Ingest is intentionally lenient about authentication: a client may emit events
before login (a route change on the login page, say), and rejecting those would
lose the start of every session. Whatever token *is* present decides the
`username` on the row; the client never supplies it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from logging import getLogger
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.user_events import UserEvents
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services.auth import _username_from_token, load_user
from app.services.permissions import require_global
from app.services.telemetry import config as telemetry_config
from app.services.telemetry.export import build_query, stream_csv, stream_jsonl
from app.services.telemetry.recorder import truncate_payload, recorder
from app.services.telemetry.schemas import (
    TelemetryBatch,
    TelemetryConfigUpdate,
    TelemetryIngestResponse,
)

router = APIRouter(prefix="/telemetry", tags=["telemetry"])
logger = getLogger(__name__)


async def optional_user(request: Request,
                        db: Session = Depends(get_session)) -> AuthenticatedUser | None:
    """Resolve the caller if a usable bearer token is present, else None.

    Unlike `get_current_user` this never raises: an expired token on a telemetry
    flush should cost the events their username, not fail the request and make
    the client retry a batch it can never deliver.
    """
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    username = _username_from_token(header[len("bearer "):].strip())
    if username is None:
        return None
    return load_user(username, db)


@router.get("/config")
async def read_config(db: Session = Depends(get_session)):
    """The capture configuration the frontend client should apply.

    Readable without authentication so the client can decide whether to
    instrument at all before anyone has logged in. It exposes only which
    categories are being captured, which the participant is entitled to know.
    """
    return {"success": True, "config": telemetry_config.get_config(db).as_response()}


@router.put("/config")
async def update_config(
        update: TelemetryConfigUpdate,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.TELEMETRY_MANAGE)),
):
    """Flip study logging or individual components without a redeploy."""
    current = telemetry_config.get_config(db)
    if not current.enabled:
        # Should be unreachable while the router is only mounted when enabled,
        # but stated explicitly so the lock does not depend on that wiring.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Telemetry is disabled for this deployment (USER_EVENTS_ENABLED).",
        )

    components = (frozenset(update.components)
                  if update.components is not None else None)
    resolved = telemetry_config.save_config(
        db,
        capture_enabled=update.capture_enabled,
        components=components,
        updated_by=user.username,
    )
    if resolved.capture_enabled:
        recorder.start()
    logger.info("Telemetry config updated by %s: study=%s components=%s",
                user.username, resolved.capture_enabled,
                telemetry_config.format_components(resolved.components))
    return {"success": True, "config": resolved.as_response()}


@router.post("/events", response_model=TelemetryIngestResponse)
async def ingest_events(
        batch: TelemetryBatch,
        db: Session = Depends(get_session),
        user: AuthenticatedUser | None = Depends(optional_user),
):
    """Accept a flush of client events.

    Returns 200 even when everything was dropped: a client whose batch arrives
    just after an admin switched a component off should discard it and carry on,
    not retry forever.
    """
    current = telemetry_config.get_config(db)
    if len(batch.events) > current.max_batch:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Batch exceeds the maximum of {current.max_batch} events.",
        )

    now = datetime.now(timezone.utc)
    username = user.username if user is not None else None
    rows = []
    dropped = 0
    for event in batch.events:
        if not current.captures(event.component):
            dropped += 1
            continue
        rows.append({
            "event_id": event.event_id,
            "ts": _as_naive_utc(event.ts),
            "received_at": now,
            "username": username,
            "session_id": event.session_id,
            "component": event.component.value,
            "event_type": event.event_type,
            "source": "frontend",
            "dataset_id": event.dataset_id,
            "image_id": event.image_id,
            "duration_ms": event.duration_ms,
            "payload": truncate_payload(event.payload),
            "client": event.client,
        })

    accepted = recorder.record_rows(rows)
    # Anything the queue refused is a drop too, so the client's counters and the
    # server's agree.
    dropped += len(rows) - accepted
    return TelemetryIngestResponse(accepted=accepted, dropped=dropped)


@router.get("/events")
async def list_events(
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.TELEMETRY_MANAGE)),
        start: datetime | None = None,
        end: datetime | None = None,
        username: str | None = None,
        session_id: str | None = None,
        component: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
):
    """Paged view of stored events, for checking a study run is capturing."""
    query = build_query(db, start=start, end=end, username=username,
                        session_id=session_id, component=component)
    total = query.count()
    events = query.offset(offset).limit(limit).all()
    return {
        "success": True,
        "total": total,
        "events": [event.as_dict() for event in events],
    }


@router.get("/sessions")
async def list_sessions(
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.TELEMETRY_MANAGE)),
):
    """One row per captured session: who, when, how many events.

    The index a study runner works from -- it turns "which session was that
    participant's second run?" into something answerable without an export.
    """
    rows = (
        db.query(
            UserEvents.session_id,
            # One row per session, not per (session, username). The unload flush
            # goes out via sendBeacon, which cannot carry an Authorization header,
            # so those events have a null username -- grouping on it split a single
            # session into a named row and a phantom "anonymous" one. `max` skips
            # nulls, and a session is one login, so the non-null values all agree.
            func.max(UserEvents.username).label("username"),
            func.count(UserEvents.id).label("event_count"),
            func.min(UserEvents.ts).label("started_at"),
            func.max(UserEvents.ts).label("ended_at"),
        )
        .group_by(UserEvents.session_id)
        .order_by(func.min(UserEvents.ts).desc())
        .all()
    )
    return {
        "success": True,
        "sessions": [
            {
                "session_id": row.session_id,
                "username": row.username,
                "event_count": row.event_count,
                "started_at": row.started_at.isoformat() if row.started_at else None,
                "ended_at": row.ended_at.isoformat() if row.ended_at else None,
            }
            for row in rows
        ],
    }


@router.get("/export")
async def export_events(
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.TELEMETRY_MANAGE)),
        format: Literal["jsonl", "csv"] = "jsonl",
        start: datetime | None = None,
        end: datetime | None = None,
        username: str | None = None,
        session_id: str | None = None,
        component: str | None = None,
):
    """Stream the filtered event log for offline analysis."""
    query = build_query(db, start=start, end=end, username=username,
                        session_id=session_id, component=component)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if format == "csv":
        stream, media_type, suffix = stream_csv(query), "text/csv", "csv"
    else:
        stream, media_type, suffix = stream_jsonl(query), "application/x-ndjson", "jsonl"

    return StreamingResponse(
        stream,
        media_type=media_type,
        headers={"Content-Disposition":
                 f'attachment; filename="user-events-{stamp}.{suffix}"'},
    )


@router.delete("/events")
async def purge_events(
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.TELEMETRY_MANAGE)),
        start: datetime | None = None,
        end: datetime | None = None,
        username: str | None = None,
        session_id: str | None = None,
        component: str | None = None,
        confirm: bool = False,
):
    """Delete captured events -- pilot runs, or a participant's withdrawal.

    Takes the same filters as ``GET /telemetry/events`` and ``/export`` (via the
    shared ``build_query``), so a purge always removes exactly the set a caller
    was just looking at or exporting -- never a superset of it. An earlier version
    filtered on ``before``/``session_id``/``username`` only; a request scoped by
    component or a date range alone matched no branch there and silently deleted
    everything, which is the opposite of what "scoped" is supposed to mean for a
    destructive call.

    `confirm=true` is required when no filter narrows the delete at all, so a
    mistyped query string cannot wipe a study's data in one request.
    """
    if not any([start, end, username, session_id, component]) and not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Refusing to delete every event without confirm=true.",
        )

    # `build_query` orders by ts/id for the read paths; a bulk DELETE cannot carry
    # that ORDER BY (SQLAlchemy raises InvalidRequestError), so it is dropped here.
    query = build_query(db, start=start, end=end, username=username,
                        session_id=session_id, component=component).order_by(None)
    deleted = query.delete(synchronize_session=False)
    db.commit()
    logger.info("%s purged %d telemetry event(s).", user.username, deleted)
    return {"success": True, "deleted": deleted}


def _as_naive_utc(value: datetime) -> datetime:
    """Normalise a client timestamp to naive UTC.

    The column is timezone-naive (SQLite has no native tz type), so a mix of
    aware and naive values would otherwise sort incorrectly against each other.
    """
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)

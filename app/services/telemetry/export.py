"""Study-data export: stream stored events as JSONL or CSV.

Streamed in id order and in chunks rather than materialised, because a study
session is easily hundreds of thousands of rows and the export is the one place
that reads all of them at once.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Iterator

from sqlalchemy.orm import Query, Session

from app.database.user_events import EXPORT_FIELDS, UserEvents

#: Rows pulled from the database per round trip while streaming.
_CHUNK = 1000


def build_query(db: Session,
                *,
                start: datetime | None = None,
                end: datetime | None = None,
                username: str | None = None,
                session_id: str | None = None,
                component: str | None = None) -> Query:
    """Filtered event query, oldest first (the order analysis expects)."""
    query = db.query(UserEvents)
    if start is not None:
        query = query.filter(UserEvents.ts >= start)
    if end is not None:
        query = query.filter(UserEvents.ts <= end)
    if username:
        query = query.filter(UserEvents.username == username)
    if session_id:
        query = query.filter(UserEvents.session_id == session_id)
    if component:
        query = query.filter(UserEvents.component == component)
    return query.order_by(UserEvents.ts, UserEvents.id)


def stream_jsonl(query: Query) -> Iterator[str]:
    """One JSON object per line. `payload` is inlined as an object, not a string."""
    for event in query.yield_per(_CHUNK):
        row = event.as_dict()
        row["payload"] = _decode_payload(row["payload"])
        yield json.dumps(row, default=str) + "\n"


def stream_csv(query: Query) -> Iterator[str]:
    """CSV with a header row; `payload` stays a JSON string in its cell."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EXPORT_FIELDS,
                            extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    yield _drain(buffer)

    for event in query.yield_per(_CHUNK):
        writer.writerow(event.as_dict())
        yield _drain(buffer)


def _drain(buffer: io.StringIO) -> str:
    value = buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)
    return value


def _decode_payload(raw: str | None):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw

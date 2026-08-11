"""Buffered writer for user events.

Telemetry must never be the reason an annotation request is slow, so nothing on
a request path touches the database. `record()` drops a dict into a bounded
in-memory queue and returns; a single background thread drains the queue and
writes in bulk.

A thread (rather than an asyncio task) is deliberate: emit sites live in async
routes, sync service functions and the WebSocket handlers alike, and a plain
`queue.Queue` is safe from all three without every caller needing a running loop.

Overflow drops rather than blocks. A study that loses a handful of events under
load is a far better outcome than one where the annotation UI stalls behind its
own instrumentation; the drop count is logged so the loss is never silent.
"""
from __future__ import annotations

import json
import queue
import threading
from datetime import datetime, timezone
from logging import getLogger
from typing import Any
from uuid import uuid4

import config as app_config
from app.database import SessionLocal
from app.database.user_events import UserEvents
from app.services.telemetry.config import TelemetryComponent, get_config

logger = getLogger(__name__)

#: How long the drain thread waits for more events before writing what it has.
_DRAIN_INTERVAL_SECONDS = 2.0
#: Upper bound on rows per INSERT.
_DRAIN_BATCH_SIZE = 500

_SHUTDOWN = object()


def truncate_payload(payload: dict[str, Any] | None) -> str | None:
    """JSON-encode an event payload, capped at the configured byte budget.

    An over-long payload is replaced rather than sliced: a truncated JSON string
    is not parseable, and a row that cannot be parsed at analysis time is worse
    than one that says plainly why it is missing.
    """
    if not payload:
        return None
    try:
        encoded = json.dumps(payload, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        logger.warning("Telemetry payload was not JSON-encodable; storing a marker.")
        return json.dumps({"_error": "payload_not_serialisable"})

    limit = app_config.USER_EVENTS_MAX_PAYLOAD_BYTES
    if len(encoded.encode("utf-8")) <= limit:
        return encoded
    return json.dumps({"_error": "payload_too_large", "_bytes": len(encoded)})


class TelemetryRecorder:
    """Queue plus drain thread. One instance per process (see `recorder` below)."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=app_config.USER_EVENTS_QUEUE_SIZE)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._dropped = 0
        #: Set in tests to write synchronously and skip the thread entirely.
        self.synchronous = False

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Start the drain thread. Idempotent; a no-op when capture is locked off."""
        if not get_config().enabled:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._drain_forever, name="telemetry-drain", daemon=True)
            self._thread.start()
        logger.info("Telemetry recorder started.")

    def stop(self, timeout: float = 5.0) -> None:
        """Flush what is queued and stop the drain thread."""
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is None:
            return
        try:
            self._queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            # The sentinel cannot get in, so make room by dropping one event.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(_SHUTDOWN)
            except (queue.Empty, queue.Full):
                pass
        thread.join(timeout=timeout)
        if self._dropped:
            logger.warning("Telemetry dropped %d event(s) due to queue overflow.",
                           self._dropped)
        logger.info("Telemetry recorder stopped.")

    # -- ingest -----------------------------------------------------------

    def record(self,
               component: TelemetryComponent,
               event_type: str,
               *,
               source: str = "backend",
               username: str | None = None,
               session_id: str | None = None,
               dataset_id: int | None = None,
               image_id: int | None = None,
               duration_ms: int | None = None,
               payload: dict[str, Any] | None = None,
               client: str | None = None,
               ts: datetime | None = None,
               event_id: str | None = None) -> bool:
        """Queue one event. Returns False when it was gated off or dropped.

        Safe to call unconditionally from anywhere: the component gate is checked
        here, so emit sites do not each need their own `if enabled` guard.
        """
        if not get_config().captures(component):
            return False

        row = {
            "event_id": event_id or str(uuid4()),
            "ts": ts or datetime.now(timezone.utc),
            "received_at": datetime.now(timezone.utc),
            "username": username,
            "session_id": session_id,
            "component": component.value,
            "event_type": event_type,
            "source": source,
            "dataset_id": dataset_id,
            "image_id": image_id,
            "duration_ms": duration_ms,
            "payload": truncate_payload(payload),
            "client": client,
        }
        return self._enqueue(row)

    def record_rows(self, rows: list[dict[str, Any]]) -> int:
        """Queue pre-built rows (the ingest endpoint's path). Returns how many took."""
        return sum(1 for row in rows if self._enqueue(row))

    def _enqueue(self, row: dict[str, Any]) -> bool:
        if self.synchronous:
            _write_rows([row])
            return True
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning("Telemetry queue full; dropped %d event(s) so far.",
                               self._dropped)
            return False
        # A queue with nothing draining it would fill up and then drop everything,
        # so a first event after boot starts the thread rather than waiting for a
        # lifespan hook that a test client may never run.
        if self._thread is None or not self._thread.is_alive():
            self.start()
        return True

    @property
    def dropped(self) -> int:
        return self._dropped

    # -- drain ------------------------------------------------------------

    def _drain_forever(self) -> None:
        while True:
            batch, shutting_down = self._collect_batch()
            if batch:
                _write_rows(batch)
            if shutting_down:
                return

    def _collect_batch(self) -> tuple[list[dict[str, Any]], bool]:
        """Block for one event, then sweep up whatever else is already waiting."""
        batch: list[dict[str, Any]] = []
        try:
            first = self._queue.get(timeout=_DRAIN_INTERVAL_SECONDS)
        except queue.Empty:
            return batch, False
        if first is _SHUTDOWN:
            return batch, True
        batch.append(first)

        while len(batch) < _DRAIN_BATCH_SIZE:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _SHUTDOWN:
                return batch, True
            batch.append(item)
        return batch, False


def _write_rows(rows: list[dict[str, Any]]) -> None:
    """Bulk-insert a batch, tolerating replays of an already-stored event.

    A client that retries a flush resends events the server already has. The
    unique `event_id` turns that into an IntegrityError for the whole batch, so
    the retry falls back to inserting row by row and skipping the duplicates.
    """
    session = SessionLocal()
    try:
        session.bulk_insert_mappings(UserEvents, rows)
        session.commit()
    except Exception:
        session.rollback()
        _write_rows_individually(session, rows)
    finally:
        session.close()


def _write_rows_individually(session, rows: list[dict[str, Any]]) -> None:
    stored = 0
    for row in rows:
        try:
            session.add(UserEvents(**row))
            session.commit()
            stored += 1
        except Exception:
            session.rollback()
    if stored != len(rows):
        logger.debug("Telemetry stored %d/%d event(s); the rest were duplicates or invalid.",
                     stored, len(rows))


#: Process-wide recorder. Import this, not the class.
recorder = TelemetryRecorder()

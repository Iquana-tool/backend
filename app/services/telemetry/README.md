# User event capture

Records individual user actions, in sequence, attributed to a username, for user
studies. Off by default.

This is not telemetry in the ops sense — there is no aggregate or anonymous mode.
Every row is one person's action. "Telemetry" survives only as the internal module
and route name.

## The two flags

`USER_EVENTS_ENABLED` and `USER_EVENTS_CAPTURE` are **two levels of one gate, not
two kinds of data**. Nothing is recorded unless both are true.

| | `USER_EVENTS_ENABLED` | `USER_EVENTS_CAPTURE` |
|---|---|---|
| Role | Deployment lock | Capture switch |
| Read | Once, at boot | Per event |
| To change | Restart | Live API call |
| Stored in | Env only | Env default, `telemetry_settings` row wins |
| Controlled by | Whoever deploys | Any admin with `telemetry.manage` |
| Effect when false | Routes and middleware do not exist | They exist; nothing is recorded |

The split is about *who* can change it and *how fast*. Starting a study is routine —
an admin flips a switch mid-session. Deciding a deployment may collect at all is a
governance act that should need access to the deploy config, not an admin password.

## Switching it on

Set `USER_EVENTS_ENABLED=true` and restart. Then, live and without a restart:

```bash
curl -X PUT localhost:8000/telemetry/config \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"capture_enabled": true, "components": ["annotation", "ai"]}'
```

Runtime choices persist in `telemetry_settings` across restarts. The lock always
wins: a stored row can never enable a deployment whose environment says off.

## Components

Each is switchable on its own.

| Component | Captures |
|---|---|
| `annotation` | tool/mode switches, prompt placement, contour create/edit/delete, labelling, undo/redo, image open |
| `ai` | invocation, latency, result counts, suggestion accept/reject |
| `navigation` | login/logout, route changes with dwell time, tab visibility |
| `api` | HTTP request and WebSocket message timings, statuses, errors |

## Endpoints

| Method | Path | Auth |
|---|---|---|
| `GET` | `/telemetry/config` | none — the client needs it before login |
| `PUT` | `/telemetry/config` | `telemetry.manage` |
| `POST` | `/telemetry/events` | optional; the token decides the username |
| `GET` | `/telemetry/events` | `telemetry.manage` |
| `GET` | `/telemetry/sessions` | `telemetry.manage` |
| `GET` | `/telemetry/export?format=jsonl\|csv` | `telemetry.manage` |
| `DELETE` | `/telemetry/events` | `telemetry.manage` |

Export and purge take `start`, `end`, `username`, `session_id`, `component`.
An unfiltered purge needs `confirm=true`.

## Adding a capture point

Backend — call an `emit.py` helper. Gating happens inside, so no `if enabled` guard:

```python
from app.services.telemetry.emit import emit_annotation, track_duration
from app.services.telemetry.config import TelemetryComponent

emit_annotation("mask.submit", username=user.username, image_id=image_id)

with track_duration(TelemetryComponent.AI, "ai.prompted.invoke",
                    username=user_id) as span:
    span["model"] = model_key
    result = await service.inference(request)   # failures are recorded too
```

Frontend — same idea, `track()` is a no-op until config says otherwise:

```js
import { trackAnnotation } from '../../services/telemetry';

trackAnnotation('tool.switch', { payload: { from, to } });
```

New component? Add it to `TelemetryComponent` in `config.py` and to the mirror in
`services/telemetry.js`. The config endpoint, client and admin toggle pick it up.

## Design notes

**Nothing on a request path touches the database.** `record()` puts a dict on a
bounded queue; a background thread bulk-inserts. Overflow drops and logs rather
than blocking — a study losing a few events beats an annotation UI stalling
behind its own instrumentation.

**`user_events` has no foreign keys.** Study data has to stay analysable after the
dataset it was gathered on is deleted; a cascade would destroy the record of a
session a paper depends on.

**Replayed flushes dedup** on the client-generated `event_id`, so a batch that
timed out and was resent is not double-counted.

**High-frequency actions are deliberately not captured** — `moveVertex` fires on
every pointer frame of a drag. The vertex count at edit end captures the outcome
without thousands of near-identical rows.

**Payloads are structured and capped** (`USER_EVENTS_MAX_PAYLOAD_BYTES`). No image
data, no coordinates, no label text — only ids and counts. An over-long payload is
replaced with a marker, not sliced, since truncated JSON will not parse at
analysis time.

## Privacy

Event rows carry the authenticated username, so this is personal data under GDPR.
Export and purge are admin-only, and `DELETE /telemetry/events?session_id=…`
removes one participant's data if they withdraw. Consent and retention are the
study's responsibility, not the tool's.

Pseudonymisation (hashing the username at ingest) is a contained change to
`ingest_events` in `routes/general/telemetry.py` if an ethics review asks for it.

## Files

| | |
|---|---|
| `config.py` | env lock + persisted runtime override |
| `schemas.py` | ingest/config wire format |
| `recorder.py` | queue + background writer |
| `emit.py` | helpers for backend capture points |
| `middleware.py` | HTTP capture for `api` |
| `export.py` | JSONL/CSV streaming |
| `app/database/user_events.py` | the event table |
| `app/database/telemetry_settings.py` | the single-row override |
| `tests/test_telemetry.py` | backend tests |
| `frontend-react/src/services/telemetry.js` | client |

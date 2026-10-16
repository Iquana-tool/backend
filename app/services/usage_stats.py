"""Per-user usage statistics for the admin Usage view.

Two sources, with different availability:

* ``ai_suggestions`` (via `suggestion_stats`) is always written, so what each person
  did with AI suggestions -- kept, edited (and by how much), rejected -- is available
  on every deployment.
* ``user_events`` only fills while the activity log captures. Workflow time, tools,
  AI invocations, model choices, the per-day timeline and per-image effort all come
  from there, and are empty on a deployment with capture off. The response says
  which case applies (``events_enabled``) so the UI can explain an empty section
  instead of suggesting nobody used the tool.

Event payloads are JSON text, so the summaries that need payload fields read the
relevant event types and aggregate in Python. Only the handful of event types the
view needs are loaded -- never the high-volume ``api.request`` rows.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median
from typing import Any, Iterable

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.database.ai_suggestions import AiSuggestions
from app.database.user_events import UserEvents
from app.services.activity_log.config import get_config
from app.services.suggestion_stats import get_suggestion_stats

#: Route patterns mapped to the workflow a person is in. First match wins, so the
#: more specific patterns come first. Paths are the frontend's (react-router) paths.
_WORKFLOWS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^/dataset/\d+/annotate"), "Annotation"),
    (re.compile(r"^/annotate-v2"), "Annotation"),
    (re.compile(r"^/dataset/\d+/view"), "Viewing"),
    (re.compile(r"^/dataset/\d+/review"), "Review"),
    (re.compile(r"^/dataset/\d+/correct"), "Correction"),
    (re.compile(r"^/dataset/\d+/quantifications"), "Quantification"),
    (re.compile(r"^/dataset/\d+/training"), "Training"),
    (re.compile(r"^/dataset/\d+/model-orchestration"), "Model orchestration"),
    (re.compile(r"^/dataset/\d+/inference"), "Batch inference"),
    (re.compile(r"^/dataset/\d+/datamanagement"), "Data management"),
    (re.compile(r"^/dataset/\d+/access"), "Access management"),
    (re.compile(r"^/datasets"), "Datasets overview"),
    (re.compile(r"^/models"), "Model zoo"),
    (re.compile(r"^/admin"), "Admin"),
    (re.compile(r"^/docs"), "Documentation"),
    (re.compile(r"^/(account|organizations)"), "Account"),
    (re.compile(r"^/(login|invites)"), "Sign-in"),
    (re.compile(r"^/$"), "Home"),
]

#: Readable names for the backend AI invocation events.
_AI_FEATURES = {
    "ai.prompted.invoke": "Prompted segmentation",
    "ai.semantic.invoke": "Semantic segmentation",
    "ai.instance.invoke": "Instance segmentation",
    "ai.suggestion.invoke": "Instance suggestion",
}

#: Event types the detail view reads payloads from.
_DETAIL_EVENT_TYPES = ["route.change", "tool.switch", "image.visit", "ai.model.select",
                       *_AI_FEATURES]

_INTERACTION_KEYS = ["clicks", "prompts", "edits", "creates", "deletes", "relabels", "undos"]


def workflow_for_path(path: str | None) -> str:
    """The workflow a frontend path belongs to, or "Other"."""
    if not path:
        return "Other"
    for pattern, name in _WORKFLOWS:
        if pattern.search(path):
            return name
    return "Other"


def _payload(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _day(value) -> str:
    """`func.date` returns a string on SQLite and a date on PostgreSQL."""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _int(value) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _event_filter(username: str | None, since: datetime | None) -> list:
    conditions = []
    if username is not None:
        conditions.append(UserEvents.username == username)
    if since is not None:
        conditions.append(UserEvents.ts >= since)
    return conditions


# -- Overview ---------------------------------------------------------------


def list_user_usage(db: Session, since: datetime | None) -> dict:
    """One summary row per user who has suggestions or captured events."""
    users: dict[str, dict] = {}

    def row(username: str) -> dict:
        return users.setdefault(username, {
            "username": username, "suggestions": 0, "as_is": 0, "edited": 0, "rejected": 0,
            "mean_edit_iou": None, "events": 0, "sessions": 0, "active_days": 0,
            "images_visited": 0, "active_ms": 0, "last_seen": None,
        })

    rejected = AiSuggestions.deleted_at.is_not(None)
    edited = AiSuggestions.deleted_at.is_(None) & AiSuggestions.edited_at.is_not(None)
    suggestion_conditions = [AiSuggestions.superseded_at.is_(None),
                             AiSuggestions.username.is_not(None)]
    if since is not None:
        suggestion_conditions.append(AiSuggestions.created_at >= since)
    for username, total, n_rejected, n_edited, mean_iou, last in db.execute(
            select(AiSuggestions.username, func.count(),
                   func.sum(case((rejected, 1), else_=0)),
                   func.sum(case((edited, 1), else_=0)),
                   func.avg(case((edited, AiSuggestions.final_iou))),
                   func.max(AiSuggestions.created_at))
            .where(*suggestion_conditions)
            .group_by(AiSuggestions.username)):
        entry = row(username)
        n_rejected, n_edited = int(n_rejected or 0), int(n_edited or 0)
        entry.update(suggestions=total, rejected=n_rejected, edited=n_edited,
                     as_is=total - n_rejected - n_edited,
                     mean_edit_iou=round(float(mean_iou), 4) if mean_iou is not None else None)
        entry["last_seen"] = last

    event_conditions = [UserEvents.username.is_not(None), *_event_filter(None, since)]
    for username, n_events, n_sessions, n_days, last in db.execute(
            select(UserEvents.username, func.count(),
                   func.count(func.distinct(UserEvents.session_id)),
                   func.count(func.distinct(func.date(UserEvents.ts))),
                   func.max(UserEvents.ts))
            .where(*event_conditions)
            .group_by(UserEvents.username)):
        entry = row(username)
        entry.update(events=n_events, sessions=n_sessions, active_days=n_days)
        if last is not None and (entry["last_seen"] is None or last > entry["last_seen"]):
            entry["last_seen"] = last

    visit_images: dict[str, set] = defaultdict(set)
    for username, image_id, raw in db.execute(
            select(UserEvents.username, UserEvents.image_id, UserEvents.payload)
            .where(UserEvents.event_type == "image.visit", *event_conditions)):
        users[username]["active_ms"] += _int(_payload(raw).get("active_ms"))
        if image_id is not None:
            visit_images[username].add(image_id)
    for username, images in visit_images.items():
        users[username]["images_visited"] = len(images)

    # Most recently active first; ties (and people never seen) by name.
    ordered = sorted(users.values(), key=lambda u: u["username"])
    ordered.sort(key=lambda u: u["last_seen"] or datetime.min, reverse=True)
    for entry in ordered:
        entry["last_seen"] = entry["last_seen"].isoformat() if entry["last_seen"] else None
    return {"events_enabled": get_config(db).enabled, "users": ordered}


# -- Detail -----------------------------------------------------------------


def get_user_usage(db: Session, username: str, since: datetime | None) -> dict:
    """Everything the Usage view shows for one person."""
    events = db.execute(
        select(UserEvents.event_type, UserEvents.ts, UserEvents.image_id,
               UserEvents.duration_ms, UserEvents.payload)
        .where(UserEvents.event_type.in_(_DETAIL_EVENT_TYPES),
               *_event_filter(username, since))
        .order_by(UserEvents.ts)
    ).all()

    return {
        "username": username,
        "events_enabled": get_config(db).enabled,
        "suggestions": get_suggestion_stats(None, since, db, username=username),
        "iou_distribution": _iou_distribution(db, username, since),
        "workflows": _workflows(events),
        "tools": _tools(events),
        "ai_features": _ai_features(events),
        "model_selections": _model_selections(events),
        "images": _image_effort(events),
        "timeline": _timeline(db, username, since, events),
    }


#: Bucket edges for how much an edit kept of the suggested outline.
_IOU_BUCKETS = [(0.95, "≥ 0.95 (minor touch-up)"), (0.8, "0.80–0.95"),
                (0.5, "0.50–0.80"), (0.0, "< 0.50 (largely redrawn)")]


def _iou_distribution(db: Session, username: str, since: datetime | None) -> list[dict]:
    conditions = [AiSuggestions.username == username, AiSuggestions.superseded_at.is_(None),
                  AiSuggestions.deleted_at.is_(None), AiSuggestions.final_iou.is_not(None)]
    if since is not None:
        conditions.append(AiSuggestions.created_at >= since)
    counts = Counter()
    for (iou,) in db.execute(select(AiSuggestions.final_iou).where(*conditions)):
        for edge, name in _IOU_BUCKETS:
            if iou >= edge:
                counts[name] += 1
                break
    return [{"bucket": name, "count": counts[name]} for _, name in _IOU_BUCKETS]


def _workflows(events: Iterable) -> list[dict]:
    """Time on each workflow, from the dwell time route changes report on leaving."""
    totals: dict[str, dict] = {}
    for event_type, _ts, _image, _duration, raw in events:
        if event_type != "route.change":
            continue
        payload = _payload(raw)
        if not payload.get("from"):
            continue
        name = workflow_for_path(payload["from"])
        entry = totals.setdefault(name, {"workflow": name, "time_ms": 0, "visits": 0})
        entry["time_ms"] += _int(payload.get("dwell_ms"))
        entry["visits"] += 1
    return sorted(totals.values(), key=lambda w: -w["time_ms"])


def _tools(events: Iterable) -> list[dict]:
    """How often each annotation tool was switched to."""
    counts = Counter(_payload(raw).get("to") for event_type, _ts, _image, _duration, raw in events
                     if event_type == "tool.switch")
    counts.pop(None, None)
    return [{"tool": tool, "count": n} for tool, n in counts.most_common()]


def _ai_features(events: Iterable) -> list[dict]:
    """AI invocations by feature and model, with latency and failure counts."""
    groups: dict[tuple[str, str], dict] = {}
    for event_type, _ts, _image, duration, raw in events:
        if event_type not in _AI_FEATURES:
            continue
        payload = _payload(raw)
        model = payload.get("model") or "unknown"
        entry = groups.setdefault((event_type, model), {
            "feature": _AI_FEATURES[event_type], "model": model, "count": 0,
            "failures": 0, "_durations": [],
        })
        entry["count"] += 1
        if payload.get("ok") is False:
            entry["failures"] += 1
        if duration is not None:
            entry["_durations"].append(duration)
    result = []
    for entry in sorted(groups.values(), key=lambda g: -g["count"]):
        durations = entry.pop("_durations")
        entry["median_ms"] = int(median(durations)) if durations else None
        result.append(entry)
    return result


def _model_selections(events: Iterable) -> list[dict]:
    counts = Counter()
    for event_type, _ts, _image, _duration, raw in events:
        if event_type == "ai.model.select":
            payload = _payload(raw)
            counts[(payload.get("tool") or "unknown", payload.get("model") or "unknown")] += 1
    return [{"tool": tool, "model": model, "count": n}
            for (tool, model), n in counts.most_common()]


def _image_effort(events: Iterable) -> dict:
    """Per-image effort from `image.visit` summaries."""
    active, idle, ai_wait, open_ = [], [], [], []
    images = set()
    interactions = Counter()
    for event_type, _ts, image_id, _duration, raw in events:
        if event_type != "image.visit":
            continue
        payload = _payload(raw)
        active.append(_int(payload.get("active_ms")))
        idle.append(_int(payload.get("idle_ms")))
        ai_wait.append(_int(payload.get("ai_wait_ms")))
        open_.append(_int(payload.get("open_ms")))
        if image_id is not None:
            images.add(image_id)
        for key, value in (payload.get("interactions") or {}).items():
            if key in _INTERACTION_KEYS:
                interactions[key] += _int(value)

    visits = len(active)

    def mean(values: list[int]) -> int | None:
        return int(sum(values) / len(values)) if values else None

    return {
        "visits": visits,
        "images": len(images),
        "total_active_ms": sum(active),
        "mean_active_ms": mean(active),
        "median_active_ms": int(median(active)) if active else None,
        "mean_idle_ms": mean(idle),
        "mean_ai_wait_ms": mean(ai_wait),
        "mean_open_ms": mean(open_),
        "interactions": {key: interactions[key] for key in _INTERACTION_KEYS},
    }


def _timeline(db: Session, username: str, since: datetime | None, events: list) -> list[dict]:
    """One row per day with any activity: what was used, and how much."""
    days: dict[str, dict] = {}

    def day(key) -> dict:
        key = _day(key)
        return days.setdefault(key, {"date": key, "events": 0, "active_ms": 0,
                                     "suggestions": 0, "workflows": {}})

    for value, n in db.execute(
            select(func.date(UserEvents.ts), func.count())
            .where(*_event_filter(username, since))
            .group_by(func.date(UserEvents.ts))):
        day(value)["events"] = n

    suggestion_conditions = [AiSuggestions.username == username,
                             AiSuggestions.superseded_at.is_(None)]
    if since is not None:
        suggestion_conditions.append(AiSuggestions.created_at >= since)
    for value, n in db.execute(
            select(func.date(AiSuggestions.created_at), func.count())
            .where(*suggestion_conditions)
            .group_by(func.date(AiSuggestions.created_at))):
        day(value)["suggestions"] = n

    for event_type, ts, _image, _duration, raw in events:
        if event_type == "image.visit":
            day(ts.date())["active_ms"] += _int(_payload(raw).get("active_ms"))
        elif event_type == "route.change":
            payload = _payload(raw)
            if payload.get("from"):
                workflows = day(ts.date())["workflows"]
                name = workflow_for_path(payload["from"])
                workflows[name] = workflows.get(name, 0) + _int(payload.get("dwell_ms"))

    return sorted(days.values(), key=lambda d: d["date"])

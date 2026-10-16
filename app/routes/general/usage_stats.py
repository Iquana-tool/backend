"""Per-user usage statistics for admins.

Mounted on every deployment, unlike ``/activity-log``: the AI suggestion part comes
from the suggestion log, which is written whether or not study capture is enabled.
The response's ``events_enabled`` tells the client when the event-based sections
are empty because nothing is captured. See `app.services.usage_stats`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_session
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services import usage_stats
from app.services.permissions import require_global

router = APIRouter(prefix="/usage-stats", tags=["usage-stats"])


def _since(days: int) -> datetime | None:
    """Start of the window; `days=0` means all time. Naive UTC, like the stored timestamps."""
    if not days:
        return None
    return datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)


@router.get("/users")
async def list_users(
        days: int = Query(default=30, ge=0, le=3650),
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.ACTIVITY_LOG_MANAGE)),
):
    """One summary row per user: suggestion outcomes, activity and active time."""
    since = _since(days)
    return {"success": True, "since": since.isoformat() if since else None,
            **usage_stats.list_user_usage(db, since)}


@router.get("/users/{username}")
async def user_detail(
        username: str,
        days: int = Query(default=30, ge=0, le=3650),
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.ACTIVITY_LOG_MANAGE)),
):
    """Suggestions by model, workflows, tools, AI use, timeline and per-image effort."""
    since = _since(days)
    return {"success": True, "since": since.isoformat() if since else None,
            **usage_stats.get_user_usage(db, username, since)}

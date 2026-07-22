"""Per-dataset membership and invite links.

`DatasetMembers` replaces the old flat `dataset_user_association` table: sharing a
dataset now records *which* role the collaborator holds rather than just that they
have access.

Ownership is modelled as a membership row (`role == owner`) so it can be
transferred. `datasets.created_by` is kept untouched as immutable provenance — it
answers "who made this", not "who controls it".
"""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import relationship

from app.database import database
from app.schemas.permissions import DatasetRole


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DatasetMembers(database):
    """A user's role on one dataset.

    `extra_permissions` / `denied_permissions` are the escape hatch that keeps the
    role list short: they let one collaborator be granted (or refused) a single
    capability without inventing a role for the exception. Both are stored as
    lists of `Permission` values; denied wins over granted.
    """

    __tablename__ = "dataset_members"

    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True)
    username = Column(String, ForeignKey("users.username", ondelete="CASCADE"), primary_key=True)
    role = Column(String(20), nullable=False, default=DatasetRole.VIEWER.value)
    extra_permissions = Column(JSON, nullable=False, default=list)
    denied_permissions = Column(JSON, nullable=False, default=list)
    granted_by = Column(String, ForeignKey("users.username", ondelete="SET NULL"), nullable=True)
    granted_at = Column(DateTime, nullable=False, default=_utcnow)

    dataset = relationship("Datasets", back_populates="memberships")
    user = relationship("Users", foreign_keys=[username], back_populates="memberships")


class DatasetInvites(database):
    """A shareable invite link granting a fixed role on one dataset.

    Only the SHA-256 of the token is stored; the raw token is returned once at
    creation and never again. An invite can never grant `owner` — that is enforced
    in the schema and re-checked when the link is redeemed.
    """

    __tablename__ = "dataset_invites"

    id = Column(Integer, primary_key=True, autoincrement=True)
    token_hash = Column(String(64), nullable=False, unique=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False, default=DatasetRole.ANNOTATOR.value)
    created_by = Column(String, ForeignKey("users.username", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    expires_at = Column(DateTime, nullable=True)
    max_uses = Column(Integer, nullable=True)  # None => unlimited
    uses = Column(Integer, nullable=False, default=0)
    revoked_at = Column(DateTime, nullable=True)

    dataset = relationship("Datasets", back_populates="invites")

    def is_valid(self, now: datetime | None = None) -> bool:
        """Whether this link can still be redeemed."""
        now = now or _utcnow()
        if self.revoked_at is not None:
            return False
        if self.expires_at is not None and _as_utc(self.expires_at) <= now:
            return False
        if self.max_uses is not None and (self.uses or 0) >= self.max_uses:
            return False
        return True


def _as_utc(value: datetime) -> datetime:
    """SQLite hands back naive datetimes; treat those as UTC for comparisons."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

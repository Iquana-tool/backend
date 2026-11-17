"""Registration links: the only way to create your own account.

There is no open self-registration. An admin mints a link on the admin page and sends
it to the person; following it lets them choose a username and password, once (or as
often as ``max_uses`` allows) and until it expires. Like dataset invite links, only the
SHA-256 of the token is stored: the link is a bearer secret, shown once at creation.

The account is created as a plain platform member and joins no dataset; dataset access
is granted afterwards, by a dataset invite or by the dataset's owner.
"""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String

from app.database import database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite and PostgreSQL return naive datetimes here; they are written in UTC."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class AccountInvites(database):
    __tablename__ = "account_invites"

    id = Column(Integer, primary_key=True, autoincrement=True)
    token_hash = Column(String(64), nullable=False, unique=True, index=True)
    created_by = Column(String, ForeignKey("users.username", ondelete="CASCADE", onupdate="CASCADE"),
                        nullable=False)
    #: Who the link is for, as the admin wrote it ("Anna, coral lab"); never shown to the invitee.
    note = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    expires_at = Column(DateTime, nullable=True)
    max_uses = Column(Integer, nullable=True)  # None => unlimited
    uses = Column(Integer, nullable=False, default=0)
    revoked_at = Column(DateTime, nullable=True)

    def is_valid(self, now: datetime | None = None) -> bool:
        """Whether this link can still be used to register."""
        now = now or _utcnow()
        if self.revoked_at is not None:
            return False
        if self.expires_at is not None and _as_utc(self.expires_at) <= now:
            return False
        if self.max_uses is not None and (self.uses or 0) >= self.max_uses:
            return False
        return True

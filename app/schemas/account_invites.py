"""Schemas for registration links (see ``app.database.account_invites``)."""
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class AccountInviteCreate(BaseModel):
    """What an admin chooses when minting a registration link."""

    expires_in_hours: int | None = Field(
        168, ge=1, le=24 * 365,
        description="Lifetime of the link in hours. None means it never expires.",
    )
    max_uses: int | None = Field(
        1, ge=1,
        description="How many accounts may be created with the link. Defaults to one; "
                    "None means unlimited.",
    )
    note: str | None = Field(
        None, max_length=100,
        description="Who the link is for, for the admin's own list. Not shown to the invitee.",
    )

    @field_validator("note", mode="before")
    @classmethod
    def _blank_note_is_none(cls, value):
        if isinstance(value, str):
            return value.strip() or None
        return value


class AccountInviteRead(BaseModel):
    """A link as its creator sees it. The raw token is never included."""

    id: int
    created_by: str
    note: str | None = None
    created_at: datetime
    expires_at: datetime | None = None
    max_uses: int | None = None
    uses: int = 0
    revoked_at: datetime | None = None
    is_valid: bool


class AccountInvitePreview(BaseModel):
    """What someone following the link sees before registering."""

    is_valid: bool
    invited_by: str
    expires_at: datetime | None = None


class InviteRegistration(BaseModel):
    """Create your own account with a registration link."""

    token: str = Field(min_length=1)
    # Same bounds as an account an admin creates (app.schemas.review.AdminUserCreate).
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("username", mode="before")
    @classmethod
    def _strip_username(cls, value):
        """Trim surrounding whitespace so a pasted name still matches at login."""
        return value.strip() if isinstance(value, str) else value

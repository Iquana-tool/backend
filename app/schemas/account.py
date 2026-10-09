"""Request bodies for managing one's own account."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, EmailStr, Field, field_validator

#: Upper bound on the stored preferences, serialised. They ride along on every
#: /auth/me, so they are meant for a handful of UI choices, not for documents.
MAX_PREFERENCES_BYTES = 16 * 1024


def normalise_email(value):
    """Trim and lower-case, so uniqueness is case-insensitive; blank means none."""
    if isinstance(value, str):
        value = value.strip().lower()
        return value or None
    return value


class PasswordChange(BaseModel):
    """Changing one's own password, which also signs out every other session."""

    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class ProfileUpdate(BaseModel):
    """Changes to one's own profile.

    Only the fields that are sent change. ``display_name`` or ``email`` sent as null
    clears it. ``preferences`` is merged into what is stored rather than replacing
    it, so separate parts of the UI can each save their own keys; a key sent as
    null is removed.
    """

    display_name: str | None = Field(None, max_length=100)
    email: EmailStr | None = None
    preferences: dict[str, Any] | None = None

    @field_validator("display_name", mode="before")
    @classmethod
    def _blank_display_name_is_none(cls, value):
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    _normalise_email = field_validator("email", mode="before")(normalise_email)

"""Schemas for the review step: rejections, their reasons and membership payloads.

A rejection is how a reviewer sends work back to the annotator. It carries a
machine-readable `reason` from a fixed vocabulary plus an optional free-text
`note`, so a later review pipeline can aggregate "how often is this annotator
told the outline is bad?" without parsing prose.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from app.schemas.permissions import DatasetRole, GlobalRole, Permission


class RejectionReason(StrEnum):
    """Predefined reasons a reviewer can send an annotation back.

    `OTHER` requires a note; every other reason may carry one.
    """

    BAD_OUTLINE = "bad_outline"
    WRONG_LABEL = "wrong_label"
    MISSING_OBJECTS = "missing_objects"
    EXTRA_OBJECTS = "extra_objects"
    DUPLICATE_OBJECT = "duplicate_object"
    WRONG_HIERARCHY = "wrong_hierarchy"
    IMAGE_QUALITY = "image_quality"
    OTHER = "other"


#: Human-readable labels for the frontend, so the wording lives with the enum.
REJECTION_REASON_LABELS: dict[RejectionReason, str] = {
    RejectionReason.BAD_OUTLINE: "Outline is inaccurate",
    RejectionReason.WRONG_LABEL: "Wrong label assigned",
    RejectionReason.MISSING_OBJECTS: "Objects are missing",
    RejectionReason.EXTRA_OBJECTS: "Contains objects that are not there",
    RejectionReason.DUPLICATE_OBJECT: "Duplicate of another object",
    RejectionReason.WRONG_HIERARCHY: "Wrong parent/child nesting",
    RejectionReason.IMAGE_QUALITY: "Image quality too poor to annotate",
    RejectionReason.OTHER: "Other (please describe)",
}


class RejectionCreate(BaseModel):
    """Request body for rejecting a mask or a single contour."""

    reason: RejectionReason = Field(..., description="Predefined reason code.")
    note: str | None = Field(
        None,
        max_length=1000,
        description="Free-text detail. Required when reason is 'other'.",
    )
    contour_id: int | None = Field(
        None,
        description="Reject one contour. Omit to reject the mask as a whole.",
    )

    @model_validator(mode="after")
    def _require_note_for_other(self) -> "RejectionCreate":
        if self.reason is RejectionReason.OTHER and not (self.note or "").strip():
            raise ValueError("A note is required when the rejection reason is 'other'.")
        return self


class RejectionRead(BaseModel):
    """A rejection as returned to the client."""

    id: int
    mask_id: int
    contour_id: int | None
    reason: RejectionReason
    reason_label: str
    note: str | None
    created_by: str
    created_at: datetime
    resolved_at: datetime | None
    resolved_by: str | None

    @property
    def is_open(self) -> bool:
        return self.resolved_at is None


class RejectionReasonOption(BaseModel):
    """One selectable option for the reviewer's reason dropdown."""

    value: RejectionReason
    label: str
    requires_note: bool


class MemberRead(BaseModel):
    """A dataset membership as returned to the client."""

    username: str
    role: DatasetRole
    extra_permissions: list[Permission] = Field(default_factory=list)
    denied_permissions: list[Permission] = Field(default_factory=list)
    granted_by: str | None = None
    granted_at: datetime | None = None


class MemberGrant(BaseModel):
    """Request body for granting or changing a member's dataset role."""

    username: str = Field(..., description="Account to grant the role to.")
    role: DatasetRole = Field(..., description="Role to grant.")
    extra_permissions: list[Permission] = Field(
        default_factory=list,
        description="Permissions granted on top of the role bundle.",
    )
    denied_permissions: list[Permission] = Field(
        default_factory=list,
        description="Permissions withheld even though the role bundle includes them.",
    )


class InviteCreate(BaseModel):
    """Request body for minting a dataset invite link."""

    role: DatasetRole = Field(
        DatasetRole.ANNOTATOR,
        description="Role granted on redemption. Owner cannot be invited to; "
                    "ownership transfer is always an explicit action.",
    )
    expires_in_hours: int | None = Field(
        168,
        ge=1,
        le=24 * 365,
        description="Lifetime of the link in hours. None means it never expires.",
    )
    max_uses: int | None = Field(
        None,
        ge=1,
        description="How many accounts may redeem the link. None means unlimited.",
    )

    @model_validator(mode="after")
    def _reject_owner_invites(self) -> "InviteCreate":
        if self.role is DatasetRole.OWNER:
            raise ValueError(
                "Invite links cannot grant ownership. Use the ownership transfer endpoint."
            )
        return self


class InviteRead(BaseModel):
    """An invite as returned to its creator. The raw token is never included here."""

    id: int
    dataset_id: int
    role: DatasetRole
    created_by: str
    created_at: datetime
    expires_at: datetime | None
    max_uses: int | None
    uses: int
    revoked_at: datetime | None
    is_valid: bool


class InvitePreview(BaseModel):
    """What an invitee sees before deciding to accept."""

    dataset_id: int
    dataset_name: str
    dataset_description: str | None
    role: DatasetRole
    invited_by: str
    expires_at: datetime | None
    is_valid: bool
    already_member: bool
    current_role: DatasetRole | None = None


class GlobalRoleUpdate(BaseModel):
    """Request body for changing an account's platform-level role."""

    global_role: GlobalRole

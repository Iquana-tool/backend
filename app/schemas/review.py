"""Schemas for the review step: rejections, their reasons and membership payloads.

A rejection is how a reviewer sends work back to the annotator. It carries a
machine-readable `reason` from a fixed vocabulary plus an optional free-text
`note`, so a later review pipeline can aggregate "how often is this annotator
told the outline is bad?" without parsing prose.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.permissions import DatasetRole, GlobalRole, Permission


class RejectionReason(StrEnum):
    """Predefined reasons a reviewer can send an annotation back.

    `OTHER` requires a note; every other reason may carry one.
    """

    BAD_OUTLINE = "bad_outline"
    WRONG_LABEL = "wrong_label"
    MISSING_OBJECTS = "missing_objects"
    #: One outline covers several real objects and has to be split.
    MERGED_OBJECTS = "merged_objects"
    #: The outline covers only part of its object -- typically because another
    #: object overlaps it and the rest was never traced. Distinct from
    #: BAD_OUTLINE (which is about accuracy) and MISSING_OBJECTS (whole objects
    #: absent from the image), because the fix is "extend this outline".
    MISSING_PARTS = "missing_parts"
    EXTRA_OBJECTS = "extra_objects"
    DUPLICATE_OBJECT = "duplicate_object"
    WRONG_HIERARCHY = "wrong_hierarchy"
    IMAGE_QUALITY = "image_quality"
    OTHER = "other"


class RejectionResolution(StrEnum):
    """How a rejection was closed.

    Recorded when the annotator works through the correction queue: ``FIXED`` means
    the annotation was reworked, ``WONT_FIX`` means it was looked at and left as is
    ("I checked, it is correct"). NULL on rows resolved before this vocabulary
    existed, so callers must treat a missing resolution as "unspecified".
    """

    FIXED = "fixed"
    WONT_FIX = "wont_fix"


#: Human-readable labels for the frontend, so the wording lives with the enum.
REJECTION_REASON_LABELS: dict[RejectionReason, str] = {
    RejectionReason.BAD_OUTLINE: "Outline is inaccurate",
    RejectionReason.WRONG_LABEL: "Wrong label assigned",
    RejectionReason.MISSING_OBJECTS: "Objects are missing",
    RejectionReason.MERGED_OBJECTS: "Merged objects — should be several",
    RejectionReason.MISSING_PARTS: "Only part of the object is outlined",
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


class RejectionResolve(BaseModel):
    """Request body for resolving a rejection.

    The body is optional on the wire (an empty PATCH still resolves); ``resolution``
    records *how* it was closed when the caller knows, e.g. the correction queue's
    "Mark as done" (``fixed``) versus "Won't fix" (``wont_fix``).
    """

    resolution: RejectionResolution | None = Field(
        None,
        description="How the rejection was closed. Omit for an unspecified resolve.",
    )


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
    resolution: RejectionResolution | None = None

    @property
    def is_open(self) -> bool:
        return self.resolved_at is None


class RejectionReasonOption(BaseModel):
    """One selectable option for the reviewer's reason dropdown."""

    value: RejectionReason
    label: str
    requires_note: bool


# -- Review queue -----------------------------------------------------------
#
# The queue is how a reviewer works through pending annotations: they pick a
# granularity (whole images, or one instance at a time) and an ordering, and the
# backend returns the work items in that order. Ordering is delegated to a
# strategy registry (see `app.services.review_queue`) so that active-learning
# scorers (uncertainty, disagreement, ...) can be added without touching the
# request/response contract: a strategy only has to map each candidate to a score.


class ReviewGranularity(StrEnum):
    """What one queue item spans."""

    #: One item per image: the whole mask with all its contours at once.
    IMAGES = "images"
    #: One item per instance, ordered by the contour hierarchy (roots first).
    HIERARCHY = "hierarchy"
    #: One item per instance, restricted to a chosen set of labels.
    CUSTOM = "custom"


class ReviewSortDirection(StrEnum):
    ASCENDING = "asc"
    DESCENDING = "desc"


class ReviewQueueRequest(BaseModel):
    """Request body for building a review queue."""

    granularity: ReviewGranularity = Field(..., description="What one queue item spans.")
    sort_strategy: str = Field(
        "hierarchy",
        description="Key of the scoring strategy that orders instance items. "
                    "The available keys are listed in the review summary.",
    )
    direction: ReviewSortDirection = Field(
        ReviewSortDirection.ASCENDING,
        description="Ascending replays the strategy's natural order "
                    "(for 'hierarchy': root instances first).",
    )
    label_ids: list[int] | None = Field(
        None,
        description="Only queue instances carrying one of these labels. "
                    "Required for 'custom' granularity, ignored otherwise.",
    )
    only_submitted: bool = Field(
        True,
        description="Only queue work from masks submitted for review "
                    "(fully annotated, no open rejections). Turn off to also "
                    "sweep through work still in progress.",
    )
    include_reviewed: bool = Field(
        False,
        description="Also queue instances that already carry an approval — the "
                    "caller's own included, so a solo reviewer can re-sweep their "
                    "work. Re-accepting is a no-op; rejecting withdraws the "
                    "caller's earlier approval of that instance.",
    )

    @model_validator(mode="after")
    def _require_labels_for_custom(self) -> "ReviewQueueRequest":
        if self.granularity is ReviewGranularity.CUSTOM and not self.label_ids:
            raise ValueError("Custom review needs at least one label to filter by.")
        return self


class ReviewQueueImageItem(BaseModel):
    """One whole-image work item."""

    image_id: int
    mask_id: int
    pending_instances: int = Field(..., description="Contours nobody has approved yet.")
    total_instances: int


class ReviewQueueInstanceItem(BaseModel):
    """One single-instance work item.

    Geometry is deliberately not included — the player fetches the mask's contours
    once (they carry the SVG paths) and picks this instance plus its immediate
    children out of that list via `parent_id`.
    """

    contour_id: int
    mask_id: int
    image_id: int
    label_id: int | None
    parent_id: int | None
    depth: int = Field(..., description="0 for root instances.")
    score: float = Field(..., description="The sort strategy's score for this instance.")


class ReviewQueueRead(BaseModel):
    """A freshly built review queue.

    Exactly one of `images` / `instances` is populated, depending on granularity.
    The queue is a snapshot: it is not persisted server-side, so a stale entry
    (e.g. approved by someone else meanwhile) simply no-ops during the session.
    """

    granularity: ReviewGranularity
    sort_strategy: str
    direction: ReviewSortDirection
    # Echoed back so the session can keep acting under the same rules it was
    # built with (e.g. the image-level Accept must match the queue's eligibility).
    include_reviewed: bool = False
    total: int
    images: list[ReviewQueueImageItem] = Field(default_factory=list)
    instances: list[ReviewQueueInstanceItem] = Field(default_factory=list)


class ReviewSortStrategyOption(BaseModel):
    """One selectable queue ordering, for the setup page's dropdown."""

    key: str
    label: str
    description: str


class ReviewSummary(BaseModel):
    """The dataset's review workload, for the management card and setup page."""

    pending_instances: int = Field(..., description="Unapproved contours on submitted masks.")
    pending_images: int = Field(..., description="Images with at least one pending instance.")
    reviewed_instances: int = Field(
        0,
        description="Already-approved contours on submitted masks — re-reviewable "
                    "via the queue's include_reviewed option.",
    )
    open_rejections: int = Field(..., description="Sent-back items the annotators have not addressed yet.")
    strategies: list[ReviewSortStrategyOption] = Field(default_factory=list)


# -- Correction queue -------------------------------------------------------
#
# The mirror image of the review queue: where the review queue serves *pending*
# annotations to a reviewer, the correction queue serves *sent-back* annotations
# (open rejections) to an annotator, who works through them in the editor. Each
# item points at the mask/contour to load and carries the reviewer's reason and
# note so the annotator knows what to fix. Resolving an item (fixed / won't fix)
# closes the rejection; the queue is a snapshot, so an item someone else resolved
# meanwhile simply no-ops when acted on — the same contract as the review queue.


class CorrectionSortOrder(StrEnum):
    """The order corrections are served in. There is no scoring registry: unlike
    review, corrections are a flat worklist, walked front to back by age."""

    OLDEST = "oldest"
    NEWEST = "newest"


class CorrectionQueueRequest(BaseModel):
    """Request body for building a correction queue."""

    order: CorrectionSortOrder = Field(
        CorrectionSortOrder.OLDEST,
        description="Oldest feedback first (clear the backlog) or newest first.",
    )
    reasons: list[RejectionReason] | None = Field(
        None,
        description="Only queue rejections carrying one of these reasons. "
                    "Omit (or empty) to queue every open rejection.",
    )


class CorrectionQueueItem(BaseModel):
    """One sent-back item to correct.

    ``contour_id`` is null for a mask-level rejection (e.g. "objects are missing"),
    which the editor handles by loading the image without pre-selecting an object.
    """

    rejection_id: int
    mask_id: int
    image_id: int
    contour_id: int | None
    reason: RejectionReason
    reason_label: str
    note: str | None
    created_by: str | None
    created_at: datetime


class CorrectionQueueRead(BaseModel):
    """A freshly built correction queue.

    Items on the same image are consecutive so the editor's per-image session and
    caches carry across them, matching the review queue's grouping. The queue is a
    snapshot: it is not persisted, so a stale entry (resolved by someone else
    meanwhile) simply no-ops when acted on.
    """

    order: CorrectionSortOrder
    total: int
    items: list[CorrectionQueueItem] = Field(default_factory=list)


class CorrectionSummary(BaseModel):
    """The dataset's correction workload, for the management card and launch page."""

    open_rejections: int = Field(..., description="Sent-back items not yet resolved.")
    affected_instances: int = Field(
        ..., description="Distinct contours with an open rejection (mask-level ones excluded)."
    )
    affected_images: int = Field(..., description="Images with at least one open rejection.")


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


class AdminUserCreate(BaseModel):
    """Request body for creating an account from the admin surface.

    The admin picks the initial password and hands it over out of band: iquana
    sends no mail, so there is nowhere to deliver an activation or reset link to.
    """

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=8, max_length=128)
    global_role: GlobalRole = GlobalRole.MEMBER
    is_active: bool = True

    @field_validator("username", mode="before")
    @classmethod
    def _strip_username(cls, value):
        """Trim surrounding whitespace so a pasted name still matches at login."""
        return value.strip() if isinstance(value, str) else value


class SettingsUpdate(BaseModel):
    """Request body for changing the deployment's own configuration.

    A sparse map of ``{setting key: new value}`` -- only the fields the operator
    actually edited are sent, so two admins saving different tabs cannot clobber
    each other's work. An unknown key is refused rather than ignored, so a typo
    does not look like a saved setting.

    Values arrive as strings (or ``null``) regardless of the setting's declared
    kind; the settings service parses each one against its own spec.
    """

    values: dict[str, str | None] = Field(default_factory=dict)

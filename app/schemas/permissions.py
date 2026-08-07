"""Role and permission vocabulary for iquana.

Access control has two independent levels:

* **Global role** (`GlobalRole`, stored on `users.global_role`) answers *"what may
  this account do to the platform?"* — create datasets at all, manage other users,
  touch the model registry.
* **Dataset role** (`DatasetRole`, stored on `dataset_members.role`) answers *"what
  may this account do to **this** dataset?"* Datasets are iquana's unit of
  governance, so this is where nearly every permission lives.

Roles are only named bundles of `Permission` values; the bundles are defined by
`DATASET_ROLE_PERMISSIONS` / `GLOBAL_ROLE_PERMISSIONS` below. A membership row may
additionally carry `extra_permissions` / `denied_permissions` so a single
collaborator can be given (or refused) one capability without inventing a new role.

This lives in the backend rather than in `iquana-toolbox` on purpose: the toolbox
is consumed as a git-pinned dependency, so putting the matrix here keeps
permission changes from requiring a toolbox release + re-pin.
"""
from __future__ import annotations

from enum import StrEnum


class Permission(StrEnum):
    """A single capability that can be checked against a user (and a dataset)."""

    # -- Dataset lifecycle -------------------------------------------------
    DATASET_READ = "dataset.read"
    DATASET_UPDATE = "dataset.update"
    DATASET_DELETE = "dataset.delete"
    DATASET_TRANSFER_OWNERSHIP = "dataset.transfer_ownership"

    # -- Membership --------------------------------------------------------
    MEMBER_LIST = "member.list"
    MEMBER_GRANT = "member.grant"
    MEMBER_REVOKE = "member.revoke"
    INVITE_CREATE = "invite.create"
    INVITE_REVOKE = "invite.revoke"

    # -- Image data --------------------------------------------------------
    IMAGE_READ = "image.read"
    IMAGE_UPLOAD = "image.upload"
    IMAGE_DELETE = "image.delete"
    # Changing the pixel scale silently rescales every quantification number in
    # the dataset, which is why it is not bundled with ordinary annotation work.
    PIXEL_SCALE_SET = "pixel_scale.set"
    # The same argument for every other calibration kind (intensity, colour, ...):
    # they change what the stored measurements mean, not what is annotated. Scale
    # keeps its own older permission so an existing grant of it is not widened.
    CALIBRATION_SET = "calibration.set"

    # -- Label space -------------------------------------------------------
    LABEL_READ = "label.read"
    # Deleting a label cascades into every contour carrying it.
    LABEL_MANAGE = "label.manage"

    # -- Annotation --------------------------------------------------------
    ANNOTATION_READ = "annotation.read"
    ANNOTATION_CREATE = "annotation.create"
    ANNOTATION_EDIT_OWN = "annotation.edit_own"
    ANNOTATION_DELETE_OWN = "annotation.delete_own"
    ANNOTATION_EDIT_ANY = "annotation.edit_any"
    MASK_SUBMIT = "mask.submit"
    MASK_REOPEN = "mask.reopen"
    MASK_DELETE = "mask.delete"

    # -- Review ------------------------------------------------------------
    REVIEW_APPROVE = "review.approve"
    REVIEW_REJECT = "review.reject"
    REVIEW_REVOKE = "review.revoke"
    REVIEW_PURGE_UNREVIEWED = "review.purge_unreviewed"

    # -- AI assistance -----------------------------------------------------
    AI_INTERACTIVE = "ai.interactive"
    AI_BATCH_INFER = "ai.batch_infer"
    AI_TRAIN = "ai.train"

    # -- Export ------------------------------------------------------------
    # Split three ways so collaborators can pull measurements without pulling
    # the raw imagery off the platform.
    EXPORT_ANNOTATIONS = "export.annotations"
    EXPORT_IMAGES = "export.images"
    EXPORT_QUANTIFICATION = "export.quantification"

    # -- Global (never dataset-scoped) -------------------------------------
    DATASET_CREATE = "dataset.create"
    USER_MANAGE = "user.manage"
    USER_SET_GLOBAL_ROLE = "user.set_global_role"
    SYSTEM_MANAGE_MODELS = "system.manage_models"


#: Permissions that are meaningless per dataset and are answered by the global role.
GLOBAL_PERMISSIONS: frozenset[Permission] = frozenset({
    Permission.DATASET_CREATE,
    Permission.USER_MANAGE,
    Permission.USER_SET_GLOBAL_ROLE,
    Permission.SYSTEM_MANAGE_MODELS,
})


class GlobalRole(StrEnum):
    """Platform-level role, stored on the user account."""

    ADMIN = "admin"
    MEMBER = "member"
    GUEST = "guest"


class DatasetRole(StrEnum):
    """Per-dataset role, stored on the membership row.

    Ordered from least to most privileged; see `DATASET_ROLE_ORDER`.
    """

    VIEWER = "viewer"
    ANNOTATOR = "annotator"
    REVIEWER = "reviewer"
    CURATOR = "curator"
    OWNER = "owner"


#: Privilege ranking, used to avoid downgrading a member when an invite is redeemed.
DATASET_ROLE_ORDER: dict[DatasetRole, int] = {
    DatasetRole.VIEWER: 0,
    DatasetRole.ANNOTATOR: 1,
    DatasetRole.REVIEWER: 2,
    DatasetRole.CURATOR: 3,
    DatasetRole.OWNER: 4,
}


_VIEWER: frozenset[Permission] = frozenset({
    Permission.DATASET_READ,
    Permission.IMAGE_READ,
    Permission.LABEL_READ,
    Permission.ANNOTATION_READ,
})

_ANNOTATOR: frozenset[Permission] = _VIEWER | {
    Permission.ANNOTATION_CREATE,
    Permission.ANNOTATION_EDIT_OWN,
    Permission.ANNOTATION_DELETE_OWN,
    Permission.MASK_SUBMIT,
    Permission.AI_INTERACTIVE,
}

_REVIEWER: frozenset[Permission] = _ANNOTATOR | {
    Permission.ANNOTATION_EDIT_ANY,
    Permission.MASK_REOPEN,
    Permission.MASK_DELETE,
    Permission.REVIEW_APPROVE,
    Permission.REVIEW_REJECT,
    Permission.REVIEW_REVOKE,
    Permission.REVIEW_PURGE_UNREVIEWED,
    Permission.EXPORT_ANNOTATIONS,
    Permission.EXPORT_QUANTIFICATION,
}

_CURATOR: frozenset[Permission] = _REVIEWER | {
    Permission.DATASET_UPDATE,
    Permission.IMAGE_UPLOAD,
    Permission.IMAGE_DELETE,
    Permission.PIXEL_SCALE_SET,
    Permission.CALIBRATION_SET,
    Permission.LABEL_MANAGE,
    Permission.AI_BATCH_INFER,
    Permission.AI_TRAIN,
    Permission.EXPORT_IMAGES,
    Permission.MEMBER_LIST,
    Permission.INVITE_CREATE,
}

_OWNER: frozenset[Permission] = _CURATOR | {
    Permission.DATASET_DELETE,
    Permission.DATASET_TRANSFER_OWNERSHIP,
    Permission.MEMBER_GRANT,
    Permission.MEMBER_REVOKE,
    Permission.INVITE_REVOKE,
}

DATASET_ROLE_PERMISSIONS: dict[DatasetRole, frozenset[Permission]] = {
    DatasetRole.VIEWER: _VIEWER,
    DatasetRole.ANNOTATOR: _ANNOTATOR,
    DatasetRole.REVIEWER: _REVIEWER,
    DatasetRole.CURATOR: _CURATOR,
    DatasetRole.OWNER: _OWNER,
}

GLOBAL_ROLE_PERMISSIONS: dict[GlobalRole, frozenset[Permission]] = {
    # Admin is handled by an explicit bypass in the permission service rather than
    # by enumeration, so this set only needs the genuinely global permissions.
    GlobalRole.ADMIN: frozenset(GLOBAL_PERMISSIONS),
    GlobalRole.MEMBER: frozenset({Permission.DATASET_CREATE}),
    # A guest can only work inside datasets they were invited to.
    GlobalRole.GUEST: frozenset(),
}


def permissions_for_dataset_role(role: DatasetRole | str) -> frozenset[Permission]:
    """Return the permission bundle for a dataset role, empty for unknown roles."""
    try:
        return DATASET_ROLE_PERMISSIONS[DatasetRole(role)]
    except ValueError:
        return frozenset()


def permissions_for_global_role(role: GlobalRole | str) -> frozenset[Permission]:
    """Return the platform-level permission bundle for a global role."""
    try:
        return GLOBAL_ROLE_PERMISSIONS[GlobalRole(role)]
    except ValueError:
        return frozenset()


def is_at_least(role: DatasetRole | str, minimum: DatasetRole) -> bool:
    """Whether `role` ranks at or above `minimum` in the dataset role hierarchy."""
    try:
        return DATASET_ROLE_ORDER[DatasetRole(role)] >= DATASET_ROLE_ORDER[minimum]
    except ValueError:
        return False

"""The authenticated caller, enriched with everything needed to answer
permission questions without a database round trip per check.

This sits in `schemas` rather than in `services.permissions` so that
`services.auth` (which builds it) and `services.permissions` (which enforces with
it) can both import it without forming a cycle.
"""
from __future__ import annotations

from logging import getLogger
from typing import Iterable

from iquana_toolbox.schemas.user import User
from pydantic import BaseModel, Field

from app.schemas.permissions import (
    DatasetRole,
    GLOBAL_PERMISSIONS,
    GlobalRole,
    Permission,
    permissions_for_dataset_role,
    permissions_for_global_role,
)

logger = getLogger(__name__)


def coerce_permissions(raw: Iterable[str] | None) -> frozenset[Permission]:
    """Turn a stored JSON list into permissions, dropping values we no longer know.

    Unknown entries are ignored rather than fatal so that removing a permission
    from the enum does not break logins for users who still carry it.
    """
    resolved: set[Permission] = set()
    for value in raw or ():
        try:
            resolved.add(Permission(value))
        except ValueError:
            logger.warning("Ignoring unknown permission %r stored on a membership row.", value)
    return frozenset(resolved)


def effective_permissions(role: DatasetRole,
                          extra: Iterable[str] | None = None,
                          denied: Iterable[str] | None = None) -> frozenset[Permission]:
    """Role bundle, plus per-member grants, minus per-member denials.

    Denials win, so a permission listed in both cannot be re-granted.
    """
    return (permissions_for_dataset_role(role) | coerce_permissions(extra)) - coerce_permissions(denied)


class MembershipInfo(BaseModel):
    """A resolved membership: the role plus its expanded permission set."""

    role: DatasetRole
    permissions: frozenset[Permission] = Field(default_factory=frozenset)


class AuthenticatedUser(User):
    """The current caller.

    Subclasses the toolbox `User` so every route already typed as `User` keeps
    working unchanged; `owned_datasets` / `accessible_datasets` / `is_admin` are
    populated with the same meaning they had before roles existed.
    """

    global_role: GlobalRole = Field(GlobalRole.MEMBER, description="Platform-level role.")
    is_active: bool = Field(True, description="Whether the account is enabled.")
    memberships: dict[int, MembershipInfo] = Field(
        default_factory=dict,
        description="Dataset id -> the caller's role and permissions on it.",
    )

    @classmethod
    def from_query(cls, user_db) -> "AuthenticatedUser":
        """Build the caller from a `Users` row and its membership rows."""
        memberships: dict[int, MembershipInfo] = {}
        for membership in user_db.memberships:
            try:
                role = DatasetRole(membership.role)
            except ValueError:
                logger.warning("Unknown dataset role %r on dataset %s; treating as viewer.",
                               membership.role, membership.dataset_id)
                role = DatasetRole.VIEWER
            memberships[membership.dataset_id] = MembershipInfo(
                role=role,
                permissions=effective_permissions(
                    role, membership.extra_permissions, membership.denied_permissions
                ),
            )

        # Legacy safety net: datasets created before ownership became a membership
        # row have no grant to fall back on, so honour `created_by` as ownership.
        for dataset in user_db.owned_datasets:
            if dataset.id not in memberships:
                memberships[dataset.id] = MembershipInfo(
                    role=DatasetRole.OWNER,
                    permissions=permissions_for_dataset_role(DatasetRole.OWNER),
                )

        try:
            global_role = GlobalRole(user_db.global_role)
        except ValueError:
            logger.warning("Unknown global role %r for %s; treating as guest.",
                           user_db.global_role, user_db.username)
            global_role = GlobalRole.GUEST

        owned = [ds_id for ds_id, info in memberships.items() if info.role is DatasetRole.OWNER]
        accessible = [ds_id for ds_id in memberships if ds_id not in owned]

        return cls(
            username=user_db.username,
            is_admin=global_role is GlobalRole.ADMIN,
            global_role=global_role,
            is_active=bool(getattr(user_db, "is_active", True)),
            owned_datasets=owned,
            accessible_datasets=accessible,
            memberships=memberships,
        )

    def role_for(self, dataset_id: int) -> DatasetRole | None:
        """The caller's role on a dataset, or None if they are not a member."""
        membership = self.memberships.get(dataset_id)
        return membership.role if membership else None

    def permissions_for(self, dataset_id: int) -> frozenset[Permission]:
        """Every dataset-scoped permission the caller holds on one dataset."""
        if self.global_role is GlobalRole.ADMIN:
            return frozenset(set(Permission) - GLOBAL_PERMISSIONS)
        membership = self.memberships.get(dataset_id)
        return membership.permissions if membership else frozenset()

    def has_permission(self, dataset_id: int, permission: Permission) -> bool:
        """Whether the caller may perform `permission` on `dataset_id`."""
        if not self.is_active:
            return False
        if permission in GLOBAL_PERMISSIONS:
            return self.has_global_permission(permission)
        return permission in self.permissions_for(dataset_id)

    def has_global_permission(self, permission: Permission) -> bool:
        """Whether the caller holds a platform-level permission."""
        if not self.is_active:
            return False
        if self.global_role is GlobalRole.ADMIN:
            return True
        return permission in permissions_for_global_role(self.global_role)

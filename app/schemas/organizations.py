"""Request bodies for organisations, teams and team grants on datasets."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.schemas.permissions import (
    DATASET_ROLE_ORDER,
    TEAM_GRANT_MAX_ROLE,
    DatasetRole,
    OrganizationRole,
    Permission,
    TeamRole,
)


def _strip(value):
    return value.strip() if isinstance(value, str) else value


def _strip_or_none(value):
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


class OrganizationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)

    _strip_name = field_validator("name", mode="before")(_strip)


class OrganizationUpdate(BaseModel):
    """Only the fields sent change. ``is_default`` needs ``organization.manage``."""

    name: str | None = Field(None, min_length=1, max_length=100)
    is_default: bool | None = None
    allow_personal_keys: bool | None = Field(
        None, description="Whether members' personal API keys are used for the organisation's work.")

    _strip_name = field_validator("name", mode="before")(_strip)


class OrganizationMemberSet(BaseModel):
    role: OrganizationRole = OrganizationRole.MEMBER


class TeamCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(None, max_length=255)
    parent_team_id: int | None = Field(None, description="The team this one sits inside.")

    _strip_name = field_validator("name", mode="before")(_strip)
    _strip_description = field_validator("description", mode="before")(_strip_or_none)


class TeamUpdate(BaseModel):
    """Only the fields sent change; ``parent_team_id`` sent as null moves the team to the top."""

    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = Field(None, max_length=255)
    parent_team_id: int | None = None

    _strip_name = field_validator("name", mode="before")(_strip)
    _strip_description = field_validator("description", mode="before")(_strip_or_none)


class TeamMemberSet(BaseModel):
    role: TeamRole = TeamRole.MEMBER


class TeamGrant(BaseModel):
    """A team's role on a dataset, with the same per-grant exceptions as a member's."""

    role: DatasetRole
    extra_permissions: list[Permission] = Field(default_factory=list)
    denied_permissions: list[Permission] = Field(default_factory=list)

    @field_validator("role")
    @classmethod
    def _at_most_curator(cls, role: DatasetRole) -> DatasetRole:
        if DATASET_ROLE_ORDER[role] > DATASET_ROLE_ORDER[TEAM_GRANT_MAX_ROLE]:
            raise ValueError(f"A team can be at most {TEAM_GRANT_MAX_ROLE.value}; "
                             "the owner is always one person.")
        return role


class DatasetOrganizationSet(BaseModel):
    organization_id: int | None = Field(..., description="None makes the dataset personal.")

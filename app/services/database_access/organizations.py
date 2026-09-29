"""Organisations, teams and team grants on datasets.

The authorisation for these lives here as ``ensure_*`` helpers rather than in
``app.services.permissions``: it turns on organisation and team roles, not on the
dataset roles that module is built around. A holder of ``organization.manage``
(platform admins) passes every check.
"""
from logging import getLogger

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.dataset_members import DatasetMembers
from app.database.datasets import Datasets
from app.database.organizations import (
    DatasetTeamGrants,
    OrganizationMembers,
    Organizations,
    TeamMembers,
    Teams,
)
from app.database.users import Users, utc_now
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.organizations import OrganizationCreate, OrganizationUpdate, TeamCreate, TeamGrant, TeamUpdate
from app.schemas.permissions import DatasetRole, OrganizationRole, Permission, TeamRole
from app.services.database_access import members as members_db

logger = getLogger(__name__)


def _not_found(what: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{what} not found.")


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


# -- Authorisation ---------------------------------------------------------------

def manages_organizations(user: AuthenticatedUser) -> bool:
    return user.has_global_permission(Permission.ORGANIZATION_MANAGE)


def ensure_organization_member(user: AuthenticatedUser, organization_id: int) -> None:
    if manages_organizations(user) or organization_id in user.organizations:
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                        detail="You are not a member of this organisation.")


def ensure_organization_admin(user: AuthenticatedUser, organization_id: int) -> None:
    if manages_organizations(user) or user.organizations.get(organization_id) is OrganizationRole.ADMIN:
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                        detail="Only the organisation's admins can do this.")


def ensure_team_manager(user: AuthenticatedUser, team: Teams, db: Session) -> None:
    """Organisation admins manage every team; a maintainer, their own team's members."""
    if manages_organizations(user) or user.organizations.get(team.organization_id) is OrganizationRole.ADMIN:
        return
    role = db.query(TeamMembers.role).filter_by(team_id=team.id, username=user.username).scalar()
    if role == TeamRole.MAINTAINER.value:
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                        detail="Only the team's maintainers and the organisation's admins can do this.")


# -- Organisations ---------------------------------------------------------------

def get_organization(organization_id: int, db: Session) -> Organizations:
    organization = db.get(Organizations, organization_id)
    if organization is None:
        raise _not_found("Organisation")
    return organization


def list_organizations(user: AuthenticatedUser, db: Session) -> list[dict]:
    """The caller's organisations; every organisation for those who manage them."""
    query = db.query(Organizations).order_by(Organizations.name)
    if not manages_organizations(user):
        query = query.filter(Organizations.id.in_(list(user.organizations)))
    member_counts = dict(db.query(OrganizationMembers.organization_id, func.count())
                         .group_by(OrganizationMembers.organization_id))
    team_counts = dict(db.query(Teams.organization_id, func.count()).group_by(Teams.organization_id))
    return [
        {
            "id": organization.id,
            "name": organization.name,
            "is_default": bool(organization.is_default),
            "created_at": _iso(organization.created_at),
            "my_role": user.organizations[organization.id].value
            if organization.id in user.organizations else None,
            "member_count": member_counts.get(organization.id, 0),
            "team_count": team_counts.get(organization.id, 0),
        }
        for organization in query
    ]


def _commit_or_conflict(db: Session, detail: str) -> None:
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


def create_organization(body: OrganizationCreate, creator: str, db: Session) -> Organizations:
    """Create an organisation with its creator as its first admin."""
    organization = Organizations(name=body.name)
    db.add(organization)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="An organisation with that name already exists.")
    db.add(OrganizationMembers(organization_id=organization.id, username=creator,
                               role=OrganizationRole.ADMIN.value))
    db.commit()
    return organization


def update_organization(organization: Organizations, body: OrganizationUpdate,
                        user: AuthenticatedUser, db: Session) -> None:
    sent = body.model_fields_set
    if "name" in sent and body.name is not None:
        ensure_organization_admin(user, organization.id)
        organization.name = body.name
    if "is_default" in sent and body.is_default is not None:
        if not manages_organizations(user):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="Only platform admins choose the default organisation.")
        if body.is_default:
            # Cleared first, and flushed, so the one-default index never sees two.
            (db.query(Organizations).filter(Organizations.id != organization.id)
             .update({Organizations.is_default: False}))
            db.flush()
        organization.is_default = body.is_default
    _commit_or_conflict(db, "An organisation with that name already exists.")


def delete_organization(organization: Organizations, db: Session) -> None:
    """Delete an empty-of-datasets organisation, with its memberships and teams."""
    if db.query(Datasets.id).filter_by(organization_id=organization.id).first() is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="The organisation still has datasets. Move or delete them first.")
    db.delete(organization)
    db.commit()


# -- Organisation members --------------------------------------------------------

def list_organization_members(organization_id: int, db: Session) -> list[dict]:
    rows = (db.query(OrganizationMembers, Users.display_name)
            .join(Users, Users.username == OrganizationMembers.username)
            .filter(OrganizationMembers.organization_id == organization_id)
            .all())
    rows.sort(key=lambda row: (row[0].role != OrganizationRole.ADMIN.value, row[0].username))
    return [
        {"username": member.username, "display_name": display_name,
         "role": member.role, "joined_at": _iso(member.joined_at)}
        for member, display_name in rows
    ]


def _admin_count(organization_id: int, db: Session) -> int:
    return (db.query(func.count()).select_from(OrganizationMembers)
            .filter_by(organization_id=organization_id, role=OrganizationRole.ADMIN.value)
            .scalar())


def _refuse_losing_last_admin(member: OrganizationMembers, db: Session) -> None:
    if member.role == OrganizationRole.ADMIN.value and _admin_count(member.organization_id, db) == 1:
        raise _bad_request("This is the organisation's last admin. Make someone else an admin first.")


def set_organization_member(organization_id: int, username: str, role: OrganizationRole,
                            db: Session) -> OrganizationMembers:
    """Add an account to the organisation, or change its role there."""
    if db.query(Users.username).filter_by(username=username).scalar() is None:
        raise _not_found(f"User '{username}'")
    member = db.get(OrganizationMembers, (organization_id, username))
    if member is None:
        member = OrganizationMembers(organization_id=organization_id, username=username)
        db.add(member)
    elif role is not OrganizationRole.ADMIN:
        _refuse_losing_last_admin(member, db)
    member.role = role.value
    db.commit()
    return member


def remove_organization_member(organization_id: int, username: str, db: Session) -> None:
    """Remove an account from the organisation, and from every team in it.

    A team is part of its organisation, so nobody outside the organisation stays in
    one -- and with the team memberships go the dataset access they carried.
    """
    member = db.get(OrganizationMembers, (organization_id, username))
    if member is None:
        raise _not_found(f"{username} in this organisation")
    _refuse_losing_last_admin(member, db)
    team_ids = db.query(Teams.id).filter_by(organization_id=organization_id)
    (db.query(TeamMembers)
     .filter(TeamMembers.username == username, TeamMembers.team_id.in_(team_ids))
     .delete(synchronize_session=False))
    db.delete(member)
    db.commit()


def join_default_organization(username: str, db: Session) -> None:
    """Make a new account a member of the default organisation, if there is one.

    Flushes, so the account row is written before the membership that points at
    it, but does not commit: the caller owns the transaction.
    """
    default_id = db.query(Organizations.id).filter_by(is_default=True).scalar()
    if default_id is None:
        return
    db.flush()
    db.add(OrganizationMembers(organization_id=default_id, username=username,
                               role=OrganizationRole.MEMBER.value))


# -- Teams -----------------------------------------------------------------------

def get_team(team_id: int, db: Session) -> Teams:
    team = db.get(Teams, team_id)
    if team is None:
        raise _not_found("Team")
    return team


def list_teams(organization_id: int, db: Session) -> list[dict]:
    """Every team in the organisation, flat; ``parent_team_id`` gives the tree."""
    member_counts = dict(db.query(TeamMembers.team_id, func.count()).group_by(TeamMembers.team_id))
    return [
        {
            "id": team.id,
            "name": team.name,
            "description": team.description,
            "parent_team_id": team.parent_team_id,
            "member_count": member_counts.get(team.id, 0),
            "created_at": _iso(team.created_at),
        }
        for team in db.query(Teams).filter_by(organization_id=organization_id).order_by(Teams.name)
    ]


def _check_parent(organization_id: int, team_id: int | None, parent_id: int | None, db: Session) -> None:
    """A parent must be in the same organisation, and must not sit below the team itself."""
    if parent_id is None:
        return
    parent = db.get(Teams, parent_id)
    if parent is None or parent.organization_id != organization_id:
        raise _bad_request("The parent team must be in the same organisation.")
    seen: set[int] = set()
    current = parent
    while current is not None and current.id not in seen:
        if current.id == team_id:
            raise _bad_request("A team cannot sit inside itself or one of its own sub-teams.")
        seen.add(current.id)
        current = db.get(Teams, current.parent_team_id) if current.parent_team_id else None


def create_team(organization_id: int, body: TeamCreate, db: Session) -> Teams:
    _check_parent(organization_id, None, body.parent_team_id, db)
    team = Teams(organization_id=organization_id, name=body.name,
                 description=body.description, parent_team_id=body.parent_team_id)
    db.add(team)
    _commit_or_conflict(db, "The organisation already has a team with that name.")
    return team


def update_team(team: Teams, body: TeamUpdate, db: Session) -> None:
    sent = body.model_fields_set
    if "name" in sent and body.name is not None:
        team.name = body.name
    if "description" in sent:
        team.description = body.description
    if "parent_team_id" in sent:
        _check_parent(team.organization_id, team.id, body.parent_team_id, db)
        team.parent_team_id = body.parent_team_id
    _commit_or_conflict(db, "The organisation already has a team with that name.")


def delete_team(team: Teams, db: Session) -> None:
    """Delete a team with its memberships and grants; its sub-teams move to the top."""
    db.delete(team)
    db.commit()


# -- Team members ----------------------------------------------------------------

def list_team_members(team_id: int, db: Session) -> list[dict]:
    rows = (db.query(TeamMembers, Users.display_name)
            .join(Users, Users.username == TeamMembers.username)
            .filter(TeamMembers.team_id == team_id)
            .all())
    rows.sort(key=lambda row: (row[0].role != TeamRole.MAINTAINER.value, row[0].username))
    return [
        {"username": member.username, "display_name": display_name,
         "role": member.role, "joined_at": _iso(member.joined_at)}
        for member, display_name in rows
    ]


def set_team_member(team: Teams, username: str, role: TeamRole, db: Session) -> TeamMembers:
    """Add an organisation member to the team, or change their role in it."""
    if db.get(OrganizationMembers, (team.organization_id, username)) is None:
        raise _bad_request(f"{username} is not a member of the team's organisation.")
    member = db.get(TeamMembers, (team.id, username))
    if member is None:
        member = TeamMembers(team_id=team.id, username=username)
        db.add(member)
    member.role = role.value
    db.commit()
    return member


def remove_team_member(team: Teams, username: str, db: Session) -> None:
    member = db.get(TeamMembers, (team.id, username))
    if member is None:
        raise _not_found(f"{username} in this team")
    db.delete(member)
    db.commit()


# -- Team grants on datasets -----------------------------------------------------

def get_dataset(dataset_id: int, db: Session) -> Datasets:
    dataset = db.get(Datasets, dataset_id)
    if dataset is None:
        raise _not_found("Dataset")
    return dataset


def list_team_grants(dataset_id: int, db: Session) -> list[dict]:
    rows = (db.query(DatasetTeamGrants, Teams.name)
            .join(Teams, Teams.id == DatasetTeamGrants.team_id)
            .filter(DatasetTeamGrants.dataset_id == dataset_id)
            .order_by(Teams.name)
            .all())
    return [
        {
            "team_id": grant.team_id,
            "team_name": team_name,
            "role": grant.role,
            "extra_permissions": list(grant.extra_permissions or []),
            "denied_permissions": list(grant.denied_permissions or []),
            "granted_by": grant.granted_by,
            "granted_at": _iso(grant.granted_at),
        }
        for grant, team_name in rows
    ]


def grant_team(dataset_id: int, team_id: int, body: TeamGrant, granted_by: str,
               db: Session) -> DatasetTeamGrants:
    """Give a team a role on a dataset of its own organisation, or change it.

    Limited to the dataset's organisation: sharing across organisations goes
    through individual members, so an organisation's admins stay in control of who
    reaches its data through its teams.
    """
    dataset = get_dataset(dataset_id, db)
    team = get_team(team_id, db)
    if dataset.organization_id is None or team.organization_id != dataset.organization_id:
        raise _bad_request("A dataset can only be shared with teams of its own organisation.")
    grant = db.get(DatasetTeamGrants, (dataset_id, team_id))
    if grant is None:
        grant = DatasetTeamGrants(dataset_id=dataset_id, team_id=team_id)
        db.add(grant)
    grant.role = body.role.value
    grant.extra_permissions = [p.value for p in body.extra_permissions]
    grant.denied_permissions = [p.value for p in body.denied_permissions]
    grant.granted_by = granted_by
    grant.granted_at = utc_now()
    db.commit()
    return grant


def revoke_team(dataset_id: int, team_id: int, db: Session) -> bool:
    grant = db.get(DatasetTeamGrants, (dataset_id, team_id))
    if grant is None:
        return False
    db.delete(grant)
    db.commit()
    return True


# -- Datasets and their organisation ---------------------------------------------

def organization_for_new_dataset(user: AuthenticatedUser, requested: int | None,
                                 db: Session) -> int | None:
    """The organisation a dataset the caller creates belongs to.

    The one asked for, if the caller is in it. Otherwise the caller's only
    organisation, or failing that the default one if the caller is in it. With
    neither, the dataset is personal and can be moved into an organisation later.
    """
    if requested is not None:
        ensure_organization_member(user, requested)
        get_organization(requested, db)
        return requested
    if len(user.organizations) == 1:
        return next(iter(user.organizations))
    if user.organizations:
        return (db.query(Organizations.id)
                .filter(Organizations.is_default.is_(True),
                        Organizations.id.in_(list(user.organizations)))
                .scalar())
    return None


def set_dataset_organization(dataset_id: int, organization_id: int | None,
                             user: AuthenticatedUser, db: Session) -> int:
    """Move a dataset into another organisation (or out of any); returns grants dropped.

    Its team grants all belong to the old organisation, so they go: left in place
    they would reach across the boundary `grant_team` enforces.
    """
    dataset = get_dataset(dataset_id, db)
    if organization_id is not None:
        ensure_organization_member(user, organization_id)
        get_organization(organization_id, db)
    if organization_id == dataset.organization_id:
        return 0
    dropped = (db.query(DatasetTeamGrants).filter_by(dataset_id=dataset_id)
               .delete(synchronize_session=False))
    dataset.organization_id = organization_id
    db.commit()
    return dropped


def list_organization_datasets(organization_id: int, db: Session) -> list[dict]:
    """The organisation's datasets and their owners -- names only, never contents."""
    datasets = db.query(Datasets).filter_by(organization_id=organization_id).order_by(Datasets.name).all()
    owners: dict[int, list[str]] = {}
    for dataset_id, username in (db.query(DatasetMembers.dataset_id, DatasetMembers.username)
                                 .filter(DatasetMembers.role == DatasetRole.OWNER.value,
                                         DatasetMembers.dataset_id.in_([d.id for d in datasets]))):
        owners.setdefault(dataset_id, []).append(username)
    return [
        {"id": dataset.id, "name": dataset.name,
         "owners": owners.get(dataset.id, [dataset.created_by])}
        for dataset in datasets
    ]


def reassign_owner(organization_id: int, dataset_id: int, new_owner: str, db: Session) -> None:
    """Hand an organisation's dataset to a new owner, e.g. when its owner has left.

    The new owner has to be in the organisation. The previous owner becomes a
    curator, as in an ordinary transfer; deactivating their account is what takes
    their access away.
    """
    dataset = get_dataset(dataset_id, db)
    if dataset.organization_id != organization_id:
        raise _not_found("Dataset in this organisation")
    if db.get(OrganizationMembers, (organization_id, new_owner)) is None:
        raise _bad_request(f"{new_owner} is not a member of the organisation.")
    current_owner = (db.query(DatasetMembers.username)
                     .filter_by(dataset_id=dataset_id, role=DatasetRole.OWNER.value)
                     .scalar()) or dataset.created_by
    members_db.transfer_ownership(dataset_id, new_owner, current_owner=current_owner, db=db)
    logger.info("Dataset %s reassigned from %r to %r by an admin of organisation %s.",
                dataset_id, current_owner, new_owner, organization_id)

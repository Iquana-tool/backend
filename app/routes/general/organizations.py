"""Organisations and their teams.

Who may do what:

* platform admins (``organization.manage``) create and delete organisations and
  choose the default one, and pass every check below;
* an organisation's admins run it: its name, its members, its teams, and handing
  one of its datasets to a new owner when the owner has left;
* a team's maintainers decide who is in that team;
* any member of an organisation can see its members and teams.

None of these roles reaches into a dataset. Access to data comes only from a
dataset role, held directly or through a team (see ``/datasets/{id}/teams``).
"""
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.database import get_session
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.organizations import (
    OrganizationCreate,
    OrganizationMemberSet,
    OrganizationUpdate,
    TeamCreate,
    TeamMemberSet,
    TeamUpdate,
)
from app.schemas.permissions import Permission
from app.services.auth import get_current_user
from app.services.database_access import organizations as organizations_db
from app.services.permissions import require_global

router = APIRouter(tags=["organizations"])


# -- Organisations ---------------------------------------------------------------

@router.get("/organizations")
async def list_organizations(db: Session = Depends(get_session),
                             user: AuthenticatedUser = Depends(get_current_user)):
    """The caller's organisations (every organisation, for platform admins)."""
    return {"success": True, "organizations": organizations_db.list_organizations(user, db)}


@router.post("/organizations", status_code=status.HTTP_201_CREATED)
async def create_organization(body: OrganizationCreate,
                              db: Session = Depends(get_session),
                              user: AuthenticatedUser = Depends(require_global(Permission.ORGANIZATION_MANAGE))):
    """Create an organisation; its creator becomes its first admin."""
    organization = organizations_db.create_organization(body, user.username, db)
    return {"success": True, "organization": {"id": organization.id, "name": organization.name}}


@router.patch("/organizations/{organization_id}")
async def update_organization(organization_id: int,
                              body: OrganizationUpdate,
                              db: Session = Depends(get_session),
                              user: AuthenticatedUser = Depends(get_current_user)):
    """Rename an organisation, or make it the default that new accounts join."""
    organization = organizations_db.get_organization(organization_id, db)
    organizations_db.update_organization(organization, body, user, db)
    return {"success": True, "organization": {"id": organization.id, "name": organization.name,
                                              "is_default": bool(organization.is_default)}}


@router.delete("/organizations/{organization_id}")
async def delete_organization(organization_id: int,
                              db: Session = Depends(get_session),
                              user: AuthenticatedUser = Depends(require_global(Permission.ORGANIZATION_MANAGE))):
    """Delete an organisation that no longer has datasets, with its teams."""
    organizations_db.delete_organization(organizations_db.get_organization(organization_id, db), db)
    return {"success": True, "message": "Organisation deleted."}


# -- Organisation members --------------------------------------------------------

@router.get("/organizations/{organization_id}/members")
async def list_organization_members(organization_id: int,
                                    db: Session = Depends(get_session),
                                    user: AuthenticatedUser = Depends(get_current_user)):
    organizations_db.ensure_organization_member(user, organization_id)
    organizations_db.get_organization(organization_id, db)
    return {"success": True, "members": organizations_db.list_organization_members(organization_id, db)}


@router.put("/organizations/{organization_id}/members/{username}")
async def set_organization_member(organization_id: int,
                                  username: str,
                                  body: OrganizationMemberSet,
                                  db: Session = Depends(get_session),
                                  user: AuthenticatedUser = Depends(get_current_user)):
    """Add an account to the organisation, or change its role there."""
    organizations_db.ensure_organization_admin(user, organization_id)
    organizations_db.get_organization(organization_id, db)
    member = organizations_db.set_organization_member(organization_id, username, body.role, db)
    return {"success": True, "member": {"username": member.username, "role": member.role}}


@router.delete("/organizations/{organization_id}/members/{username}")
async def remove_organization_member(organization_id: int,
                                     username: str,
                                     db: Session = Depends(get_session),
                                     user: AuthenticatedUser = Depends(get_current_user)):
    """Remove an account from the organisation and from all of its teams."""
    organizations_db.ensure_organization_admin(user, organization_id)
    organizations_db.remove_organization_member(organization_id, username, db)
    return {"success": True, "message": f"Removed {username} from the organisation."}


# -- Teams -----------------------------------------------------------------------

@router.get("/organizations/{organization_id}/teams")
async def list_teams(organization_id: int,
                     db: Session = Depends(get_session),
                     user: AuthenticatedUser = Depends(get_current_user)):
    organizations_db.ensure_organization_member(user, organization_id)
    organizations_db.get_organization(organization_id, db)
    return {"success": True, "teams": organizations_db.list_teams(organization_id, db)}


@router.post("/organizations/{organization_id}/teams", status_code=status.HTTP_201_CREATED)
async def create_team(organization_id: int,
                      body: TeamCreate,
                      db: Session = Depends(get_session),
                      user: AuthenticatedUser = Depends(get_current_user)):
    """Create a team, optionally inside another one (e.g. a department)."""
    organizations_db.ensure_organization_admin(user, organization_id)
    organizations_db.get_organization(organization_id, db)
    team = organizations_db.create_team(organization_id, body, db)
    return {"success": True, "team": {"id": team.id, "name": team.name,
                                      "parent_team_id": team.parent_team_id}}


@router.patch("/teams/{team_id}")
async def update_team(team_id: int,
                      body: TeamUpdate,
                      db: Session = Depends(get_session),
                      user: AuthenticatedUser = Depends(get_current_user)):
    """Rename a team, change its description, or move it within its organisation."""
    team = organizations_db.get_team(team_id, db)
    organizations_db.ensure_organization_admin(user, team.organization_id)
    organizations_db.update_team(team, body, db)
    return {"success": True, "team": {"id": team.id, "name": team.name,
                                      "parent_team_id": team.parent_team_id}}


@router.delete("/teams/{team_id}")
async def delete_team(team_id: int,
                      db: Session = Depends(get_session),
                      user: AuthenticatedUser = Depends(get_current_user)):
    """Delete a team with its memberships and dataset grants; sub-teams move up to the top."""
    team = organizations_db.get_team(team_id, db)
    organizations_db.ensure_organization_admin(user, team.organization_id)
    organizations_db.delete_team(team, db)
    return {"success": True, "message": "Team deleted."}


# -- Team members ----------------------------------------------------------------

@router.get("/teams/{team_id}/members")
async def list_team_members(team_id: int,
                            db: Session = Depends(get_session),
                            user: AuthenticatedUser = Depends(get_current_user)):
    team = organizations_db.get_team(team_id, db)
    organizations_db.ensure_organization_member(user, team.organization_id)
    return {"success": True, "members": organizations_db.list_team_members(team_id, db)}


@router.put("/teams/{team_id}/members/{username}")
async def set_team_member(team_id: int,
                          username: str,
                          body: TeamMemberSet,
                          db: Session = Depends(get_session),
                          user: AuthenticatedUser = Depends(get_current_user)):
    """Add a member of the organisation to the team, or change their role in it."""
    team = organizations_db.get_team(team_id, db)
    organizations_db.ensure_team_manager(user, team, db)
    member = organizations_db.set_team_member(team, username, body.role, db)
    return {"success": True, "member": {"username": member.username, "role": member.role}}


@router.delete("/teams/{team_id}/members/{username}")
async def remove_team_member(team_id: int,
                             username: str,
                             db: Session = Depends(get_session),
                             user: AuthenticatedUser = Depends(get_current_user)):
    team = organizations_db.get_team(team_id, db)
    organizations_db.ensure_team_manager(user, team, db)
    organizations_db.remove_team_member(team, username, db)
    return {"success": True, "message": f"Removed {username} from the team."}


# -- The organisation's datasets -------------------------------------------------

@router.get("/organizations/{organization_id}/datasets")
async def list_organization_datasets(organization_id: int,
                                     db: Session = Depends(get_session),
                                     user: AuthenticatedUser = Depends(get_current_user)):
    """The organisation's datasets and their owners, for its admins. Names only."""
    organizations_db.ensure_organization_admin(user, organization_id)
    return {"success": True, "datasets": organizations_db.list_organization_datasets(organization_id, db)}


@router.post("/organizations/{organization_id}/datasets/{dataset_id}/transfer_ownership")
async def reassign_dataset_owner(organization_id: int,
                                 dataset_id: int,
                                 new_owner: str,
                                 db: Session = Depends(get_session),
                                 user: AuthenticatedUser = Depends(get_current_user)):
    """Hand one of the organisation's datasets to a new owner, e.g. after its owner left.

    The one thing an organisation admin can do to a dataset without a role on it --
    otherwise a dataset whose owner has gone could never be shared or managed again.
    """
    organizations_db.ensure_organization_admin(user, organization_id)
    organizations_db.reassign_owner(organization_id, dataset_id, new_owner, db)
    return {"success": True, "message": f"{new_owner} now owns dataset {dataset_id}."}

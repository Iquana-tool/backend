"""Dataset collaborator management: roles, team grants, invite links and ownership."""
from logging import getLogger

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_session
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import (
    DATASET_ROLE_PERMISSIONS,
    DatasetRole,
    GLOBAL_PERMISSIONS,
    Permission,
)
from app.schemas.organizations import DatasetOrganizationSet, TeamGrant
from app.schemas.review import InviteCreate, MemberGrant
from app.services.auth import get_current_user
from app.services.database_access import members as members_db
from app.services.database_access import organizations as organizations_db
from app.services.permissions import require

router = APIRouter(prefix="/datasets", tags=["members"])
logger = getLogger(__name__)


@router.get("/roles/catalog")
async def get_role_catalog(user: AuthenticatedUser = Depends(get_current_user)):
    """The role -> permission matrix, so the UI can explain what a role grants.

    Served rather than duplicated in the frontend, which otherwise drifts out of
    sync with the backend the moment a permission is added.
    """
    return {
        "success": True,
        "roles": [
            {
                "role": role.value,
                "permissions": sorted(p.value for p in permissions),
            }
            for role, permissions in DATASET_ROLE_PERMISSIONS.items()
        ],
        "global_permissions": sorted(p.value for p in GLOBAL_PERMISSIONS),
    }


@router.get("/{dataset_id}/members")
async def list_members(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.MEMBER_LIST)),
):
    """List everyone with a role on this dataset."""
    return {
        "success": True,
        "members": [member.model_dump(mode="json") for member in members_db.list_members(dataset_id, db)],
    }


@router.put("/{dataset_id}/members")
async def grant_member_role(
        dataset_id: int,
        body: MemberGrant,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.MEMBER_GRANT)),
):
    """Add a collaborator or change their role.

    `extra_permissions` / `denied_permissions` are the escape hatch for one-off
    exceptions (e.g. letting a single annotator download quantification results)
    without adding a role for it.
    """
    membership = members_db.grant_role(
        dataset_id,
        body.username,
        body.role,
        granted_by=user.username,
        db=db,
        extra_permissions=body.extra_permissions,
        denied_permissions=body.denied_permissions,
    )
    return {
        "success": True,
        "message": f"{body.username} is now {membership.role} on dataset {dataset_id}.",
        "member": {"username": membership.username, "role": membership.role},
    }


@router.delete("/{dataset_id}/members/{username}")
async def revoke_member(
        dataset_id: int,
        username: str,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.MEMBER_REVOKE)),
):
    """Remove a collaborator's access. Their annotations are left untouched."""
    removed = members_db.revoke_membership(dataset_id, username, db)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"{username} is not a member of dataset {dataset_id}.")
    return {"success": True, "message": f"Removed {username} from dataset {dataset_id}."}


@router.post("/{dataset_id}/transfer_ownership")
async def transfer_dataset_ownership(
        dataset_id: int,
        new_owner: str,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.DATASET_TRANSFER_OWNERSHIP)),
):
    """Hand ownership to another account; the current owner becomes a curator."""
    members_db.transfer_ownership(dataset_id, new_owner, current_owner=user.username, db=db)
    return {
        "success": True,
        "message": f"{new_owner} now owns dataset {dataset_id}. You are a curator on it.",
    }


# -- Teams and the dataset's organisation ----------------------------------

@router.get("/{dataset_id}/teams")
async def list_team_grants(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.MEMBER_LIST)),
):
    """List the teams with a role on this dataset.

    Each grant reaches the team's members and the members of every team below it.
    """
    return {"success": True, "teams": organizations_db.list_team_grants(dataset_id, db)}


@router.put("/{dataset_id}/teams/{team_id}")
async def grant_team_role(
        dataset_id: int,
        team_id: int,
        body: TeamGrant,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.MEMBER_GRANT)),
):
    """Give a team of the dataset's organisation a role on it (at most curator)."""
    grant = organizations_db.grant_team(dataset_id, team_id, body, granted_by=user.username, db=db)
    return {
        "success": True,
        "message": f"Team {team_id} is now {grant.role} on dataset {dataset_id}.",
        "team": {"team_id": grant.team_id, "role": grant.role},
    }


@router.delete("/{dataset_id}/teams/{team_id}")
async def revoke_team_role(
        dataset_id: int,
        team_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.MEMBER_REVOKE)),
):
    """Take a team's role away. Members who also hold a direct role keep that."""
    if not organizations_db.revoke_team(dataset_id, team_id, db):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Team {team_id} has no role on dataset {dataset_id}.")
    return {"success": True, "message": f"Removed team {team_id} from dataset {dataset_id}."}


@router.put("/{dataset_id}/organization")
async def set_dataset_organization(
        dataset_id: int,
        body: DatasetOrganizationSet,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.DATASET_TRANSFER_OWNERSHIP)),
):
    """Move the dataset into one of the caller's organisations, or make it personal.

    Its team grants belong to the old organisation and are removed.
    """
    dropped = organizations_db.set_dataset_organization(dataset_id, body.organization_id, user, db)
    return {
        "success": True,
        "message": f"Dataset {dataset_id} moved; {dropped} team grant(s) removed.",
        "organization_id": body.organization_id,
        "team_grants_removed": dropped,
    }


# -- Invite links ----------------------------------------------------------

@router.post("/{dataset_id}/invites")
async def create_invite(
        dataset_id: int,
        body: InviteCreate,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.INVITE_CREATE)),
):
    """Mint a shareable invite link granting a fixed role on this dataset.

    The raw token is returned exactly once — only its hash is stored, so a leaked
    database does not hand out dataset access.
    """
    invite, token = members_db.create_invite(dataset_id, body, created_by=user.username, db=db)
    return {
        "success": True,
        "message": "Invite link created. Copy it now; the token is not stored and cannot be shown again.",
        "invite": {
            "id": invite.id,
            "dataset_id": invite.dataset_id,
            "role": invite.role,
            "expires_at": invite.expires_at.isoformat() if invite.expires_at else None,
            "max_uses": invite.max_uses,
        },
        "token": token,
        # Relative on purpose: the backend does not know the frontend's origin.
        "invite_path": f"/invites/{token}",
    }


@router.get("/{dataset_id}/invites")
async def list_invites(
        dataset_id: int,
        include_inactive: bool = False,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.INVITE_CREATE)),
):
    """List this dataset's invite links (metadata only, never the tokens)."""
    return {
        "success": True,
        "invites": [inv.model_dump(mode="json")
                    for inv in members_db.list_invites(dataset_id, db, include_inactive)],
    }


@router.delete("/{dataset_id}/invites/{invite_id}")
async def revoke_invite(
        dataset_id: int,
        invite_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.INVITE_REVOKE)),
):
    """Disable an invite link. Members who already joined through it keep access."""
    revoked = members_db.revoke_invite(invite_id, dataset_id, db)
    if not revoked:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invite not found.")
    return {"success": True, "message": "Invite link revoked."}


# Redemption is deliberately not under /{dataset_id}: the invitee holds a token,
# not a dataset id, and must not need access to the dataset to look it up.
invite_router = APIRouter(prefix="/invites", tags=["members"])


@invite_router.get("/{token}")
async def preview_invite(
        token: str,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(get_current_user),
):
    """Show what accepting this link would grant, before accepting it."""
    preview = members_db.preview_invite(token, user.username, db)
    return {"success": True, "invite": preview.model_dump(mode="json")}


@invite_router.post("/{token}/accept")
async def accept_invite(
        token: str,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(get_current_user),
):
    """Join a dataset via an invite link.

    Redeeming never lowers an existing role, so following an annotator link as a
    curator is harmless.
    """
    dataset_id, role = members_db.accept_invite(token, user.username, db)
    return {
        "success": True,
        "message": f"You are now {role.value} on dataset {dataset_id}.",
        "dataset_id": dataset_id,
        "role": role.value,
    }

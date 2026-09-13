"""Dataset membership and invite-link management."""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from logging import getLogger

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.database.dataset_members import DatasetInvites, DatasetMembers
from app.database.datasets import Datasets
from app.database.users import Users
from app.schemas.permissions import DATASET_ROLE_ORDER, DatasetRole, Permission
from app.schemas.review import InviteCreate, InvitePreview, InviteRead, MemberRead

logger = getLogger(__name__)

#: How many bytes of entropy an invite token carries before base64url encoding.
_TOKEN_BYTES = 32


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _hash_token(token: str) -> str:
    """Invite tokens are stored hashed, like passwords: the link is a bearer secret."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _as_utc(value: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; treat those as UTC."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


# -- Memberships -----------------------------------------------------------

def get_membership(dataset_id: int, username: str, db: Session) -> DatasetMembers | None:
    return db.query(DatasetMembers).filter_by(dataset_id=dataset_id, username=username).first()


def list_members(dataset_id: int, db: Session) -> list[MemberRead]:
    """Every collaborator on a dataset, most privileged first."""
    rows = db.query(DatasetMembers).filter_by(dataset_id=dataset_id).all()
    members = [
        MemberRead(
            username=row.username,
            role=DatasetRole(row.role),
            extra_permissions=[Permission(p) for p in (row.extra_permissions or [])
                               if p in set(Permission)],
            denied_permissions=[Permission(p) for p in (row.denied_permissions or [])
                                if p in set(Permission)],
            granted_by=row.granted_by,
            granted_at=_as_utc(row.granted_at),
        )
        for row in rows
    ]
    members.sort(key=lambda m: (-DATASET_ROLE_ORDER[m.role], m.username))
    return members


def grant_role(dataset_id: int,
               username: str,
               role: DatasetRole,
               granted_by: str,
               db: Session,
               extra_permissions: list[Permission] | None = None,
               denied_permissions: list[Permission] | None = None) -> DatasetMembers:
    """Create or update a membership row.

    Granting `owner` here would create a second owner, which the ownership-transfer
    endpoint exists to avoid; callers must go through `transfer_ownership`.
    """
    if role is DatasetRole.OWNER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use the ownership transfer endpoint to make someone the owner.",
        )
    if db.query(Datasets.id).filter_by(id=dataset_id).scalar() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
    if db.query(Users.username).filter_by(username=username).scalar() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User '{username}' not found.")

    membership = get_membership(dataset_id, username, db)
    if membership is not None and DatasetRole(membership.role) is DatasetRole.OWNER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The owner's role cannot be changed. Transfer ownership first.",
        )

    if membership is None:
        membership = DatasetMembers(dataset_id=dataset_id, username=username)
        db.add(membership)

    membership.role = role.value
    membership.extra_permissions = [p.value for p in (extra_permissions or [])]
    membership.denied_permissions = [p.value for p in (denied_permissions or [])]
    membership.granted_by = granted_by
    membership.granted_at = _utcnow()
    db.commit()
    db.refresh(membership)
    return membership


def revoke_membership(dataset_id: int, username: str, db: Session) -> bool:
    """Remove a collaborator. The owner cannot be revoked."""
    membership = get_membership(dataset_id, username, db)
    if membership is None:
        return False
    if DatasetRole(membership.role) is DatasetRole.OWNER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The owner cannot be removed. Transfer ownership first.",
        )
    db.delete(membership)
    db.commit()
    return True


def transfer_ownership(dataset_id: int, new_owner: str, current_owner: str, db: Session) -> None:
    """Hand ownership to another account, demoting the previous owner to curator.

    `datasets.created_by` is deliberately left alone: it records who created the
    dataset, which stays true after a transfer.
    """
    if db.query(Users.username).filter_by(username=new_owner).scalar() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User '{new_owner}' not found.")
    if new_owner == current_owner:
        return

    previous = get_membership(dataset_id, current_owner, db)
    if previous is not None:
        previous.role = DatasetRole.CURATOR.value
        previous.granted_by = current_owner
        previous.granted_at = _utcnow()

    membership = get_membership(dataset_id, new_owner, db)
    if membership is None:
        membership = DatasetMembers(dataset_id=dataset_id, username=new_owner)
        db.add(membership)
    membership.role = DatasetRole.OWNER.value
    membership.extra_permissions = []
    membership.denied_permissions = []
    membership.granted_by = current_owner
    membership.granted_at = _utcnow()
    db.commit()


def ensure_owner_membership(dataset_id: int, username: str, db: Session) -> None:
    """Give a dataset's creator their owner membership row.

    Called on dataset creation, and by the migration script for existing datasets.
    """
    if get_membership(dataset_id, username, db) is not None:
        return
    db.add(DatasetMembers(
        dataset_id=dataset_id,
        username=username,
        role=DatasetRole.OWNER.value,
        extra_permissions=[],
        denied_permissions=[],
        granted_by=username,
        granted_at=_utcnow(),
    ))
    db.commit()


# -- Invite links ----------------------------------------------------------

def create_invite(dataset_id: int,
                  body: InviteCreate,
                  created_by: str,
                  db: Session) -> tuple[DatasetInvites, str]:
    """Mint an invite link. Returns the row and the raw token (shown once)."""
    if body.role is DatasetRole.OWNER:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invite links cannot grant ownership.")

    token = secrets.token_urlsafe(_TOKEN_BYTES)
    expires_at = None
    if body.expires_in_hours is not None:
        expires_at = _utcnow() + timedelta(hours=body.expires_in_hours)

    invite = DatasetInvites(
        token_hash=_hash_token(token),
        dataset_id=dataset_id,
        role=body.role.value,
        created_by=created_by,
        created_at=_utcnow(),
        expires_at=expires_at,
        max_uses=body.max_uses,
        uses=0,
    )
    db.add(invite)
    db.commit()
    db.refresh(invite)
    return invite, token


def _to_invite_read(invite: DatasetInvites) -> InviteRead:
    return InviteRead(
        id=invite.id,
        dataset_id=invite.dataset_id,
        role=DatasetRole(invite.role),
        created_by=invite.created_by,
        created_at=_as_utc(invite.created_at),
        expires_at=_as_utc(invite.expires_at),
        max_uses=invite.max_uses,
        uses=invite.uses or 0,
        revoked_at=_as_utc(invite.revoked_at),
        is_valid=invite.is_valid(),
    )


def list_invites(dataset_id: int, db: Session, include_inactive: bool = False) -> list[InviteRead]:
    """Invite links for a dataset. Tokens are never returned, only their metadata."""
    invites = db.query(DatasetInvites).filter_by(dataset_id=dataset_id).all()
    reads = [_to_invite_read(invite) for invite in invites]
    if not include_inactive:
        reads = [read for read in reads if read.is_valid]
    reads.sort(key=lambda read: read.created_at, reverse=True)
    return reads


def get_invite_by_token(token: str, db: Session) -> DatasetInvites | None:
    return db.query(DatasetInvites).filter_by(token_hash=_hash_token(token)).first()


def preview_invite(token: str, username: str, db: Session) -> InvitePreview:
    """What the invitee sees before accepting."""
    invite = get_invite_by_token(token, db)
    if invite is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invite link not found.")
    dataset = db.query(Datasets).filter_by(id=invite.dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset no longer exists.")

    membership = get_membership(invite.dataset_id, username, db)
    return InvitePreview(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        dataset_description=dataset.description,
        role=DatasetRole(invite.role),
        invited_by=invite.created_by,
        expires_at=_as_utc(invite.expires_at),
        is_valid=invite.is_valid(),
        already_member=membership is not None,
        current_role=DatasetRole(membership.role) if membership else None,
    )


def accept_invite(token: str, username: str, db: Session) -> tuple[int, DatasetRole]:
    """Redeem an invite link, returning the dataset id and the resulting role.

    Redeeming never lowers an existing role: a curator who follows an annotator
    link stays a curator, and the use is not counted.
    """
    invite = get_invite_by_token(token, db)
    if invite is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invite link not found.")
    if not invite.is_valid():
        raise HTTPException(status_code=status.HTTP_410_GONE,
                            detail="This invite link is no longer valid.")
    if db.query(Datasets.id).filter_by(id=invite.dataset_id).scalar() is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset no longer exists.")

    invited_role = DatasetRole(invite.role)
    membership = get_membership(invite.dataset_id, username, db)

    if membership is not None:
        current_role = DatasetRole(membership.role)
        if DATASET_ROLE_ORDER[current_role] >= DATASET_ROLE_ORDER[invited_role]:
            return invite.dataset_id, current_role
        membership.role = invited_role.value
    else:
        membership = DatasetMembers(
            dataset_id=invite.dataset_id,
            username=username,
            role=invited_role.value,
            extra_permissions=[],
            denied_permissions=[],
        )
        db.add(membership)

    membership.granted_by = invite.created_by
    membership.granted_at = _utcnow()
    invite.uses = (invite.uses or 0) + 1
    db.commit()
    return invite.dataset_id, invited_role


def revoke_invite(invite_id: int, dataset_id: int, db: Session) -> bool:
    """Kill an invite link without deleting the record of it having existed."""
    invite = db.query(DatasetInvites).filter_by(id=invite_id, dataset_id=dataset_id).first()
    if invite is None:
        return False
    if invite.revoked_at is None:
        invite.revoked_at = _utcnow()
        db.commit()
    return True

"""Registration links: minting, listing, revoking, and registering with one."""
import secrets
from datetime import timedelta

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.database.account_invites import AccountInvites, _as_utc, _utcnow
from app.database.users import Users
from app.schemas.account_invites import (
    AccountInviteCreate,
    AccountInvitePreview,
    AccountInviteRead,
    InviteRegistration,
)
from app.services.auth import get_password_hash
from app.services.database_access.members import _TOKEN_BYTES, _hash_token
from app.services.database_access.organizations import join_default_organization

_NOT_FOUND = "This registration link does not exist."
_USED_UP = "This registration link has expired, was revoked, or has already been used."


def create(body: AccountInviteCreate, created_by: str, db: Session) -> tuple[AccountInvites, str]:
    """Mint a link. Returns the row and the raw token, which is shown once."""
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    invite = AccountInvites(
        token_hash=_hash_token(token),
        created_by=created_by,
        note=body.note,
        created_at=_utcnow(),
        expires_at=(_utcnow() + timedelta(hours=body.expires_in_hours)
                    if body.expires_in_hours is not None else None),
        max_uses=body.max_uses,
        uses=0,
    )
    db.add(invite)
    db.commit()
    db.refresh(invite)
    return invite, token


def to_read(invite: AccountInvites) -> AccountInviteRead:
    return AccountInviteRead(
        id=invite.id, created_by=invite.created_by, note=invite.note,
        created_at=_as_utc(invite.created_at), expires_at=_as_utc(invite.expires_at),
        max_uses=invite.max_uses, uses=invite.uses or 0, revoked_at=_as_utc(invite.revoked_at),
        is_valid=invite.is_valid(),
    )


def list_active(db: Session) -> list[AccountInviteRead]:
    """Links that can still be used, newest first. Tokens are never returned."""
    reads = [to_read(invite) for invite in db.query(AccountInvites).all()]
    return sorted((r for r in reads if r.is_valid), key=lambda r: r.created_at, reverse=True)


def revoke(invite_id: int, db: Session) -> None:
    invite = db.get(AccountInvites, invite_id)
    if invite is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    if invite.revoked_at is None:
        invite.revoked_at = _utcnow()
        db.commit()


def preview(token: str, db: Session) -> AccountInvitePreview:
    invite = db.query(AccountInvites).filter_by(token_hash=_hash_token(token)).first()
    if invite is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    return AccountInvitePreview(is_valid=invite.is_valid(), invited_by=invite.created_by,
                                expires_at=_as_utc(invite.expires_at))


def register(body: InviteRegistration, db: Session) -> Users:
    """Create an account with a registration link and use up one of its uses.

    The invite row is locked first, so two people racing on a single-use link cannot
    both get an account. The password was chosen by the invitee, so unlike an account
    an admin creates it does not have to be changed at the first sign-in.
    """
    invite = (db.query(AccountInvites)
              .filter_by(token_hash=_hash_token(body.token))
              .with_for_update()
              .first())
    if invite is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)
    if not invite.is_valid():
        raise HTTPException(status_code=status.HTTP_410_GONE, detail=_USED_UP)
    if db.query(Users).filter_by(username=body.username).first() is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="That username is taken. Choose another one.")

    account = Users(username=body.username, hashed_password=get_password_hash(body.password),
                    is_active=True, must_change_password=False)
    db.add(account)
    join_default_organization(account.username, db)
    invite.uses = (invite.uses or 0) + 1
    db.commit()
    db.refresh(account)
    return account

"""Platform administration: account listing, activation and global roles.

Everything here is gated on a global permission rather than a dataset one, so an
admin can act without being a member of the dataset in question.
"""
from logging import getLogger

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.dataset_members import DatasetMembers
from app.database.users import Users
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import GlobalRole, Permission
from app.schemas.review import AdminUserCreate, GlobalRoleUpdate
from app.services.auth import get_password_hash
from app.services.permissions import ensure_global_permission, require_global

router = APIRouter(prefix="/admin", tags=["admin"])
logger = getLogger(__name__)


@router.get("/users")
async def list_users(
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.USER_MANAGE)),
):
    """List every account with its global role and dataset count."""
    users = db.query(Users).order_by(Users.username).all()
    # One grouped query rather than an N+1 walk over each account's memberships.
    counts = dict(
        db.query(DatasetMembers.username, func.count(DatasetMembers.dataset_id))
        .group_by(DatasetMembers.username)
        .all()
    )

    return {
        "success": True,
        "users": [
            {
                "username": account.username,
                "global_role": account.global_role,
                "is_active": bool(account.is_active),
                "dataset_count": counts.get(account.username, 0),
            }
            for account in users
        ],
    }


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(
        body: AdminUserCreate,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.USER_MANAGE)),
):
    """Create an account outright, without an invite or self-registration.

    An instance is closed by default (`INSTANCE_ALLOW_REGISTRATION`), and an
    invite only ever grants access to a single dataset -- so until now there was
    no way to hand somebody an account at all. This is that way.

    The password is chosen by the admin and passed on out of band; the account
    holder should change it afterwards.
    """
    if body.global_role is not GlobalRole.MEMBER:
        # Handing out a non-default role at creation is the same act as changing
        # one afterwards, so it answers to the same permission.
        ensure_global_permission(user, Permission.USER_SET_GLOBAL_ROLE)

    account = Users(
        username=body.username,
        hashed_password=get_password_hash(body.password),
        global_role=body.global_role.value,
        is_active=body.is_active,
    )
    db.add(account)
    try:
        db.commit()
    except IntegrityError:
        # Checked by letting the unique constraint answer rather than by a prior
        # SELECT, which two concurrent creates could both pass.
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Username already exists.")

    logger.info("Account %r created by %r as a platform %s.",
                account.username, user.username, account.global_role)
    return {
        "success": True,
        "message": f"Account {account.username} created.",
        "user": {
            "username": account.username,
            "global_role": account.global_role,
            "is_active": bool(account.is_active),
            "dataset_count": 0,
        },
    }


@router.patch("/users/{username}/global_role")
async def set_global_role(
        username: str,
        body: GlobalRoleUpdate,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.USER_SET_GLOBAL_ROLE)),
):
    """Change an account's platform-level role."""
    account = db.query(Users).filter_by(username=username).first()
    if account is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    if account.username == user.username and body.global_role is not GlobalRole.ADMIN:
        # Otherwise the last admin can lock themselves — and possibly everyone — out.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="You cannot remove your own admin role.")

    account.global_role = body.global_role.value
    db.commit()
    return {
        "success": True,
        "message": f"{username} is now a platform {body.global_role.value}.",
    }


@router.patch("/users/{username}/active")
async def set_user_active(
        username: str,
        is_active: bool,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.USER_MANAGE)),
):
    """Enable or disable an account.

    Deactivation is preferred over deletion: it revokes access immediately while
    leaving the annotations and review history the account produced intact.
    """
    account = db.query(Users).filter_by(username=username).first()
    if account is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    if account.username == user.username and not is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="You cannot deactivate your own account.")

    account.is_active = is_active
    db.commit()
    return {
        "success": True,
        "message": f"{username} is now {'active' if is_active else 'deactivated'}.",
    }

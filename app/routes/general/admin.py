"""Platform administration: accounts, and the instance's own configuration.

Everything here is gated on a global permission rather than a dataset one, so an
admin can act without being a member of the dataset in question.

Two jobs live side by side under this prefix because one page presents them:
administering *people* (``user.manage``) and administering the *deployment*
(``system.manage_settings``). They answer to separate permissions so the split
survives a future role that only does one of them.
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
from app.schemas.review import AdminUserCreate, GlobalRoleUpdate, SettingsUpdate
from app.services import settings as settings_service
from app.services.ai_services import ai_config
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


# -- Instance configuration ---------------------------------------------------
#
# Separate permission from the account endpoints above: reading a credential's
# existence and rotating it is a different job from deciding who has an account.


def _ai_service_values(specs, db: Session) -> dict[str, str | None]:
    """The AI-service-scoped settings among ``specs``, keyed by their env var."""
    keys = [spec.key for spec in specs if spec.scope == settings_service.SCOPE_AI_SERVICE]
    if not keys:
        return {}
    resolved = settings_service.get_many(*keys, db=db)
    return {settings_service.BY_KEY[key].env_var: resolved[key] for key in keys}


@router.get("/settings")
async def read_settings(
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.SYSTEM_MANAGE_SETTINGS)),
):
    """Describe every editable setting, plus the AI service's live state.

    Secret values are never returned -- only whether they are set and their last
    four characters, which distinguishes two keys without handing either back.
    """
    return {
        "success": True,
        "groups": [
            {"key": key, "label": label, "description": description}
            for key, label, description in settings_service.GROUPS
        ],
        "settings": settings_service.describe(db),
        # Lets the page show that the AI service is holding a different token
        # from the one stored here, instead of leaving a failed push invisible.
        "ai_service": await ai_config.read_config(),
    }


@router.patch("/settings")
async def update_settings(
        body: SettingsUpdate,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.SYSTEM_MANAGE_SETTINGS)),
):
    """Store overrides for the given settings and push the ones another service owns."""
    try:
        changed = settings_service.apply(db, body.values, user.username)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Unknown setting: {exc.args[0]}")
    db.commit()

    # Pushed after the commit, not before: the database is the record of what the
    # operator asked for, and a service that was down when they asked still has
    # to be able to catch up later from what is stored.
    push = None
    remote = _ai_service_values(changed, db)
    if remote:
        push = await ai_config.push_config(remote)

    return {
        "success": True,
        "message": f"{len(changed)} setting{'' if len(changed) == 1 else 's'} updated.",
        "settings": settings_service.describe(db),
        "push": push,
        "ai_service": await ai_config.read_config(),
    }


@router.delete("/settings/{key}")
async def clear_setting(
        key: str,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.SYSTEM_MANAGE_SETTINGS)),
):
    """Drop one override, falling back to the deployment's own configuration."""
    if key not in settings_service.BY_KEY:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Unknown setting: {key}")

    spec = settings_service.clear(db, key, user.username)
    db.commit()

    push = None
    remote = _ai_service_values([spec], db)
    if remote:
        push = await ai_config.push_config(remote)

    return {
        "success": True,
        "message": f"{spec.label} reset to the value configured for this deployment.",
        "settings": settings_service.describe(db),
        "push": push,
        "ai_service": await ai_config.read_config(),
    }


@router.post("/settings/push")
async def push_settings(
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require_global(Permission.SYSTEM_MANAGE_SETTINGS)),
):
    """Re-send the AI-service-scoped settings.

    The AI service holds its configuration in memory, so restarting it drops
    whatever was pushed. This is how an operator puts it back without editing a
    second ``.env`` -- and it is why the settings page shows the far side's live
    state rather than assuming the last push still holds.
    """
    result = await ai_config.push_config(_ai_service_values(settings_service.SETTINGS, db))
    return {
        "success": bool(result["pushed"]),
        "message": ("Sent to the AI service."
                    if result["pushed"]
                    else f"Could not reach the AI service: {result['error']}"),
        "push": result,
        "ai_service": await ai_config.read_config(),
    }

import json

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm
from iquana_toolbox.schemas.user import User
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.users import Users, utc_now
from app.schemas.account import MAX_PREFERENCES_BYTES, PasswordChange, ProfileUpdate
from app.schemas.auth_user import AuthenticatedUser
from app.services.auth import create_access_token, get_current_user, verify_password, get_password_hash
from app.services.instance import get_instance_config
from app.services.activity_log.emit import emit_navigation

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register")
def register_user(name, password, db: Session = Depends(get_session)):
    # Instances hold real research data and mostly hand out accounts by hand, so
    # self-registration is refused unless this deployment opted in. Checked here
    # rather than only hidden on the sign-in page: a link the frontend declines
    # to render is not a closed door.
    #
    # The exception is the very first account. Nothing else creates users — there
    # is no seeding script and no admin bootstrap — so a closed instance with an
    # empty user table could never be signed into at all. Allowing exactly the
    # first registration keeps "closed by default" from meaning "unusable by
    # default", and the window shuts the moment that account exists.
    if not get_instance_config().allow_registration:
        if db.query(Users).first() is not None:
            raise HTTPException(
                status_code=403,
                detail="Self-registration is disabled on this instance. Ask its administrator for an account.",
            )
    # Check if user already exists
    existing_user = db.query(Users).filter(Users.username == name).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    # Create new user
    new_user = Users(username=name, hashed_password=get_password_hash(password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {
        "success": True,
        "message": "User registered successfully",
        "name": new_user.username}


@router.post("/login")
def login_user(request: Request,
               form_data: OAuth2PasswordRequestForm = Depends(),
               db: Session = Depends(get_session)):
    user = db.query(Users).filter_by(username=form_data.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username")
    elif not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid password")
    access_token = create_access_token(data={"sub": user.username})
    user.last_login_at = utc_now()
    db.commit()
    # Marks the start of a participant's session server-side. Failed attempts are
    # deliberately not recorded here: the api component already logs the 401, and a
    # study has no use for a per-attempt record tied to a password entry.
    emit_navigation("session.login",
                    username=user.username,
                    session_id=request.headers.get("x-activity-session"))
    return {
        "success": True,
        "message": "Successfully logged in.",
        "access_token": access_token,
        "token_type": "bearer"}


@router.get("/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.patch("/me")
def update_profile(body: ProfileUpdate,
                   current_user: AuthenticatedUser = Depends(get_current_user),
                   db: Session = Depends(get_session)):
    """Change one's own display name, email address or preferences.

    Answers with the same shape as ``GET /auth/me``, so the client can replace what
    it holds instead of fetching it again.
    """
    account = db.query(Users).filter_by(username=current_user.username).one()
    sent = body.model_fields_set
    if "display_name" in sent:
        account.display_name = body.display_name
    if "email" in sent:
        account.email = body.email
    if body.preferences:
        merged = dict(account.preferences or {})
        for key, value in body.preferences.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        if len(json.dumps(merged)) > MAX_PREFERENCES_BYTES:
            raise HTTPException(status_code=422, detail="Preferences are too large to store.")
        # Assigned as a new dict: an in-place change to a JSON column is not seen
        # by the session and would silently not be saved.
        account.preferences = merged
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="That email address is already in use.")
    db.refresh(account)
    return AuthenticatedUser.from_query(account)


@router.post("/password")
def change_password(body: PasswordChange,
                    current_user: AuthenticatedUser = Depends(get_current_user),
                    db: Session = Depends(get_session)):
    """Change one's own password, signing out every other session.

    The response carries a fresh token for this session, which would otherwise be
    signed out along with the rest.
    """
    account = db.query(Users).filter_by(username=current_user.username).one()
    if not verify_password(body.current_password, account.hashed_password):
        # 400, not 401: the frontend treats any 401 as an expired session and
        # signs the user out, which a mistyped password must not do.
        raise HTTPException(status_code=400, detail="The current password is not correct.")
    if body.new_password == body.current_password:
        raise HTTPException(status_code=400, detail="Choose a password different from the current one.")

    account.hashed_password = get_password_hash(body.new_password)
    account.must_change_password = False
    account.sign_out_everywhere()
    db.commit()
    return {
        "success": True,
        "message": "Password changed. Other sessions have been signed out.",
        "access_token": create_access_token(data={"sub": account.username}),
        "token_type": "bearer",
    }
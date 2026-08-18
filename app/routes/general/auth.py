from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from iquana_toolbox.schemas.user import User
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.users import Users
from app.services.auth import create_access_token, get_current_user, verify_password, get_password_hash
from app.services.instance import get_instance_config

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
def login_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_session)):
    user = db.query(Users).filter_by(username=form_data.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username")
    elif not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid password")
    access_token = create_access_token(data={"sub": user.username})
    return {
        "success": True,
        "message": "Successfully logged in.",
        "access_token": access_token,
        "token_type": "bearer"}


@router.get("/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
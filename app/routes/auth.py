from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.users import Users
from app.services.auth import TokenResponse, create_access_token, get_current_user, verify_password, get_password_hash

router = APIRouter()


# JWT settings (move to config in production)


@router.post("/register")
def register_user(name, password, db: Session = Depends(get_session)):
    # Check if user already exists
    existing_user = db.query(Users).filter(Users.name == name).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    # Create new user
    new_user = Users(name=name, enc_password=get_password_hash(password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {
        "success": True,
        "message": "User registered successfully",
        "id": new_user.id,
        "name": new_user.name}

@router.post("/login", response_model=TokenResponse)
def login_user(name, password, db: Session = Depends(get_session)):
    user = db.query(Users).filter(Users.name == name).first()
    if not user or not verify_password(password, user.enc_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    access_token = create_access_token(data={"sub": user.name, "user_id": user.id})
    return {
        "success": True,
        "message": "Successfully logged in.",
        "access_token": access_token,
        "token_type": "bearer"}


@router.get("/me")
def read_users_me(current_user: Users = Depends(get_current_user)):
    return {"id": current_user.id, "name": current_user.name}
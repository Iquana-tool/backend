from datetime import timedelta, datetime, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi import status
from fastapi.security import OAuth2PasswordBearer
from fastapi.websockets import WebSocket
from jwt import InvalidTokenError
from pwdlib import PasswordHash
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.users import Users
from app.schemas.auth_user import AuthenticatedUser
from config import SECRET_KEY

password_hash = PasswordHash.recommended()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def verify_password(plain_password, hashed_password):
    return password_hash.verify(plain_password, hashed_password)


def get_password_hash(password):
    return password_hash.hash(password)


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def _username_from_token(token: str) -> str | None:
    """Decode a bearer token and return its subject, or None if it is not usable."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except InvalidTokenError:
        return None
    return payload.get("sub")


def load_user(username: str, db: Session) -> AuthenticatedUser | None:
    """Load a user together with the memberships their permissions derive from."""
    user_db = db.query(Users).filter_by(username=username).first()
    if user_db is None:
        return None
    return AuthenticatedUser.from_query(user_db)


async def get_current_user(token: str = Depends(oauth2_scheme),
                           db: Session = Depends(get_session)) -> AuthenticatedUser:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    username = _username_from_token(token)
    if username is None:
        raise credentials_exception
    user = load_user(username, db)
    if user is None:
        raise credentials_exception
    if not user.is_active:
        # Deactivated accounts keep their annotations but cannot act.
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="This account has been deactivated.")
    return user


async def authenticate_websocket(websocket: WebSocket, db: Session) -> AuthenticatedUser | None:
    """Resolve the caller behind a WebSocket connection, or None if unauthenticated.

    Browsers cannot set headers on a WebSocket handshake, so the token is accepted
    from the `token` query parameter as well as from an `Authorization` header for
    non-browser clients. The identity always comes from the token: the `user_id` in
    the URL is display information and is never trusted.
    """
    token = websocket.query_params.get("token")
    if not token:
        header = websocket.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = header[len("bearer "):].strip()
    if not token:
        return None

    username = _username_from_token(token)
    if username is None:
        return None
    user = load_user(username, db)
    if user is None or not user.is_active:
        return None
    return user

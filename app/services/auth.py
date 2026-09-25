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
from app.database.users import Users, as_utc
from app.schemas.auth_user import AuthenticatedUser
from config import ACCESS_TOKEN_EXPIRE_MINUTES, SECRET_KEY

password_hash = PasswordHash.recommended()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
ALGORITHM = "HS256"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def verify_password(plain_password, hashed_password):
    return password_hash.verify(plain_password, hashed_password)


def get_password_hash(password):
    return password_hash.hash(password)


def create_access_token(data: dict, expires_delta: timedelta = None):
    now = datetime.now(timezone.utc)
    to_encode = data.copy()
    # `iat` is what lets an account be signed out everywhere: see _revoked().
    to_encode.update({
        "iat": now,
        "exp": now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)),
    })
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def _decode_token(token: str) -> dict | None:
    """Decode a bearer token, or None if it is malformed, forged or expired."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except InvalidTokenError:
        return None


def _username_from_token(token: str) -> str | None:
    """Decode a bearer token and return its subject, or None if it is not usable.

    Does not look at the account, so a token revoked by `sign_out_everywhere` still
    names its subject here. Only for attributing activity; anything that grants
    access goes through `user_for_token`.
    """
    payload = _decode_token(token)
    return payload.get("sub") if payload else None


def _revoked(payload: dict, user_db: Users) -> bool:
    """Whether the token was issued before the account was last signed out everywhere.

    `iat` has whole-second precision, so the cut-off is compared at that precision
    too: the token handed out right after a password change is issued in the same
    second as the cut-off and has to survive it. Tokens without `iat` predate this
    check and cannot show their age, so they fall at the first revocation.
    """
    cutoff = user_db.tokens_valid_after
    if cutoff is None:
        return False
    issued_at = payload.get("iat")
    if issued_at is None:
        return True
    return int(issued_at) < int(as_utc(cutoff).timestamp())


def user_for_token(token: str, db: Session) -> AuthenticatedUser | None:
    """The caller a token signs in, with their permissions, or None if it no longer does."""
    payload = _decode_token(token)
    if payload is None or payload.get("sub") is None:
        return None
    user_db = db.query(Users).filter_by(username=payload["sub"]).first()
    if user_db is None or _revoked(payload, user_db):
        return None
    return AuthenticatedUser.from_query(user_db)


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
    user = user_for_token(token, db)
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

    user = user_for_token(token, db)
    if user is None or not user.is_active:
        return None
    return user

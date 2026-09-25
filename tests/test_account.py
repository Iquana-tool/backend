"""Checks for managing one's own account: login bookkeeping, profile, password.

These go through real tokens rather than overriding ``get_current_user``, because
what is under test is partly the token check itself: a password change has to
invalidate the tokens issued before it and keep working with the one it hands out.
"""
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database, get_session, import_models
from app.database.users import Users, utc_now
from app.routes.general.auth import router as auth_router
from app.schemas.account import MAX_PREFERENCES_BYTES
from app.services.auth import ALGORITHM, get_password_hash, verify_password
from config import SECRET_KEY

import_models()


def _token_issued_ago(username: str, seconds: int, with_iat: bool = True) -> str:
    """A login token issued some seconds back, i.e. clearly before a cut-off set now."""
    issued = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    claims = {"sub": username, "exp": issued + timedelta(hours=1)}
    if with_iat:
        claims["iat"] = issued
    return jwt.encode(claims, SECRET_KEY, algorithm=ALGORITHM)


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def ctx(tmp_path):
    """The auth router on a temp database holding two accounts."""
    engine = create_engine(f"sqlite:///{tmp_path / 'account.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    session = Session()
    session.add_all([
        Users(username="ann", hashed_password=get_password_hash("ann-pass-123"),
              must_change_password=True),
        Users(username="bob", hashed_password=get_password_hash("bob-pass-123"),
              email="bob@example.org"),
    ])
    session.commit()
    session.close()

    app = FastAPI()
    app.include_router(auth_router)

    def override_session():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_session] = override_session
    yield {"client": TestClient(app), "Session": Session}
    engine.dispose()


def _account(ctx, username: str) -> Users:
    session = ctx["Session"]()
    try:
        return session.query(Users).filter_by(username=username).one()
    finally:
        session.close()


def _login(ctx, username: str, password: str) -> str:
    response = ctx["client"].post("/auth/login", data={"username": username, "password": password})
    assert response.status_code == 200
    return response.json()["access_token"]


# -- Login and /auth/me ---------------------------------------------------------

def test_login_records_when_it_happened(ctx):
    before = utc_now().replace(microsecond=0)
    _login(ctx, "ann", "ann-pass-123")

    assert _account(ctx, "ann").last_login_at >= before


def test_me_reports_the_profile_and_the_password_flag(ctx):
    token = _login(ctx, "ann", "ann-pass-123")

    me = ctx["client"].get("/auth/me", headers=_bearer(token)).json()

    assert me["username"] == "ann"
    assert me["must_change_password"] is True
    assert me["display_name"] is None
    assert me["preferences"] == {}


# -- Profile ------------------------------------------------------------------

def test_profile_update_changes_only_what_was_sent(ctx):
    token = _login(ctx, "bob", "bob-pass-123")

    response = ctx["client"].patch("/auth/me", headers=_bearer(token),
                                   json={"display_name": "  Bob Builder "})

    assert response.status_code == 200
    assert response.json()["display_name"] == "Bob Builder"
    assert response.json()["email"] == "bob@example.org"


def test_email_is_lower_cased_and_can_be_cleared(ctx):
    token = _login(ctx, "ann", "ann-pass-123")
    client = ctx["client"]

    assert client.patch("/auth/me", headers=_bearer(token),
                        json={"email": "Ann@Example.ORG"}).json()["email"] == "ann@example.org"
    assert client.patch("/auth/me", headers=_bearer(token),
                        json={"email": None}).json()["email"] is None


def test_email_taken_by_someone_else_conflicts(ctx):
    token = _login(ctx, "ann", "ann-pass-123")

    response = ctx["client"].patch("/auth/me", headers=_bearer(token),
                                   json={"email": "BOB@example.org"})

    assert response.status_code == 409
    assert _account(ctx, "ann").email is None


def test_preferences_are_merged_and_null_removes_a_key(ctx):
    token = _login(ctx, "ann", "ann-pass-123")
    client = ctx["client"]

    client.patch("/auth/me", headers=_bearer(token),
                 json={"preferences": {"theme": "dark", "tool": "lasso"}})
    response = client.patch("/auth/me", headers=_bearer(token),
                            json={"preferences": {"tool": None, "mode": "review"}})

    assert response.json()["preferences"] == {"theme": "dark", "mode": "review"}
    assert _account(ctx, "ann").preferences == {"theme": "dark", "mode": "review"}


def test_oversized_preferences_are_refused(ctx):
    token = _login(ctx, "ann", "ann-pass-123")

    response = ctx["client"].patch("/auth/me", headers=_bearer(token),
                                   json={"preferences": {"blob": "x" * MAX_PREFERENCES_BYTES}})

    assert response.status_code == 422
    assert _account(ctx, "ann").preferences == {}


# -- Password -----------------------------------------------------------------

def test_password_change_signs_out_other_sessions_but_not_this_one(ctx):
    old_token = _token_issued_ago("ann", 10)
    client = ctx["client"]

    response = client.post("/auth/password", headers=_bearer(old_token), json={
        "current_password": "ann-pass-123", "new_password": "ann-new-pass-456"})

    assert response.status_code == 200
    new_token = response.json()["access_token"]
    assert client.get("/auth/me", headers=_bearer(new_token)).status_code == 200
    assert client.get("/auth/me", headers=_bearer(old_token)).status_code == 401

    account = _account(ctx, "ann")
    assert verify_password("ann-new-pass-456", account.hashed_password)
    assert account.must_change_password is False


def test_wrong_current_password_is_a_400_not_a_sign_out(ctx):
    token = _login(ctx, "ann", "ann-pass-123")

    response = ctx["client"].post("/auth/password", headers=_bearer(token), json={
        "current_password": "wrong", "new_password": "ann-new-pass-456"})

    assert response.status_code == 400
    assert verify_password("ann-pass-123", _account(ctx, "ann").hashed_password)


def test_new_password_must_differ_and_be_long_enough(ctx):
    token = _login(ctx, "ann", "ann-pass-123")
    client = ctx["client"]

    same = client.post("/auth/password", headers=_bearer(token), json={
        "current_password": "ann-pass-123", "new_password": "ann-pass-123"})
    short = client.post("/auth/password", headers=_bearer(token), json={
        "current_password": "ann-pass-123", "new_password": "short"})

    assert same.status_code == 400
    assert short.status_code == 422


# -- Token cut-off ------------------------------------------------------------

def test_tokens_without_issue_time_work_until_the_first_sign_out(ctx):
    legacy = _token_issued_ago("bob", 10, with_iat=False)
    client = ctx["client"]
    assert client.get("/auth/me", headers=_bearer(legacy)).status_code == 200

    session = ctx["Session"]()
    session.query(Users).filter_by(username="bob").one().sign_out_everywhere()
    session.commit()
    session.close()

    # It cannot show that it was issued after the cut-off, so it falls with it.
    assert client.get("/auth/me", headers=_bearer(legacy)).status_code == 401

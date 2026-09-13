"""Checks for the admin account-creation endpoint.

An instance is closed by default and invites are per dataset, so `POST
/admin/users` is the only way to hand somebody an account. These cover who may
call it, what it stores, and that the created account can actually sign in.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database, get_session
import app.database.dataset_members  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.users  # noqa: F401
from app.database.users import Users
from app.routes.general.admin import router as admin_router
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import GlobalRole
from app.services.auth import get_current_user, verify_password


def _as_caller(username: str, role: GlobalRole) -> AuthenticatedUser:
    """The signed-in caller the route sees, with no dataset memberships."""
    return AuthenticatedUser(username=username, is_admin=role is GlobalRole.ADMIN,
                             global_role=role, owned_datasets=[], accessible_datasets=[])


@pytest.fixture
def ctx(tmp_path):
    """The admin router on a temp database, with the caller swappable per test."""
    engine = create_engine(f"sqlite:///{tmp_path / 'admin.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    session = Session()
    session.add_all([
        Users(username="root", hashed_password="x", global_role=GlobalRole.ADMIN.value),
        Users(username="ann", hashed_password="x", global_role=GlobalRole.MEMBER.value),
    ])
    session.commit()
    session.close()

    app = FastAPI()
    app.include_router(admin_router)

    caller = {"user": _as_caller("root", GlobalRole.ADMIN)}

    def override_session():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_current_user] = lambda: caller["user"]

    return {"client": TestClient(app), "caller": caller, "Session": Session}


def test_admin_creates_a_usable_account(ctx):
    response = ctx["client"].post("/admin/users",
                                  json={"username": "demo", "password": "demo-pass-123"})
    assert response.status_code == 201
    assert response.json()["user"] == {
        "username": "demo",
        "global_role": GlobalRole.MEMBER.value,
        "is_active": True,
        "dataset_count": 0,
    }

    session = ctx["Session"]()
    account = session.query(Users).filter_by(username="demo").one()
    # Stored hashed, and the hash verifies against the password the admin chose --
    # otherwise the account exists but nobody can sign into it.
    assert account.hashed_password != "demo-pass-123"
    assert verify_password("demo-pass-123", account.hashed_password)
    session.close()


def test_username_is_trimmed(ctx):
    response = ctx["client"].post("/admin/users",
                                  json={"username": "  demo  ", "password": "demo-pass-123"})
    assert response.status_code == 201
    assert response.json()["user"]["username"] == "demo"


def test_duplicate_username_conflicts(ctx):
    response = ctx["client"].post("/admin/users",
                                  json={"username": "ann", "password": "demo-pass-123"})
    assert response.status_code == 409


def test_short_password_is_rejected(ctx):
    response = ctx["client"].post("/admin/users",
                                  json={"username": "demo", "password": "short"})
    assert response.status_code == 422


def test_non_admin_cannot_create_accounts(ctx):
    ctx["caller"]["user"] = _as_caller("ann", GlobalRole.MEMBER)
    response = ctx["client"].post("/admin/users",
                                  json={"username": "demo", "password": "demo-pass-123"})
    assert response.status_code == 403

    session = ctx["Session"]()
    assert session.query(Users).filter_by(username="demo").first() is None
    session.close()


def test_admin_can_seed_a_guest_or_an_admin(ctx):
    for username, role in (("visitor", GlobalRole.GUEST), ("root2", GlobalRole.ADMIN)):
        response = ctx["client"].post("/admin/users", json={
            "username": username, "password": "demo-pass-123", "global_role": role.value,
        })
        assert response.status_code == 201
        assert response.json()["user"]["global_role"] == role.value


def test_account_can_be_created_deactivated(ctx):
    response = ctx["client"].post("/admin/users", json={
        "username": "later", "password": "demo-pass-123", "is_active": False,
    })
    assert response.status_code == 201
    assert response.json()["user"]["is_active"] is False

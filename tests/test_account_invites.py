"""Registration links: an admin mints one, someone registers with it, it runs out."""
from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database, get_session, import_models
from app.database.account_invites import AccountInvites, _utcnow
from app.database.users import Users
from app.routes.general import admin as admin_routes
from app.routes.general import auth as auth_routes
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import GlobalRole
from app.services.auth import get_current_user, verify_password
from app.services.database_access import account_invites as invites_db

import_models()


def _admin():
    return AuthenticatedUser(username="root", is_admin=True, global_role=GlobalRole.ADMIN,
                             is_active=True, owned_datasets=[], accessible_datasets=[],
                             memberships={})


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'invites.db'}")
    database.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(Users(username="root", hashed_password="x", global_role=GlobalRole.ADMIN.value))
    db.commit()
    monkeypatch.setattr(invites_db, "join_default_organization", lambda username, session: None)

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.include_router(auth_routes.router)
    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[get_current_user] = _admin
    yield TestClient(app), db
    db.close()


def _mint(client, **body):
    response = client.post("/admin/account-invites", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_a_link_lets_one_person_register_and_is_then_used_up(ctx):
    client, db = ctx
    minted = _mint(client, note="Anna, coral lab")
    assert minted["invite_path"] == f"/join/{minted['token']}"
    assert minted["invite"]["max_uses"] == 1  # single-use by default
    # Only the hash is stored.
    assert db.query(AccountInvites).one().token_hash != minted["token"]

    preview = client.get(f"/auth/invites/{minted['token']}").json()["invite"]
    assert preview == {"is_valid": True, "invited_by": "root", "expires_at": preview["expires_at"]}

    response = client.post("/auth/register", json={"token": minted["token"], "username": " anna ",
                                                   "password": "correct horse"})
    assert response.status_code == 200, response.text
    anna = db.query(Users).filter_by(username="anna").one()
    assert anna.global_role == GlobalRole.MEMBER.value and anna.is_active
    assert not anna.must_change_password  # she chose it herself
    assert verify_password("correct horse", anna.hashed_password)

    again = client.post("/auth/register", json={"token": minted["token"], "username": "bob",
                                                "password": "correct horse"})
    assert again.status_code == 410
    assert client.get(f"/auth/invites/{minted['token']}").json()["invite"]["is_valid"] is False
    assert client.get("/admin/account-invites").json()["invites"] == []


def test_registering_needs_a_real_link(ctx):
    client, db = ctx
    response = client.post("/auth/register", json={"token": "made-up", "username": "eve",
                                                   "password": "correct horse"})
    assert response.status_code == 404
    assert db.query(Users).filter_by(username="eve").first() is None
    assert client.post("/auth/register", json={"username": "eve", "password": "x" * 8}).status_code == 422


def test_revoked_and_expired_links_refuse(ctx):
    client, db = ctx
    revoked = _mint(client)
    assert client.delete(f"/admin/account-invites/{revoked['invite']['id']}").status_code == 200
    expired = _mint(client, expires_in_hours=1)
    row = db.get(AccountInvites, expired["invite"]["id"])
    row.expires_at = _utcnow() - timedelta(minutes=1)
    db.commit()
    for minted in (revoked, expired):
        response = client.post("/auth/register", json={"token": minted["token"], "username": "late",
                                                       "password": "correct horse"})
        assert response.status_code == 410


def test_a_taken_username_does_not_use_up_the_link(ctx):
    client, db = ctx
    minted = _mint(client)
    response = client.post("/auth/register", json={"token": minted["token"], "username": "root",
                                                   "password": "correct horse"})
    assert response.status_code == 409
    assert db.get(AccountInvites, minted["invite"]["id"]).uses == 0


def test_a_multi_use_link_counts_its_uses(ctx):
    client, db = ctx
    minted = _mint(client, max_uses=None, expires_in_hours=None)
    for name in ("one", "two", "three"):
        assert client.post("/auth/register", json={"token": minted["token"], "username": name,
                                                   "password": "correct horse"}).status_code == 200
    listed = client.get("/admin/account-invites").json()["invites"]
    assert [(i["uses"], i["max_uses"], i["expires_at"]) for i in listed] == [(3, None, None)]

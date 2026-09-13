"""Checks for the instance settings surface behind the admin page.

The two properties worth defending are that a stored override beats the
environment (that is the whole point -- otherwise an operator's edit does
nothing until the next restart) and that a secret never comes back out of the
API once written.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database, get_session
import app.database.dataset_members  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.instance_settings  # noqa: F401
import app.database.users  # noqa: F401
from app.routes.general.admin import router as admin_router
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import GlobalRole
from app.services import settings as settings_service
from app.services.ai_services import ai_config
from app.services.auth import get_current_user


def _as_caller(username: str, role: GlobalRole) -> AuthenticatedUser:
    return AuthenticatedUser(username=username, is_admin=role is GlobalRole.ADMIN,
                             global_role=role, owned_datasets=[], accessible_datasets=[])


def _find(payload: dict, key: str) -> dict:
    return next(item for item in payload["settings"] if item["key"] == key)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """The admin router on a temp database, with the AI service stubbed out."""
    engine = create_engine(f"sqlite:///{tmp_path / 'settings.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

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

    # The settings service resolves against the environment as it stood the first
    # time each variable was asked for; a fresh snapshot per test keeps one test's
    # writes out of the next one's defaults.
    monkeypatch.setattr(settings_service, "_ENV_SNAPSHOT", {})

    pushed = []

    async def fake_push(values):
        pushed.append(values)
        return {"pushed": True, "error": None}

    async def fake_read():
        return {"reachable": True, "hf_token_set": bool(pushed and pushed[-1].get("HF_ACCESS_TOKEN"))}

    monkeypatch.setattr(ai_config, "push_config", fake_push)
    monkeypatch.setattr(ai_config, "read_config", fake_read)

    return {"client": TestClient(app), "caller": caller, "Session": Session, "pushed": pushed}


def test_environment_supplies_the_default(ctx, monkeypatch):
    monkeypatch.setenv("INSTANCE_NAME", "Reef Lab")

    setting = _find(ctx["client"].get("/admin/settings").json(), "instance_name")
    assert setting["value"] == "Reef Lab"
    # Nothing was stored, so the page must not claim the deployment is overridden.
    assert setting["overridden"] is False


def test_a_stored_value_overrides_the_environment(ctx, monkeypatch):
    monkeypatch.setenv("INSTANCE_CONTACT", "old@example.org")

    response = ctx["client"].patch("/admin/settings",
                                   json={"values": {"instance_contact": "new@example.org"}})
    assert response.status_code == 200

    setting = _find(response.json(), "instance_contact")
    assert setting["value"] == "new@example.org"
    assert setting["overridden"] is True
    assert setting["updated_by"] == "root"

    # And it survives into a fresh read rather than living only in the response.
    assert _find(ctx["client"].get("/admin/settings").json(), "instance_contact")["value"] \
        == "new@example.org"


def test_clearing_falls_back_to_the_deployments_own_value(ctx, monkeypatch):
    monkeypatch.setenv("INSTANCE_NAME", "Configured In Env")
    ctx["client"].patch("/admin/settings", json={"values": {"instance_name": "Edited"}})

    response = ctx["client"].delete("/admin/settings/instance_name")
    assert response.status_code == 200

    setting = _find(response.json(), "instance_name")
    assert setting["value"] == "Configured In Env"
    assert setting["overridden"] is False


def test_a_secret_is_never_read_back(ctx):
    ctx["client"].patch("/admin/settings", json={"values": {"llm_api_key": "sk-supersecret"}})

    setting = _find(ctx["client"].get("/admin/settings").json(), "llm_api_key")
    assert setting["is_set"] is True
    assert setting["value"] is None
    # Enough to tell two keys apart, not enough to use one.
    assert setting["hint"] == "…cret"


def test_an_empty_secret_field_leaves_the_stored_one_alone(ctx):
    """The field renders blank because the value is never sent back.

    Saving the page after editing something else must not wipe the key that was
    simply not retyped.
    """
    ctx["client"].patch("/admin/settings", json={"values": {"llm_api_key": "sk-keepme"}})
    ctx["client"].patch("/admin/settings",
                        json={"values": {"llm_api_key": "", "llm_model": "openai/gpt-4o"}})

    payload = ctx["client"].get("/admin/settings").json()
    assert _find(payload, "llm_api_key")["hint"] == "…epme"
    assert _find(payload, "llm_model")["value"] == "openai/gpt-4o"


def test_booleans_are_normalised(ctx):
    ctx["client"].patch("/admin/settings", json={"values": {"allow_registration": "yes"}})
    assert _find(ctx["client"].get("/admin/settings").json(), "allow_registration")["value"] == "true"

    ctx["client"].patch("/admin/settings", json={"values": {"allow_registration": "false"}})
    assert _find(ctx["client"].get("/admin/settings").json(), "allow_registration")["value"] == "false"


def test_an_unknown_key_is_refused_rather_than_ignored(ctx):
    response = ctx["client"].patch("/admin/settings", json={"values": {"secret_key": "pwned"}})
    assert response.status_code == 400
    assert "secret_key" in response.json()["detail"]


def test_an_ai_service_setting_is_pushed_across(ctx):
    ctx["client"].patch("/admin/settings", json={"values": {"hf_token": "hf_abcdefgh"}})
    assert ctx["pushed"] == [{"HF_ACCESS_TOKEN": "hf_abcdefgh"}]


def test_a_backend_setting_is_not_pushed_across(ctx):
    ctx["client"].patch("/admin/settings", json={"values": {"llm_model": "openai/gpt-4o"}})
    assert ctx["pushed"] == []


def test_push_resends_every_ai_service_setting(ctx):
    ctx["client"].patch("/admin/settings", json={"values": {"hf_token": "hf_abcdefgh"}})
    ctx["pushed"].clear()

    response = ctx["client"].post("/admin/settings/push")
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert ctx["pushed"] == [{"HF_ACCESS_TOKEN": "hf_abcdefgh"}]


def test_a_member_cannot_read_or_write_settings(ctx):
    ctx["caller"]["user"] = _as_caller("ann", GlobalRole.MEMBER)

    assert ctx["client"].get("/admin/settings").status_code == 403
    assert ctx["client"].patch("/admin/settings",
                               json={"values": {"instance_name": "x"}}).status_code == 403

"""Checks for personal and organisation API keys, and for which one a call uses.

The fixture:

* organisation "Lab" (personal keys allowed): ann (admin), bob (member);
* organisation "Strict" (personal keys switched off): sam (admin);
* solo -- in no organisation;
* dataset "Reef" in Lab, dataset "Vault" in Strict, on which bob is an annotator.

The instance key is stored as a row rather than taken from the environment, which
is cleared, so a developer's ``.env`` cannot leak into the results.
"""
import pytest
from cryptography.fernet import Fernet
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database, get_session, import_models
from app.database.credentials import Credentials
from app.database.dataset_members import DatasetMembers
from app.database.datasets import Datasets
from app.database.instance_settings import InstanceSettings
from app.database.organizations import OrganizationMembers, Organizations
from app.database.users import Users
from app.routes.general.auth import router as auth_router
from app.routes.general.organizations import router as organizations_router
from app.routes.services import label_space_router
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.label_space import LabelSpaceDraft
from app.services import secrets
from app.services import settings as settings_service
from app.services.auth import get_current_user
from app.services.credentials import resolve_llm

import_models()


def _key(db, *, username=None, organization_id=None, api_key: str, model="openai/gpt-4o"):
    db.add(Credentials(kind="llm", username=username, organization_id=organization_id, model=model,
                       secret=secrets.encrypt(api_key), hint=secrets.hint(api_key)))


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_service, "_ENV_SNAPSHOT", {})
    for variable in ("LABEL_SPACE_LLM_API_KEY", "LABEL_SPACE_LLM_MODEL", "LABEL_SPACE_LLM_API_BASE"):
        monkeypatch.delenv(variable, raising=False)

    engine = create_engine(f"sqlite:///{tmp_path / 'credentials.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    db = Session()
    for username in ("ann", "bob", "sam", "solo"):
        db.add(Users(username=username, hashed_password="x"))
    lab = Organizations(name="Lab", is_default=True)
    strict = Organizations(name="Strict", allow_personal_keys=False)
    db.add_all([lab, strict])
    db.flush()
    db.add_all([
        OrganizationMembers(organization_id=lab.id, username="ann", role="admin"),
        OrganizationMembers(organization_id=lab.id, username="bob", role="member"),
        OrganizationMembers(organization_id=strict.id, username="sam", role="admin"),
    ])
    reef = Datasets(name="Reef", dataset_type="image", folder_path="/tmp/r", created_by="ann",
                    organization_id=lab.id)
    vault = Datasets(name="Vault", dataset_type="image", folder_path="/tmp/v", created_by="sam",
                     organization_id=strict.id)
    db.add_all([reef, vault])
    db.flush()
    db.add(DatasetMembers(dataset_id=vault.id, username="bob", role="annotator",
                          extra_permissions=[], denied_permissions=[], granted_by="sam"))
    db.add_all([
        InstanceSettings(key="llm_api_key", value=secrets.encrypt("sk-instance")),
        InstanceSettings(key="llm_model", value="anthropic/claude-opus-4-8"),
    ])
    _key(db, organization_id=lab.id, api_key="sk-lab")
    _key(db, organization_id=strict.id, api_key="sk-strict")
    db.commit()
    ids = {"lab": lab.id, "strict": strict.id, "reef": reef.id, "vault": vault.id}
    db.close()

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(organizations_router)
    app.include_router(label_space_router.router)
    caller = {"username": "ann"}

    def override_session():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    def override_user(session=Depends(override_session)):
        return AuthenticatedUser.from_query(session.query(Users).filter_by(username=caller["username"]).one())

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_current_user] = override_user
    client = TestClient(app)

    def resolve(username, dataset_id=None):
        session = Session()
        try:
            user = AuthenticatedUser.from_query(session.query(Users).filter_by(username=username).one())
            return resolve_llm(user, session, dataset_id)
        finally:
            session.close()

    def as_caller(username):
        caller["username"] = username
        return client

    yield {"as": as_caller, "resolve": resolve, "ids": ids, "Session": Session}
    engine.dispose()


# -- Which key a call uses -----------------------------------------------------

def test_the_organisations_key_comes_before_the_instances(ctx):
    config = ctx["resolve"]("bob")
    assert (config.source, config.api_key) == ("organization", "sk-lab")


def test_a_personal_key_comes_first(ctx):
    session = ctx["Session"]()
    _key(session, username="bob", api_key="sk-bob", model="anthropic/claude-opus-4-8")
    session.commit()
    session.close()

    config = ctx["resolve"]("bob")
    assert (config.source, config.api_key, config.model) == ("personal", "sk-bob", "anthropic/claude-opus-4-8")


def test_without_an_organisation_the_instances_key_is_used(ctx):
    config = ctx["resolve"]("solo")
    assert (config.source, config.api_key, config.model) == ("instance", "sk-instance",
                                                             "anthropic/claude-opus-4-8")


def test_with_no_key_anywhere_there_is_nothing_to_use(ctx):
    session = ctx["Session"]()
    session.query(InstanceSettings).delete()
    session.commit()
    session.close()

    assert ctx["resolve"]("solo") is None


def test_an_organisation_can_switch_personal_keys_off(ctx):
    session = ctx["Session"]()
    _key(session, username="sam", api_key="sk-sam")
    session.commit()
    session.close()

    assert ctx["resolve"]("sam").api_key == "sk-strict"


def test_work_on_a_dataset_follows_the_datasets_organisation(ctx):
    session = ctx["Session"]()
    _key(session, username="bob", api_key="sk-bob")
    session.commit()
    session.close()

    # Bob's own work stays on his key; work on Strict's dataset uses Strict's.
    assert ctx["resolve"]("bob").api_key == "sk-bob"
    assert ctx["resolve"]("bob", ctx["ids"]["vault"]).api_key == "sk-strict"


def test_an_unreadable_key_is_passed_over(ctx, monkeypatch):
    session = ctx["Session"]()
    _key(session, username="bob", api_key="sk-bob")
    session.commit()
    session.close()
    # Re-key: bob's personal key was written under the old key only.
    old = Fernet.generate_key()
    monkeypatch.setenv("IQUANA_SECRETS_KEY", old.decode())
    secrets.reset_key_cache()
    session = ctx["Session"]()
    lab_key = session.query(Credentials).filter_by(organization_id=ctx["ids"]["lab"]).one()
    lab_key.secret = secrets.encrypt("sk-lab")
    session.commit()
    session.close()

    assert ctx["resolve"]("bob").api_key == "sk-lab"


# -- Managing personal keys ----------------------------------------------------

def test_a_personal_key_is_stored_encrypted_and_never_returned(ctx):
    client = ctx["as"]("bob")
    response = client.put("/auth/credentials/llm",
                          json={"model": "openai/gpt-4o", "api_key": "sk-bob-secret-1234"})

    assert response.status_code == 200
    assert response.json()["credential"]["hint"] == "…1234"
    assert "sk-bob-secret-1234" not in response.text
    assert "sk-bob-secret-1234" not in client.get("/auth/credentials").text

    session = ctx["Session"]()
    stored = session.query(Credentials).filter_by(username="bob").one().secret
    assert secrets.is_encrypted(stored) and "sk-bob-secret-1234" not in stored
    session.close()


def test_a_new_key_needs_the_key(ctx):
    response = ctx["as"]("bob").put("/auth/credentials/llm", json={"model": "openai/gpt-4o"})
    assert response.status_code == 422


def test_the_model_can_change_without_resending_the_key(ctx):
    client = ctx["as"]("bob")
    client.put("/auth/credentials/llm", json={"model": "openai/gpt-4o", "api_key": "sk-bob"})
    response = client.put("/auth/credentials/llm", json={"model": "openai/gpt-4.1"})

    assert response.status_code == 200
    config = ctx["resolve"]("bob")
    assert (config.model, config.api_key) == ("openai/gpt-4.1", "sk-bob")


def test_a_new_base_url_needs_the_key_again(ctx):
    """Else whoever may edit the key could send it to a server of their own."""
    client = ctx["as"]("bob")
    client.put("/auth/credentials/llm", json={"model": "openai/gpt-4o", "api_key": "sk-bob"})

    moved = client.put("/auth/credentials/llm",
                       json={"model": "openai/gpt-4o", "api_base": "https://collector.example"})
    assert moved.status_code == 422
    assert ctx["resolve"]("bob").api_base is None


def test_the_model_must_name_its_provider(ctx):
    response = ctx["as"]("bob").put("/auth/credentials/llm", json={"model": "gpt-4o", "api_key": "sk"})
    assert response.status_code == 422


def test_a_personal_key_can_be_removed(ctx):
    client = ctx["as"]("bob")
    client.put("/auth/credentials/llm", json={"model": "openai/gpt-4o", "api_key": "sk-bob"})

    assert client.delete("/auth/credentials/llm").status_code == 200
    assert client.delete("/auth/credentials/llm").status_code == 404
    assert ctx["resolve"]("bob").source == "organization"


# -- Managing an organisation's key --------------------------------------------

def test_only_organisation_admins_set_its_key(ctx):
    lab = ctx["ids"]["lab"]
    body = {"model": "openai/gpt-4o", "api_key": "sk-lab-new"}

    assert ctx["as"]("bob").put(f"/organizations/{lab}/credentials/llm", json=body).status_code == 403
    assert ctx["as"]("ann").put(f"/organizations/{lab}/credentials/llm", json=body).status_code == 200
    assert ctx["resolve"]("bob").api_key == "sk-lab-new"


def test_organisation_admins_decide_on_personal_keys(ctx):
    lab = ctx["ids"]["lab"]
    session = ctx["Session"]()
    _key(session, username="bob", api_key="sk-bob")
    session.commit()
    session.close()

    assert ctx["as"]("bob").patch(f"/organizations/{lab}", json={"allow_personal_keys": False}).status_code == 403
    assert ctx["as"]("ann").patch(f"/organizations/{lab}", json={"allow_personal_keys": False}).status_code == 200
    assert ctx["resolve"]("bob").api_key == "sk-lab"


# -- The label-space assistant -------------------------------------------------

def test_the_assistant_reports_whose_key_it_would_use(ctx):
    assert ctx["as"]("bob").get("/label_space/config").json() == {
        "enabled": True, "model": "openai/gpt-4o", "source": "organization"}


def test_the_assistant_calls_with_the_resolved_key_and_records_its_use(ctx, monkeypatch):
    sent = {}

    class FakeCompletions:
        def create(self, **kwargs):
            sent.update(kwargs)
            return LabelSpaceDraft.model_validate({"labels": [{"name": "Coral"}]})

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(label_space_router.service, "_client", lambda: FakeClient())

    response = ctx["as"]("bob").post("/label_space/generate", json={"description": "reef photos"})

    assert response.status_code == 200
    assert (sent["api_key"], sent["model"]) == ("sk-lab", "openai/gpt-4o")
    session = ctx["Session"]()
    assert session.query(Credentials).filter_by(organization_id=ctx["ids"]["lab"]).one().last_used_at
    session.close()


def test_drafting_for_a_dataset_needs_the_right_to_change_its_labels(ctx):
    response = ctx["as"]("bob").post("/label_space/generate",
                                     json={"description": "vault", "dataset_id": ctx["ids"]["vault"]})
    assert response.status_code == 403  # an annotator cannot manage labels

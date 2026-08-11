"""Tests for user telemetry and study logging.

Covers, against a temp-file SQLite database (same pattern as the other tests):
  * the environment lock beating any runtime override,
  * per-component gating of both backend emits and client ingest,
  * the ingest endpoint: username stamped from the token, batch cap, payload cap,
    de-duplication of a replayed flush,
  * admin-only access to config, export and purge,
  * the JSONL and CSV export formats,
  * the middleware being installed only when telemetry is enabled.
"""
import csv
import io
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config as app_config
from app.database import database
import app.database.telemetry_settings  # noqa: F401  (registers the table)
import app.database.user_events  # noqa: F401
import app.database.users  # noqa: F401
from app.database.telemetry_settings import SETTINGS_ROW_ID, TelemetrySettings
from app.database.user_events import UserEvents
from app.database.users import Users
from app.schemas.permissions import GLOBAL_PERMISSIONS, GlobalRole, Permission
from app.services.telemetry import config as telemetry_config
from app.services.telemetry.config import (
    TelemetryComponent,
    format_components,
    parse_components,
)


@pytest.fixture
def db_session(tmp_path, monkeypatch):
    """An isolated database, with the telemetry modules pointed at it."""
    engine = create_engine(f"sqlite:///{tmp_path / 'telemetry.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    # The recorder opens its own sessions, so it has to be redirected too.
    import app.services.telemetry.recorder as recorder_module
    monkeypatch.setattr(recorder_module, "SessionLocal", Session)

    telemetry_config.invalidate_cache()
    yield Session
    telemetry_config.invalidate_cache()


@pytest.fixture
def enabled(monkeypatch, db_session):
    """Telemetry fully on: every component captured."""
    monkeypatch.setattr(app_config, "USER_EVENTS_ENABLED", True)
    monkeypatch.setattr(app_config, "USER_EVENTS_CAPTURE", True)
    monkeypatch.setattr(app_config, "USER_EVENTS_COMPONENTS", "annotation,ai,navigation,api")
    telemetry_config.invalidate_cache()
    return db_session


@pytest.fixture
def recorder(monkeypatch):
    """The process recorder, writing inline so assertions never race the drain thread."""
    from app.services.telemetry.recorder import recorder as instance
    monkeypatch.setattr(instance, "synchronous", True)
    return instance


# -- Configuration ---------------------------------------------------------

def test_component_parsing_ignores_unknown_names():
    parsed = parse_components("annotation, ai ,nonsense,")
    assert parsed == {TelemetryComponent.ANNOTATION, TelemetryComponent.AI}
    assert format_components(parsed) == "ai,annotation"


def test_disabled_by_default(db_session, monkeypatch):
    monkeypatch.setattr(app_config, "USER_EVENTS_ENABLED", False)
    telemetry_config.invalidate_cache()

    resolved = telemetry_config.get_config(db_session())
    assert resolved.enabled is False
    assert resolved.captures(TelemetryComponent.ANNOTATION) is False


def test_env_lock_beats_stored_override(db_session, monkeypatch):
    """A stored row must never be able to switch capture on."""
    session = db_session()
    session.add(TelemetrySettings(id=SETTINGS_ROW_ID, capture_enabled=True,
                                  components="annotation,ai,navigation,api"))
    session.commit()

    monkeypatch.setattr(app_config, "USER_EVENTS_ENABLED", False)
    telemetry_config.invalidate_cache()

    resolved = telemetry_config.get_config(session)
    assert resolved.enabled is False
    assert resolved.captures(TelemetryComponent.AI) is False


def test_runtime_override_persists_and_narrows(enabled):
    session = enabled()
    resolved = telemetry_config.save_config(
        session,
        capture_enabled=True,
        components=frozenset({TelemetryComponent.AI}),
        updated_by="root",
    )
    assert resolved.captures(TelemetryComponent.AI) is True
    assert resolved.captures(TelemetryComponent.ANNOTATION) is False

    # A fresh resolution (cache dropped) reads the same answer back from the row.
    telemetry_config.invalidate_cache()
    reread = telemetry_config.get_config(session)
    assert reread.from_runtime_override is True
    assert reread.captures(TelemetryComponent.AI) is True
    assert reread.captures(TelemetryComponent.ANNOTATION) is False


def test_capture_off_records_nothing(enabled, monkeypatch):
    monkeypatch.setattr(app_config, "USER_EVENTS_CAPTURE", False)
    telemetry_config.invalidate_cache()

    resolved = telemetry_config.get_config(enabled())
    assert resolved.enabled is True          # the deployment allows it
    assert resolved.capture_enabled is False
    for component in TelemetryComponent:
        assert resolved.captures(component) is False


def test_telemetry_manage_is_global_and_admin_only():
    assert Permission.TELEMETRY_MANAGE in GLOBAL_PERMISSIONS
    from app.schemas.permissions import GLOBAL_ROLE_PERMISSIONS
    assert Permission.TELEMETRY_MANAGE in GLOBAL_ROLE_PERMISSIONS[GlobalRole.ADMIN]
    assert Permission.TELEMETRY_MANAGE not in GLOBAL_ROLE_PERMISSIONS[GlobalRole.MEMBER]
    assert Permission.TELEMETRY_MANAGE not in GLOBAL_ROLE_PERMISSIONS[GlobalRole.GUEST]


# -- Recorder --------------------------------------------------------------

def test_recorder_writes_enabled_component(enabled, recorder):
    assert recorder.record(TelemetryComponent.ANNOTATION, "tool.switch",
                           username="ann", payload={"from": "pan", "to": "box"}) is True

    session = enabled()
    row = session.query(UserEvents).one()
    assert row.event_type == "tool.switch"
    assert row.username == "ann"
    assert row.source == "backend"
    assert json.loads(row.payload) == {"from": "pan", "to": "box"}


def test_recorder_drops_disabled_component(enabled, recorder):
    telemetry_config.save_config(enabled(), capture_enabled=True,
                                 components=frozenset({TelemetryComponent.AI}),
                                 updated_by="root")

    assert recorder.record(TelemetryComponent.ANNOTATION, "tool.switch") is False
    assert recorder.record(TelemetryComponent.AI, "ai.prompted.invoke") is True

    assert enabled().query(UserEvents).count() == 1


def test_oversized_payload_is_replaced_not_truncated(enabled, recorder):
    """A sliced JSON string would be unparseable at analysis time."""
    recorder.record(TelemetryComponent.ANNOTATION, "big",
                    payload={"blob": "x" * (app_config.USER_EVENTS_MAX_PAYLOAD_BYTES + 100)})

    stored = json.loads(enabled().query(UserEvents).one().payload)
    assert stored["_error"] == "payload_too_large"


def test_track_duration_records_failures(enabled, recorder):
    from app.services.telemetry.emit import track_duration

    with pytest.raises(ValueError):
        with track_duration(TelemetryComponent.AI, "ai.prompted.invoke",
                            username="ann") as span:
            span["model"] = "sam3"
            raise ValueError("model exploded")

    row = enabled().query(UserEvents).one()
    payload = json.loads(row.payload)
    assert payload["ok"] is False
    assert payload["error"] == "ValueError"
    assert payload["model"] == "sam3"
    assert row.duration_ms is not None


# -- Routes ----------------------------------------------------------------

@pytest.fixture
def client(enabled, recorder, monkeypatch):
    """An app with only the telemetry router, on the test database."""
    from app.database import get_session
    from app.routes.general.telemetry import router
    from app.services.auth import create_access_token

    Session = enabled
    session = Session()
    session.add_all([
        Users(username="root", hashed_password="x", global_role=GlobalRole.ADMIN.value),
        Users(username="ann", hashed_password="x", global_role=GlobalRole.MEMBER.value),
    ])
    session.commit()
    session.close()

    def override_session():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_session] = override_session

    test_client = TestClient(app)
    test_client.tokens = {
        name: {"Authorization": f"Bearer {create_access_token(data={'sub': name})}"}
        for name in ("root", "ann")
    }
    return test_client


def _event(event_id, component="annotation", event_type="tool.switch", **extra):
    return {
        "event_id": event_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "event_type": event_type,
        "session_id": "sess-1",
        **extra,
    }


def test_config_is_readable_without_auth(client):
    """The client must be able to decide whether to instrument before login."""
    body = client.get("/telemetry/config").json()
    assert body["config"]["enabled"] is True
    assert body["config"]["components"]["annotation"] is True


def test_ingest_stamps_username_from_token(client, enabled):
    response = client.post("/telemetry/events", json={"events": [_event("e1")]},
                           headers=client.tokens["ann"])
    assert response.json() == {"accepted": 1, "dropped": 0}
    assert enabled().query(UserEvents).one().username == "ann"


def test_ingest_accepts_unauthenticated_events(client, enabled):
    """sendBeacon cannot set an Authorization header; those events still count."""
    response = client.post("/telemetry/events", json={"events": [_event("e1")]})
    assert response.json()["accepted"] == 1

    row = enabled().query(UserEvents).one()
    assert row.username is None
    assert row.source == "frontend"


def test_ingest_ignores_a_client_supplied_username(client, enabled):
    """A batch must not be able to attribute events to someone else."""
    event = _event("e1")
    event["username"] = "root"
    client.post("/telemetry/events", json={"events": [event]},
                headers=client.tokens["ann"])

    assert enabled().query(UserEvents).one().username == "ann"


def test_ingest_drops_disabled_components(client, enabled):
    telemetry_config.save_config(enabled(), capture_enabled=True,
                                 components=frozenset({TelemetryComponent.AI}),
                                 updated_by="root")

    body = client.post("/telemetry/events", json={"events": [
        _event("e1", component="annotation"),
        _event("e2", component="ai", event_type="ai.suggestion.accept"),
    ]}).json()

    assert body == {"accepted": 1, "dropped": 1}
    assert enabled().query(UserEvents).one().component == "ai"


def test_ingest_rejects_an_oversized_batch(client):
    events = [_event(f"e{i}") for i in range(app_config.USER_EVENTS_MAX_BATCH + 1)]
    assert client.post("/telemetry/events", json={"events": events}).status_code == 413


def test_replayed_batch_is_deduplicated(client, enabled):
    batch = {"events": [_event("e1"), _event("e2")]}
    client.post("/telemetry/events", json=batch)
    client.post("/telemetry/events", json=batch)

    assert enabled().query(UserEvents).count() == 2


def test_config_update_requires_admin(client):
    payload = {"capture_enabled": False}
    assert client.put("/telemetry/config", json=payload,
                      headers=client.tokens["ann"]).status_code == 403
    assert client.put("/telemetry/config", json=payload,
                      headers=client.tokens["root"]).status_code == 200


def test_admin_can_switch_a_component_off_at_runtime(client, enabled):
    client.put("/telemetry/config",
               json={"capture_enabled": True, "components": ["ai"]},
               headers=client.tokens["root"])

    body = client.post("/telemetry/events",
                       json={"events": [_event("e1", component="annotation")]}).json()
    assert body == {"accepted": 0, "dropped": 1}
    assert enabled().query(UserEvents).count() == 0


def test_export_requires_admin(client):
    assert client.get("/telemetry/export",
                      headers=client.tokens["ann"]).status_code == 403


def test_jsonl_export_inlines_the_payload(client):
    client.post("/telemetry/events", json={"events": [
        _event("e1", payload={"from": "pan", "to": "box"}),
    ]}, headers=client.tokens["ann"])

    response = client.get("/telemetry/export?format=jsonl",
                          headers=client.tokens["root"])
    assert response.status_code == 200

    rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["username"] == "ann"
    # An object, not a JSON-encoded string, so the file loads straight into pandas.
    assert rows[0]["payload"] == {"from": "pan", "to": "box"}


def test_csv_export_has_a_header_and_one_row_per_event(client):
    client.post("/telemetry/events", json={"events": [_event("e1"), _event("e2")]})

    response = client.get("/telemetry/export?format=csv", headers=client.tokens["root"])
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 2
    assert {row["event_id"] for row in rows} == {"e1", "e2"}


def test_export_filters_by_session(client):
    client.post("/telemetry/events", json={"events": [
        _event("e1"),
        _event("e2", session_id="other"),
    ]})

    response = client.get("/telemetry/export?format=jsonl&session_id=sess-1",
                          headers=client.tokens["root"])
    rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert [row["event_id"] for row in rows] == ["e1"]


def test_unfiltered_purge_needs_confirmation(client, enabled):
    client.post("/telemetry/events", json={"events": [_event("e1")]})

    assert client.request("DELETE", "/telemetry/events",
                          headers=client.tokens["root"]).status_code == 400
    assert enabled().query(UserEvents).count() == 1

    body = client.request("DELETE", "/telemetry/events?confirm=true",
                          headers=client.tokens["root"]).json()
    assert body["deleted"] == 1
    assert enabled().query(UserEvents).count() == 0


def test_purge_can_target_one_session(client, enabled):
    """A participant withdrawing consent must be removable without a full wipe."""
    client.post("/telemetry/events", json={"events": [
        _event("e1"),
        _event("e2", session_id="other"),
    ]})

    client.request("DELETE", "/telemetry/events?session_id=sess-1",
                   headers=client.tokens["root"])

    remaining = enabled().query(UserEvents).all()
    assert [row.event_id for row in remaining] == ["e2"]


def test_sessions_index_summarises_each_run(client):
    client.post("/telemetry/events", json={"events": [_event("e1"), _event("e2")]},
                headers=client.tokens["ann"])

    body = client.get("/telemetry/sessions", headers=client.tokens["root"]).json()
    assert len(body["sessions"]) == 1
    assert body["sessions"][0]["session_id"] == "sess-1"
    assert body["sessions"][0]["username"] == "ann"
    assert body["sessions"][0]["event_count"] == 2


def test_events_listing_filters_by_time(client):
    past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    client.post("/telemetry/events", json={"events": [
        _event("old", ts=past),
        _event("new"),
    ]})

    # Encoded, because a bare "+" in a query string decodes to a space.
    cutoff = quote((datetime.now(timezone.utc) - timedelta(days=1)).isoformat())
    body = client.get(f"/telemetry/events?start={cutoff}",
                      headers=client.tokens["root"]).json()
    assert [event["event_id"] for event in body["events"]] == ["new"]


# -- App wiring ------------------------------------------------------------

def test_middleware_and_routes_only_exist_when_enabled(monkeypatch, tmp_path):
    """The environment lock removes the surface entirely, not just its answers."""
    from app.services.telemetry.middleware import TelemetryMiddleware

    def build(enabled_flag):
        monkeypatch.setattr(app_config, "USER_EVENTS_ENABLED", enabled_flag)
        telemetry_config.invalidate_cache()
        import app as app_package
        return app_package.create_app()

    off = build(False)
    assert not any(m.cls is TelemetryMiddleware for m in off.user_middleware)
    assert not [p for p in off.openapi()["paths"] if p.startswith("/telemetry")]

    on = build(True)
    assert any(m.cls is TelemetryMiddleware for m in on.user_middleware)
    assert sorted(p for p in on.openapi()["paths"] if p.startswith("/telemetry")) == [
        "/telemetry/config", "/telemetry/events", "/telemetry/export", "/telemetry/sessions",
    ]


# -- Session propagation ---------------------------------------------------

def test_middleware_reads_the_session_header(enabled, recorder, monkeypatch):
    """Backend-emitted events must join the session the client is already in.

    Without this the request-timing and AI-latency rows carry a null session_id
    and cannot be tied to the participant's timeline, which was the whole point
    of capturing them.
    """
    from app.services.telemetry.middleware import TelemetryMiddleware

    Session = enabled
    app = FastAPI()
    app.add_middleware(TelemetryMiddleware)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    client = TestClient(app)
    client.get("/ping", headers={"X-Telemetry-Session": "sess-header"})
    client.get("/ping")  # no header: still recorded, just unattributed
    # A CORS preflight is the browser asking permission to send those very
    # headers, so it can never carry them; it must not become an event.
    client.options("/ping")

    session = Session()
    rows = session.query(UserEvents).filter_by(event_type="api.request").order_by(
        UserEvents.id).all()
    assert [row.session_id for row in rows] == ["sess-header", None]
    session.close()


def test_websocket_endpoint_accepts_a_telemetry_session_query_param():
    """A handshake cannot carry headers, so the id arrives as a query parameter."""
    import inspect
    from app.routes.websockets.image_annotation_session import websocket_endpoint

    parameter = inspect.signature(websocket_endpoint).parameters["telemetry_session"]
    assert parameter.default is None, "must stay optional for non-study deployments"


def test_ai_operations_forward_the_session_to_their_latency_event():
    """The AI spans are the rows a study most needs joined to a session."""
    import inspect
    from app.services.annotation_session import operations

    for name in ("run_prompted_segmentation", "run_semantic_segmentation",
                 "run_instance_segmentation", "run_suggestion_segmentation"):
        parameters = inspect.signature(getattr(operations, name)).parameters
        assert "session_id" in parameters, f"{name} cannot tag its latency event"
        assert parameters["session_id"].default is None


def test_sessions_are_one_row_even_when_some_events_lack_a_username(client, enabled):
    """A beacon flush has no Authorization header, so its rows have no username.

    Grouping on username as well as session split one participant's session into
    a named row plus a phantom anonymous one, which read as two sessions in the
    admin UI.
    """
    Session = enabled
    session = Session()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    session.add_all([
        UserEvents(event_id="s-1", ts=now, received_at=now, username="ann",
                   session_id="sess-x", component="annotation",
                   event_type="tool.switch", source="frontend"),
        # The unload flush: same session, no username.
        UserEvents(event_id="s-2", ts=now, received_at=now, username=None,
                   session_id="sess-x", component="navigation",
                   event_type="app.hidden", source="frontend"),
    ])
    session.commit()
    session.close()

    body = client.get("/telemetry/sessions", headers=client.tokens["root"]).json()
    rows = [s for s in body["sessions"] if s["session_id"] == "sess-x"]
    assert len(rows) == 1, "one login must be one session row"
    assert rows[0]["username"] == "ann", "the known username should win over the null"
    assert rows[0]["event_count"] == 2


def test_purge_scoped_by_component_leaves_other_components_alone(client, enabled):
    """A component filter alone must count as a scope, not fall through to a wipe.

    The purge endpoint used to only recognise `session_id`/`username` as scoping
    filters; a request narrowed by component or date range alone matched neither,
    so the "is this scoped?" check said no and the request proceeded to delete
    every row. A destructive endpoint silently doing more than its filters say is
    exactly the failure mode a study run cannot afford.
    """
    client.post("/telemetry/events", json={"events": [
        _event("e1", component="annotation"),
        _event("e2", component="ai"),
    ]})

    response = client.request("DELETE", "/telemetry/events?component=annotation",
                              headers=client.tokens["root"])
    assert response.status_code == 200, "a component filter alone must not need confirm=true"
    assert response.json()["deleted"] == 1

    remaining = enabled().query(UserEvents).all()
    assert [row.event_id for row in remaining] == ["e2"]


def test_purge_scoped_by_date_range_leaves_events_outside_it(client, enabled):
    two_days_ago = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    one_day_ago = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None).isoformat()
    client.post("/telemetry/events", json={"events": [_event("old", ts=two_days_ago)]})
    client.post("/telemetry/events", json={"events": [_event("new")]})

    response = client.request("DELETE", f"/telemetry/events?end={one_day_ago}",
                              headers=client.tokens["root"])
    assert response.status_code == 200
    assert response.json()["deleted"] == 1

    remaining = enabled().query(UserEvents).all()
    assert [row.event_id for row in remaining] == ["new"]

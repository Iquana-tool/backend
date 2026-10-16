"""Tests for the per-user usage statistics behind the admin Usage view."""
import json
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database
from app.database.ai_suggestions import AiSuggestions
from app.database.user_events import UserEvents
from app.services import usage_stats
from app.services.activity_log.config import ActivityLogConfig

NOW = datetime(2026, 10, 15, 12, 0, 0)
SQUARE = ([0.1, 0.2, 0.2, 0.1], [0.1, 0.1, 0.2, 0.2])


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'usage.db'}")
    database.metadata.create_all(engine, tables=[AiSuggestions.__table__, UserEvents.__table__])
    monkeypatch.setattr(usage_stats, "get_config",
                        lambda db=None: ActivityLogConfig(enabled=True, capture_enabled=True))
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def suggestion(db, username, *, model="sam2", source="prompted", edited=False, deleted=False,
               superseded=False, iou=None, created=NOW):
    db.add(AiSuggestions(run_id=uuid4().hex, source=source, model_key=model, username=username,
                         dataset_id=1, image_id=1, x=list(SQUARE[0]), y=list(SQUARE[1]),
                         created_at=created, edited_at=created if edited else None,
                         deleted_at=created if deleted else None,
                         superseded_at=created if superseded else None, final_iou=iou))
    db.commit()


def event(db, username, event_type, payload=None, *, ts=NOW, image_id=None, duration_ms=None,
          session_id="s1"):
    db.add(UserEvents(event_id=uuid4().hex, ts=ts, received_at=ts, username=username,
                      session_id=session_id, component="annotation", event_type=event_type,
                      source="frontend", image_id=image_id, duration_ms=duration_ms,
                      payload=json.dumps(payload) if payload is not None else None))
    db.commit()


@pytest.mark.parametrize("path, workflow", [
    ("/dataset/3/annotate/17", "Annotation"),
    ("/dataset/3/quantifications/image/2", "Quantification"),
    ("/dataset/3/model-orchestration", "Model orchestration"),
    ("/datasets", "Datasets overview"),
    ("/", "Home"),
    ("/something-new", "Other"),
    (None, "Other"),
])
def test_paths_map_to_workflows(path, workflow):
    assert usage_stats.workflow_for_path(path) == workflow


def test_overview_counts_outcomes_and_activity_per_user(db):
    suggestion(db, "ann")
    suggestion(db, "ann", edited=True, iou=0.9)
    suggestion(db, "ann", edited=True, iou=0.7)
    suggestion(db, "ann", deleted=True)
    suggestion(db, "ann", superseded=True)  # replaced by a refinement: not counted
    suggestion(db, "bob")
    event(db, "ann", "image.visit", {"active_ms": 4000}, image_id=1)
    event(db, "ann", "image.visit", {"active_ms": 6000}, image_id=2, session_id="s2")
    event(db, "carl", "tool.switch", {"to": "brush"})

    result = usage_stats.list_user_usage(db, since=None)

    users = {u["username"]: u for u in result["users"]}
    assert result["events_enabled"] is True
    assert set(users) == {"ann", "bob", "carl"}
    ann = users["ann"]
    assert (ann["suggestions"], ann["as_is"], ann["edited"], ann["rejected"]) == (4, 1, 2, 1)
    assert ann["mean_edit_iou"] == pytest.approx(0.8)
    assert (ann["events"], ann["sessions"], ann["images_visited"], ann["active_ms"]) == (2, 2, 2, 10000)
    assert users["carl"]["suggestions"] == 0 and users["carl"]["events"] == 1


def test_overview_respects_the_time_window(db):
    suggestion(db, "ann", created=NOW - timedelta(days=40))
    event(db, "ann", "tool.switch", {"to": "brush"}, ts=NOW - timedelta(days=40))
    suggestion(db, "bob")

    result = usage_stats.list_user_usage(db, since=NOW - timedelta(days=30))

    assert [u["username"] for u in result["users"]] == ["bob"]


def test_detail_summarises_one_user(db):
    suggestion(db, "ann", model="sam2", edited=True, iou=0.97)
    suggestion(db, "ann", model="sam2", edited=True, iou=0.4)
    suggestion(db, "ann", model="yolo", source="instance_suggestion")
    suggestion(db, "bob", model="sam2", edited=True, iou=0.1)  # someone else's
    day2 = NOW + timedelta(days=1)
    event(db, "ann", "route.change", {"to": "/dataset/1/annotate/1", "from": "/datasets",
                                      "dwell_ms": 1000})
    event(db, "ann", "route.change", {"to": "/dataset/1/training", "from": "/dataset/1/annotate/1",
                                      "dwell_ms": 60000})
    event(db, "ann", "route.change", {"to": "/datasets", "from": "/dataset/1/training",
                                      "dwell_ms": 5000}, ts=day2)
    event(db, "ann", "tool.switch", {"from": "select", "to": "brush"})
    event(db, "ann", "tool.switch", {"from": "brush", "to": "prompt"})
    event(db, "ann", "tool.switch", {"from": "prompt", "to": "brush"})
    event(db, "ann", "ai.prompted.invoke", {"model": "sam2", "ok": True}, duration_ms=300)
    event(db, "ann", "ai.prompted.invoke", {"model": "sam2", "ok": False}, duration_ms=500)
    event(db, "ann", "ai.model.select", {"tool": "prompted", "model": "sam2"})
    event(db, "ann", "image.visit", {"active_ms": 4000, "idle_ms": 1000, "ai_wait_ms": 500,
                                     "open_ms": 5500, "interactions": {"edits": 2, "prompts": 3}},
          image_id=1)
    event(db, "ann", "image.visit", {"active_ms": 8000, "idle_ms": 0, "ai_wait_ms": 0,
                                     "open_ms": 8000, "interactions": {"edits": 1}},
          image_id=1, ts=day2)

    detail = usage_stats.get_user_usage(db, "ann", since=None)

    models = {m["model_key"]: m for m in detail["suggestions"]}
    assert models["sam2"]["edited"] == 2
    assert models["sam2"]["mean_edit_iou"] == pytest.approx(0.685)
    assert models["yolo"]["as_is"] == 1 and models["yolo"]["mean_edit_iou"] is None
    buckets = {b["bucket"]: b["count"] for b in detail["iou_distribution"]}
    assert sum(buckets.values()) == 2

    workflows = {w["workflow"]: w for w in detail["workflows"]}
    assert detail["workflows"][0]["workflow"] == "Annotation"
    assert workflows["Annotation"]["time_ms"] == 60000
    assert workflows["Training"]["time_ms"] == 5000

    assert detail["tools"][0] == {"tool": "brush", "count": 2}
    assert detail["ai_features"] == [{"feature": "Prompted segmentation", "model": "sam2",
                                      "count": 2, "failures": 1, "median_ms": 400}]
    assert detail["model_selections"] == [{"tool": "prompted", "model": "sam2", "count": 1}]

    images = detail["images"]
    assert (images["visits"], images["images"], images["total_active_ms"]) == (2, 1, 12000)
    assert images["mean_active_ms"] == 6000
    assert images["interactions"]["edits"] == 3 and images["interactions"]["prompts"] == 3

    timeline = {d["date"]: d for d in detail["timeline"]}
    assert set(timeline) == {"2026-10-15", "2026-10-16"}
    assert timeline["2026-10-15"]["suggestions"] == 3
    assert timeline["2026-10-15"]["workflows"] == {"Datasets overview": 1000, "Annotation": 60000}
    assert timeline["2026-10-16"]["active_ms"] == 8000


def test_detail_summarises_scene_graph_edits(db):
    event(db, "ann", "scene_graph.relationship.add", {"relationship": "touches", "count": 2})
    event(db, "ann", "scene_graph.relationship.add", {"relationship": "touches", "count": 1})
    event(db, "ann", "scene_graph.relationship.add", {"relationship": "grows on", "count": 1})
    event(db, "ann", "scene_graph.attribute.set", {"attribute_id": 5, "count": 4})
    event(db, "ann", "scene_graph.relationship.confirm", {"count": 3})
    event(db, "bob", "scene_graph.attribute.set", {"attribute_id": 5, "count": 1})

    graph = usage_stats.get_user_usage(db, "ann", since=None)["scene_graph"]

    actions = {a["action"]: a for a in graph["actions"]}
    assert actions["relationship.add"] == {"action": "relationship.add", "count": 3, "objects": 4}
    assert actions["attribute.set"]["objects"] == 4
    assert actions["relationship.confirm"]["objects"] == 3
    assert graph["relationships"] == [{"name": "touches", "count": 3}, {"name": "grows on", "count": 1}]


def test_detail_is_empty_but_well_formed_without_data(db):
    detail = usage_stats.get_user_usage(db, "nobody", since=None)

    assert detail["suggestions"] == [] and detail["workflows"] == [] and detail["timeline"] == []
    assert detail["images"]["visits"] == 0 and detail["images"]["mean_active_ms"] is None
    assert detail["scene_graph"] == {"actions": [], "relationships": []}

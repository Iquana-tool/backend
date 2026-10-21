"""HTTP tests for the active-learning additions to the annotation-queue routes."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import get_session
from app.database.image_scores import ImageScores
from app.routes.general import annotation_queue as routes
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import DatasetRole, GlobalRole, Permission
from app.services import queue_prep
from app.services.auth import get_current_user
from tests.test_queue_prep import db, make_dataset, make_image  # noqa: F401


def _annotator(dataset_id):
    return AuthenticatedUser(
        username="ann", is_admin=False, global_role=GlobalRole.MEMBER, is_active=True,
        owned_datasets=[], accessible_datasets=[dataset_id],
        memberships={dataset_id: {"role": DatasetRole.ANNOTATOR,
                                  "permissions": {Permission.ANNOTATION_CREATE,
                                                  Permission.DATASET_READ}}},
    )


@pytest.fixture
def client(db, monkeypatch):  # noqa: F811
    ds = make_dataset(db)
    image = make_image(db, ds, "a.png")
    db.commit()

    class FakeResult:
        id = "celery-1"

    monkeypatch.setattr(queue_prep.run_prep_next, "apply_async", lambda *a, **k: FakeResult())
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: _annotator(ds.id)
    return TestClient(app), db, ds, image


def test_scoring_models_route_filters_by_dataset(client, monkeypatch):
    http, _, ds, _ = client
    monkeypatch.setattr(routes, "list_available_models", lambda task: {"result": [
        {"registry_key": "trained-a", "name": "Corals v3",
         "tags": {"trained_on_dataset_id": str(ds.id)}},
        {"registry_key": "mask2former", "tags": {}},
    ]})
    response = http.get(f"/annotation-queue/datasets/{ds.id}/scoring-models")
    assert response.status_code == 200
    assert response.json()["models"] == [{"registry_key": "trained-a", "name": "Corals v3"}]


def test_prepare_starts_a_job_then_reports_progress_and_cancels(client):
    http, _, ds, _ = client
    response = http.post(f"/annotation-queue/datasets/{ds.id}/prepare",
                         json={"strategy": "uncertainty", "model_key": "m1"})
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    job_id = body["job"]["id"]
    assert body["job"]["total"] == 1 and body["job"]["done"] == 0

    status = http.get(f"/annotation-queue/datasets/{ds.id}/prep-jobs/{job_id}").json()
    assert status["job"]["status"] == "pending"

    cancelled = http.post(f"/annotation-queue/datasets/{ds.id}/prep-jobs/{job_id}/cancel")
    assert cancelled.json()["job"]["status"] == "cancelling"


def test_prepare_is_ready_when_nothing_is_missing(client):
    http, db, ds, image = client
    db.add(ImageScores(image_id=image.id, model_key="m1", score=0.3, num_detections=1,
                       confidences=[0.6]))
    db.commit()
    response = http.post(f"/annotation-queue/datasets/{ds.id}/prepare",
                         json={"strategy": "uncertainty", "model_key": "m1"})
    assert response.json() == {"success": True, "ready": True, "job": None}


def test_prepare_with_scores_respects_the_batch_inference_switch(client):
    http, db, ds, _ = client
    ds.disabled_ai_tools = "batch_inference"
    db.commit()
    response = http.post(f"/annotation-queue/datasets/{ds.id}/prepare",
                         json={"strategy": "uncertainty", "model_key": "m1"})
    assert response.status_code == 403
    assert response.json()["detail"].startswith("Uncertainty needs Batch inference")


def test_build_with_model_returns_details(client):
    http, db, ds, image = client
    db.add(ImageScores(image_id=image.id, model_key="m1", score=0.3, num_detections=1,
                       confidences=[0.6]))
    db.commit()
    response = http.post(f"/annotation-queue/datasets/{ds.id}",
                         json={"strategy": "uncertainty", "model_key": "m1"})
    assert response.status_code == 200
    queue = response.json()["queue"]
    assert queue["model_key"] == "m1"
    assert queue["details"][str(image.id)]["reason"] == "unsure"


def test_job_of_another_dataset_is_404(client):
    http, db, ds, _ = client
    other = make_dataset(db)
    make_image(db, other, "b.png")
    db.commit()
    job = queue_prep.start_prep(db, other.id, "ann", "uncertainty", "m1", enqueue=False)
    response = http.get(f"/annotation-queue/datasets/{ds.id}/prep-jobs/{job.id}")
    assert response.status_code == 404

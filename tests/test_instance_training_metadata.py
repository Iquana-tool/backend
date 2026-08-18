import asyncio
from iquana_toolbox.schemas.database.labels import Label
from iquana_toolbox.schemas.training import InstanceSegmentationTrainingRequest

from app.services.ai_services import instance_segmentation as instance_service


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"task_id": "task-1"}


class _AsyncClient:
    def __init__(self, observed):
        self.observed = observed

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, url, **kwargs):
        self.observed.update(url=url, kwargs=kwargs)
        return _Response()


def test_start_training_forwards_dataset_name_as_query_metadata(monkeypatch):
    observed = {}
    monkeypatch.setattr(
        instance_service.httpx,
        "AsyncClient",
        lambda **_kwargs: _AsyncClient(observed),
    )
    request = InstanceSegmentationTrainingRequest(
        dataset_id=4,
        image_folder_path="/tmp/images",
        model_registry_key="mask2former",
        user_id="trainer",
        labels=[Label(id=5, dataset_id=4, name="cell", value=1)],
        annotation_file_url="/tmp/annotations.json",
        hyper_parameter={"epochs": 1},
    )

    result = asyncio.run(
        instance_service.InstanceSegmentationService().start_training(
            request,
            model_run_name="custom model",
            dataset_name="Cells dataset",
        )
    )

    assert result == {"task_id": "task-1"}
    assert observed["url"].endswith("/train")
    assert observed["kwargs"]["params"] == {
        "model_run_name": "custom model",
        "dataset_name": "Cells dataset",
    }


class _StatusResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _StatusAsyncClient:
    def __init__(self, observed, response_data):
        self.observed = observed
        self.response_data = response_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, **kwargs):
        self.observed.update(url=url, kwargs=kwargs)
        return _StatusResponse(self.response_data)


def test_get_training_task_status_preserves_full_lifecycle_payload(monkeypatch):
    observed = {}
    expected_payload = {
        "task_id": "task-abc-123",
        "state": "PENDING",
        "training_state": "starting",
        "run_id": "run-xyz-789",
        "message": "Waiting for a GPU worker...",
        "queued_at": 1718000000.0,
        "start_deadline": 1718000300.0,
        "started_at": "2026-08-18T12:00:00Z",
    }

    monkeypatch.setattr(
        instance_service.httpx,
        "AsyncClient",
        lambda **_kwargs: _StatusAsyncClient(observed, expected_payload),
    )

    service = instance_service.InstanceSegmentationService()
    result = asyncio.run(service.get_training_task_status("task-abc-123"))

    assert result == expected_payload
    assert result["training_state"] == "starting"
    assert result["message"] == "Waiting for a GPU worker..."
    assert result["queued_at"] == 1718000000.0
    assert result["start_deadline"] == 1718000300.0
    assert observed["url"].endswith("/train/task-abc-123")


def test_get_training_task_state_compatibility_wrapper_returns_string(monkeypatch):
    observed = {}
    payload = {
        "task_id": "task-abc-123",
        "state": "SUCCESS",
        "training_state": "completed",
        "run_id": "run-xyz-789",
    }

    monkeypatch.setattr(
        instance_service.httpx,
        "AsyncClient",
        lambda **_kwargs: _StatusAsyncClient(observed, payload),
    )

    service = instance_service.InstanceSegmentationService()
    result = asyncio.run(service.get_training_task_state("task-abc-123"))

    assert isinstance(result, str)
    assert result == "SUCCESS"

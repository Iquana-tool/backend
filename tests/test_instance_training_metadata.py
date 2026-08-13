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

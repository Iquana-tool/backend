from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from app.services import model_registry
from app.services.inference.contract_resolver import resolve_input_contract


def test_full_model_info_returns_trained_model_metadata_name(monkeypatch):
    requested_uris = []

    def get_model_info(uri):
        requested_uris.append(uri)
        return SimpleNamespace(
            metadata={
                "registry_key": "trained-model",
                "name": "Trained Model",
            }
        )

    monkeypatch.setattr(model_registry.mlflow.models, "get_model_info", get_model_info)
    monkeypatch.setattr(model_registry.MODEL_REGISTRY.client, "get_registered_model", lambda _key: None)

    result = model_registry._full_model_info("trained-model")

    assert requested_uris == ["models:/trained-model/latest"]
    assert result["name"] == "Trained Model"


def test_full_model_info_resolves_legacy_training_ids_to_names(monkeypatch):
    class FakeQuery:
        def __init__(self, rows):
            self.rows = rows

        def filter(self, *_conditions):
            return self

        def first(self):
            return self.rows[0] if self.rows else None

        def all(self):
            return self.rows

    dataset = SimpleNamespace(id=1, name="Cells dataset")
    labels = [
        SimpleNamespace(id=5, name="cell"),
        SimpleNamespace(id=6, name="nucleus"),
    ]

    class FakeDB:
        def query(self, model):
            rows = [dataset] if model is model_registry.Datasets else labels
            return FakeQuery(rows)

    @contextmanager
    def fake_context_session():
        yield FakeDB()

    def get_model_info(_uri):
        return SimpleNamespace(
            metadata={
                "registry_key": "legacy-model",
                "name": "Legacy Model",
                "label_ids": [5, 6],
                "tags": {"dataset_id": "1"},
            }
        )

    monkeypatch.setattr(model_registry.mlflow.models, "get_model_info", get_model_info)
    monkeypatch.setattr(model_registry, "get_context_session", fake_context_session)
    monkeypatch.setattr(model_registry.MODEL_REGISTRY.client, "get_registered_model", lambda _key: None)

    result = model_registry._full_model_info("legacy-model")

    assert result["tags"]["trained_on_dataset_id"] == "1"
    assert result["tags"]["trained_on_dataset_name"] == "Cells dataset"
    assert json.loads(result["tags"]["trained_label_names"]) == ["cell", "nucleus"]


def test_full_model_info_falls_back_when_metadata_lookup_fails(monkeypatch):
    def get_model_info(_uri):
        raise RuntimeError("model version is unavailable")

    monkeypatch.setattr(model_registry.mlflow.models, "get_model_info", get_model_info)
    monkeypatch.setattr(model_registry.MODEL_REGISTRY.client, "get_registered_model", lambda _key: None)

    assert model_registry._full_model_info("missing-model") == {
        "registry_key": "missing-model",
        "name": "missing-model",
    }


def test_full_model_info_does_not_fall_back_to_stale_contract_on_malformed_registry_tag(monkeypatch):
    stale_artifact_contract = [{"task": "instance-segmentation", "conditioning": {"kind": "none"}}]

    def get_model_info(_uri):
        return SimpleNamespace(
            metadata={
                "registry_key": "broken-model",
                "name": "Broken Model",
                "input_contracts": stale_artifact_contract,
            }
        )

    registered_model = SimpleNamespace(
        name="broken-model",
        description=None,
        tags={"input_contracts": "not-json"},
    )

    monkeypatch.setattr(model_registry.mlflow.models, "get_model_info", get_model_info)
    monkeypatch.setattr(
        model_registry.MODEL_REGISTRY.client,
        "get_registered_model",
        lambda _key: registered_model,
    )

    result = model_registry._full_model_info("broken-model")

    assert "input_contracts" not in result
    assert result["tags"]["input_contracts"] == "not-json"
    with pytest.raises(ValueError, match="Malformed input_contracts tag JSON"):
        resolve_input_contract(result, "instance-segmentation")

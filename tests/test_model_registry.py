from types import SimpleNamespace

from app.services import model_registry


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

    result = model_registry._full_model_info("trained-model")

    assert requested_uris == ["models:/trained-model/latest"]
    assert result["name"] == "Trained Model"


def test_full_model_info_falls_back_when_metadata_lookup_fails(monkeypatch):
    def get_model_info(_uri):
        raise RuntimeError("model version is unavailable")

    monkeypatch.setattr(model_registry.mlflow.models, "get_model_info", get_model_info)

    assert model_registry._full_model_info("missing-model") == {
        "registry_key": "missing-model",
        "name": "missing-model",
    }

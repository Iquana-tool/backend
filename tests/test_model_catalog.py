"""Unit tests for backend model catalog with effective contracts (Issue #27)."""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.datasets  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.users  # noqa: F401
from app.database.datasets import Datasets
from app.database.labels import Labels
from app.database.users import Users
from app.schemas.inference import ModelCatalog
from app.services.inference import planning
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.training import HyperParameter


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'catalog_test.db'}")
    database.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()

    user = Users(username="tester", hashed_password="pw")
    db.add(user)
    dataset = Datasets(name="test_ds", description="", dataset_type="image",
                       folder_path=str(tmp_path), created_by="tester")
    db.add(dataset)
    db.flush()

    label1 = Labels(dataset_id=dataset.id, name="cell", value=1)
    label2 = Labels(dataset_id=dataset.id, name="nucleus", value=2)
    db.add_all([label1, label2])
    db.commit()

    return dict(db=db, dataset=dataset, label1=label1, label2=label2)


def test_model_catalog_attaches_contracts_and_provenance(monkeypatch, db_session):
    """Catalog resolves declared contracts and legacy fallback contracts with provenance."""
    db = db_session["db"]
    dataset_id = db_session["dataset"].id

    # Model 1: Declared contract for instance-segmentation
    m1_contract = InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[HyperParameter(key="threshold", label="T", type="float", default_value=0.7)],
    )
    m1_info = {
        "registry_key": "custom-m2f",
        "name": "Custom M2F",
        "input_contracts": [m1_contract.model_dump()],
        "label_ids": [db_session["label1"].id],
    }

    # Model 2: Legacy model with no input_contracts
    m2_info = {
        "registry_key": "legacy-m2f",
        "name": "Legacy M2F",
        "label_ids": [],
    }

    def fake_list_available_models(task: str):
        if task == "instance-segmentation":
            return {"result": [m1_info, m2_info]}
        return {"result": []}

    monkeypatch.setattr(planning, "list_available_models", fake_list_available_models)
    monkeypatch.setattr(planning, "strategy_options", lambda db, ds_id: [])

    catalog: ModelCatalog = planning.model_catalog(db, dataset_id)

    assert len(catalog.models) == 2
    # Custom M2F was trained on label1 in this dataset, so it sorts first
    assert catalog.models[0].registry_key == "custom-m2f"
    assert catalog.models[0].trained_on_dataset is True
    assert catalog.models[0].provenance == "declared"
    assert catalog.models[0].input_contract.parameters[0].default_value == 0.7

    # Legacy M2F resolves legacy_default
    assert catalog.models[1].registry_key == "legacy-m2f"
    assert catalog.models[1].trained_on_dataset is False
    assert catalog.models[1].provenance == "legacy_default"
    assert catalog.models[1].input_contract.task == "instance-segmentation"


def test_model_catalog_includes_concept_text_and_none_cross_image_without_embeddings(monkeypatch, db_session):
    """Cross-image models with none/concept_text conditioning are offered even when no retrieval strategy is usable."""
    db = db_session["db"]
    dataset_id = db_session["dataset"].id

    # Model with reference_images conditioning (needs embeddings/retrieval)
    ref_contract = InputContract(
        task="cross-image-suggestion",
        conditioning=ConditioningSpec(
            kind="reference_images",
            unit="image",
            min_units=1,
            max_units=1,
            requires_complete_annotation=True,
            user_selectable_count=False,
        ),
        parameters=[],
    )
    m_ref = {
        "registry_key": "sam3-ref",
        "name": "SAM3 Reference",
        "input_contracts": [ref_contract.model_dump()],
    }

    # Model with concept_text conditioning (does not need embeddings)
    text_contract = InputContract(
        task="cross-image-suggestion",
        conditioning=ConditioningSpec(kind="concept_text", user_selectable_count=False),
        parameters=[],
    )
    m_text = {
        "registry_key": "text-model",
        "name": "Text Concept Model",
        "input_contracts": [text_contract.model_dump()],
    }

    def fake_list_available_models(task: str):
        if task == "cross-image-suggestion":
            return {"result": [m_ref, m_text]}
        return {"result": []}

    monkeypatch.setattr(planning, "list_available_models", fake_list_available_models)
    # No available retrieval strategies (cross_image_usable = False)
    strat_opt = MagicMock()
    strat_opt.model_dump.return_value = {"name": "global_scene", "available": False}
    monkeypatch.setattr(planning, "strategy_options", lambda db, ds_id: [strat_opt])

    catalog: ModelCatalog = planning.model_catalog(db, dataset_id)

    # Only m_text should be included; m_ref is excluded because embeddings are unavailable
    assert len(catalog.models) == 1
    assert catalog.models[0].registry_key == "text-model"
    assert catalog.models[0].input_contract.conditioning.kind == "concept_text"


def test_model_catalog_four_task_retrieval(monkeypatch, db_session):
    """model_catalog retrieves ready models across all four canonical routing tasks."""
    from app.schemas.inference import MODEL_ROUTING_TASKS

    db = db_session["db"]
    dataset_id = db_session["dataset"].id

    models_by_task = {
        "prompted-segmentation": [{"registry_key": "sam2", "name": "SAM2", "label_ids": []}],
        "instance-suggestion": [{"registry_key": "sam3-intra", "name": "SAM3 Intra", "label_ids": []}],
        "instance-segmentation": [{"registry_key": "m2f", "name": "Mask2Former", "label_ids": []}],
        "cross-image-suggestion": [{
            "registry_key": "text-cross",
            "name": "Text Cross",
            "label_ids": [],
            "input_contracts": [
                InputContract(
                    task="cross-image-suggestion",
                    conditioning=ConditioningSpec(kind="concept_text", user_selectable_count=False),
                    parameters=[],
                ).model_dump()
            ],
        }],
    }

    monkeypatch.setattr(planning, "list_available_models", lambda task: {"result": models_by_task.get(task, [])})
    monkeypatch.setattr(planning, "strategy_options", lambda db, ds_id: [])

    # Test default tasks parameter (MODEL_ROUTING_TASKS)
    catalog = planning.model_catalog(db, dataset_id)
    assert len(catalog.models) == 4
    tasks_in_catalog = {m.task for m in catalog.models}
    assert tasks_in_catalog == set(MODEL_ROUTING_TASKS)

    # Test explicit tasks parameter
    catalog_subset = planning.model_catalog(db, dataset_id, tasks=("prompted-segmentation", "instance-suggestion"))
    assert len(catalog_subset.models) == 2
    assert {m.task for m in catalog_subset.models} == {"prompted-segmentation", "instance-suggestion"}

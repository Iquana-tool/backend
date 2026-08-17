"""Tests for inference planning generic input normalization and contract snapshotting."""
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.contours  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.users  # noqa: F401
from app.database.datasets import Datasets
from app.database.labels import Labels
from app.database.users import Users
from app.schemas.inference import InferenceStepRequest, ModelCatalog, ModelOption
from app.services.inference.planning import resolve_steps
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.training import HyperParameter


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


@pytest.fixture
def session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()
        engine.dispose()


@pytest.fixture
def dataset(session):
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    session.flush()
    ds = Datasets(name="A", description="", dataset_type="image", folder_path="/tmp/A", created_by="u")
    session.add(ds)
    session.flush()
    lbl_cell = Labels(dataset_id=ds.id, parent_id=None, name="cell", value=1)
    session.add(lbl_cell)
    session.flush()
    lbl_nucleus = Labels(dataset_id=ds.id, parent_id=lbl_cell.id, name="nucleus", value=2)
    session.add(lbl_nucleus)
    session.flush()
    return dict(session=session, ds=ds, cell=lbl_cell, nucleus=lbl_nucleus)


def _mock_catalog(monkeypatch, options: list[ModelOption]):
    catalog = ModelCatalog(models=options, default_model_per_task={})
    monkeypatch.setattr(
        "app.services.inference.planning.model_catalog",
        lambda db, ds_id: catalog,
    )


def test_resolve_steps_with_canonical_inputs(dataset, monkeypatch):
    s = dataset["session"]
    contract = InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[
            HyperParameter(key="threshold", label="T", type="float", default_value=0.5, min_value=0.0, max_value=1.0)
        ],
    )
    option = ModelOption(
        registry_key="m2f",
        task="instance-segmentation",
        name="Mask2Former",
        version="1",
        label_ids=[dataset["cell"].id],
        is_default=True,
        input_contract=contract,
        provenance="declared",
    )
    _mock_catalog(monkeypatch, [option])

    # Valid canonical request
    req = InferenceStepRequest(
        label_id=dataset["cell"].id,
        model_registry_key="m2f",
        task="instance-segmentation",
        inputs={"parameters": {"threshold": 0.75}},
    )
    resolved = resolve_steps(s, dataset["ds"].id, [req])
    assert len(resolved) == 1
    step = resolved[0]
    assert step.inputs["parameters"]["threshold"] == 0.75
    assert step.input_contract.task == "instance-segmentation"
    assert step.provenance == "declared"


def test_canonical_inputs_ignore_stale_legacy_fields():
    req = InferenceStepRequest(
        label_id=1,
        model_registry_key="m2f",
        task="instance-segmentation",
        inputs={"parameters": {"threshold": 0.75}},
        min_confidence=50.0,
        retrieval_strategy="obsolete",
        top_k=100,
    )

    assert req.inputs["parameters"]["threshold"] == 0.75
    assert req.min_confidence == 0.0
    assert req.retrieval_strategy is None
    assert req.top_k == 5


def test_resolve_steps_with_legacy_fields_synthesizes_inputs(dataset, monkeypatch):
    s = dataset["session"]
    contract = InputContract(
        task="cross-image-suggestion",
        conditioning=ConditioningSpec(
            kind="reference_images", unit="image", min_units=1, max_units=1,
            requires_complete_annotation=True, user_selectable_count=False,
        ),
        parameters=[
            HyperParameter(key="threshold", label="T", type="float", default_value=0.3, min_value=0.0, max_value=1.0),
            HyperParameter(key="mask_threshold", label="M", type="float", default_value=0.5),
            HyperParameter(key="min_target_frac", label="F", type="float", default_value=0.5),
        ],
    )
    option = ModelOption(
        registry_key="sam3",
        task="cross-image-suggestion",
        name="SAM 3",
        version="1",
        label_ids=[],
        is_default=True,
        input_contract=contract,
        provenance="declared",
    )
    _mock_catalog(monkeypatch, [option])

    # Legacy request without `inputs`
    req = InferenceStepRequest(
        label_id=dataset["cell"].id,
        model_registry_key="sam3",
        task="cross-image-suggestion",
        retrieval_strategy="global_scene",
        top_k=1,
        min_confidence=0.4,
    )
    resolved = resolve_steps(s, dataset["ds"].id, [req])
    assert len(resolved) == 1
    step = resolved[0]
    assert step.inputs["parameters"]["threshold"] == 0.4
    assert step.inputs["parameters"]["mask_threshold"] == 0.5
    assert step.inputs["parameters"]["min_target_frac"] == 0.5
    assert step.inputs["conditioning"]["strategy"] == "global_scene"
    assert step.inputs["conditioning"]["count"] == 1


def test_resolve_steps_invalid_inputs_raises_400(dataset, monkeypatch):
    s = dataset["session"]
    contract = InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[
            HyperParameter(key="threshold", label="T", type="float", default_value=0.5, min_value=0.0, max_value=1.0)
        ],
    )
    option = ModelOption(
        registry_key="m2f",
        task="instance-segmentation",
        name="Mask2Former",
        version="1",
        label_ids=[],
        is_default=True,
        input_contract=contract,
        provenance="declared",
    )
    _mock_catalog(monkeypatch, [option])

    # Out of range threshold (1.5 > max_value 1.0)
    req = InferenceStepRequest(
        label_id=dataset["cell"].id,
        model_registry_key="m2f",
        task="instance-segmentation",
        inputs={"parameters": {"threshold": 1.5}},
    )
    with pytest.raises(HTTPException) as exc:
        resolve_steps(s, dataset["ds"].id, [req])
    assert exc.value.status_code == 400
    assert "greater than max_value" in exc.value.detail


def test_resolve_steps_legacy_top_k_default_sam3(dataset, monkeypatch):
    """Legacy cross-image request with default top_k=5 maps safely to SAM 3 single-image contract."""
    s = dataset["session"]
    contract = InputContract(
        task="cross-image-suggestion",
        conditioning=ConditioningSpec(
            kind="reference_images", unit="image", min_units=1, max_units=1,
            requires_complete_annotation=True, user_selectable_count=False,
        ),
        parameters=[
            HyperParameter(key="threshold", label="T", type="float", default_value=0.3, min_value=0.0, max_value=1.0),
            HyperParameter(key="mask_threshold", label="M", type="float", default_value=0.5),
            HyperParameter(key="min_target_frac", label="F", type="float", default_value=0.5),
        ],
    )
    option = ModelOption(
        registry_key="sam3",
        task="cross-image-suggestion",
        name="SAM 3",
        version="1",
        label_ids=[],
        is_default=True,
        input_contract=contract,
        provenance="declared",
    )
    _mock_catalog(monkeypatch, [option])

    # Default top_k=5 should not raise ValueError when user_selectable_count is False / max_units=1
    req = InferenceStepRequest(
        label_id=dataset["cell"].id,
        model_registry_key="sam3",
        task="cross-image-suggestion",
        retrieval_strategy="global_scene",
        top_k=5,
        min_confidence=0.3,
    )
    resolved = resolve_steps(s, dataset["ds"].id, [req])
    assert len(resolved) == 1
    step = resolved[0]
    assert step.inputs["conditioning"]["count"] == 1
    assert step.inputs["parameters"]["threshold"] == 0.3


def test_resolved_step_migrates_persisted_legacy_plan_step():
    """Deserializing persisted plan steps lacking contract snapshots resolves correct task defaults."""
    from app.schemas.inference import ResolvedStep

    legacy_persisted_cross_image_step = {
        "label_id": 1,
        "model_registry_key": "sam3",
        "task": "cross-image-suggestion",
        "retrieval_strategy": "global_scene",
        "top_k": 5,
        "min_confidence": 0.3,
        "level": 0,
        "label_name": "cell",
        "model_name": "SAM 3",
    }

    step = ResolvedStep.model_validate(legacy_persisted_cross_image_step)
    assert step.task == "cross-image-suggestion"
    assert step.input_contract.task == "cross-image-suggestion"
    assert step.input_contract.conditioning.kind == "reference_images"
    assert step.inputs["conditioning"]["strategy"] == "global_scene"
    assert step.inputs["conditioning"]["count"] == 1
    assert step.inputs["parameters"]["threshold"] == 0.3
    assert step.provenance == "legacy_default"

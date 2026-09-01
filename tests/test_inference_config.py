"""Unit tests for dataset model routing policy persistence and service (`app.services.inference.configuration`)."""
from unittest.mock import MagicMock
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.datasets  # noqa: F401
import app.database.dataset_model_routing_configs  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.users  # noqa: F401
from app.database.datasets import Datasets
from app.database.dataset_model_routing_configs import DatasetModelRoutingConfigs
from app.database.images import Images
from app.database.inference_jobs import InferenceJobs
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.schemas.inference import (
    DatasetModelRoutingRead,
    DatasetModelRoutingWrite,
    ModelCatalog,
    ModelOption,
    ModelRoutingBinding,
    ModelRoutingTask,
    WriteMode,
)
from app.services.inference import configuration, execution, planning
from app.services.inference.contract_resolver import LEGACY_TASK_DEFAULTS
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.training import HyperParameter


@pytest.fixture
def ctx(tmp_path):
    """SQLite DB session with FK enforcement enabled, a user, dataset, and labels."""
    engine = create_engine(f"sqlite:///{tmp_path / 'routing_config.db'}")

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    database.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()

    user = Users(username="curator", hashed_password="pwd")
    db.add(user)
    dataset = Datasets(
        name="ds1",
        description="Dataset for testing",
        dataset_type="image",
        folder_path=str(tmp_path),
        created_by="curator",
    )
    db.add(dataset)
    db.flush()

    cell = Labels(dataset_id=dataset.id, name="cell", value=1)
    db.add(cell)
    db.flush()
    nucleus = Labels(dataset_id=dataset.id, name="nucleus", value=2, parent_id=cell.id)
    db.add(nucleus)
    db.commit()

    return dict(db=db, dataset=dataset, cell=cell, nucleus=nucleus, user=user)


def _stub_multi_task_catalog(monkeypatch, label_ids=()):
    """Stub planning.model_catalog to return models for all 4 routing tasks."""
    contract_inst = LEGACY_TASK_DEFAULTS["instance-segmentation"]
    contract_cross = LEGACY_TASK_DEFAULTS["cross-image-suggestion"]
    contract_prompted = LEGACY_TASK_DEFAULTS["prompted-segmentation"]
    contract_intra = LEGACY_TASK_DEFAULTS["instance-suggestion"]

    custom_contract = InputContract(
        task="cross-image-suggestion",
        conditioning=ConditioningSpec(
            kind="embeddings",
            unit="vector",
            embedding_kinds=["image_cls"],
            user_selectable_count=True,
            default_count=5,
        ),
        parameters=[HyperParameter(key="threshold", label="T", type="float", default_value=0.5)],
    )

    models = [
        ModelOption(
            registry_key="m2f",
            name="Mask2Former",
            task="instance-segmentation",
            label_ids=list(label_ids),
            input_contract=contract_inst,
            provenance="legacy_default",
        ),
        ModelOption(
            registry_key="sam3-cross",
            name="SAM3 Cross",
            task="cross-image-suggestion",
            label_ids=list(label_ids),
            input_contract=contract_cross,
            provenance="legacy_default",
        ),
        ModelOption(
            registry_key="custom-cross",
            name="Custom Cross",
            task="cross-image-suggestion",
            label_ids=list(label_ids),
            input_contract=custom_contract,
            provenance="declared",
        ),
        ModelOption(
            registry_key="sam2-prompted",
            name="SAM2 Prompted",
            task="prompted-segmentation",
            label_ids=[],
            input_contract=contract_prompted,
            provenance="legacy_default",
        ),
        ModelOption(
            registry_key="sam3-intra",
            name="SAM3 Intra",
            task="instance-suggestion",
            label_ids=[],
            input_contract=contract_intra,
            provenance="legacy_default",
        ),
    ]

    monkeypatch.setattr(
        planning,
        "model_catalog",
        lambda db, dataset_id, tasks=None: ModelCatalog(models=models),
    )


# --------------------------------------------------------------------------- #
# Schema validation tests
# --------------------------------------------------------------------------- #

def test_schema_rejects_duplicate_selectors():
    """Each (task, label_id) pair must be unique in a policy."""
    with pytest.raises(ValidationError, match="Each \\(task, label_id\\) pair may appear at most once"):
        DatasetModelRoutingWrite(
            dataset_id=1,
            bindings=[
                ModelRoutingBinding(
                    task=ModelRoutingTask.PROMPTED_SEGMENTATION,
                    label_id=10,
                    model_registry_key="sam2",
                ),
                ModelRoutingBinding(
                    task=ModelRoutingTask.PROMPTED_SEGMENTATION,
                    label_id=10,
                    model_registry_key="sam2-alt",
                ),
            ],
        )


def test_schema_accepts_same_label_across_different_tasks():
    """The same label can be bound to different models for different tasks."""
    policy = DatasetModelRoutingWrite(
        dataset_id=1,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.PROMPTED_SEGMENTATION,
                label_id=10,
                model_registry_key="sam2",
            ),
            ModelRoutingBinding(
                task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
                label_id=10,
                model_registry_key="sam3-cross",
            ),
            ModelRoutingBinding(
                task=ModelRoutingTask.INSTANCE_SEGMENTATION,
                label_id=10,
                model_registry_key="m2f",
            ),
        ],
    )
    assert len(policy.bindings) == 3


def test_schema_accepts_one_null_default_per_task_and_rejects_duplicate_null_defaults():
    """One task default (label_id=None) is allowed per task; duplicate null defaults are rejected."""
    # Valid: defaults for different tasks
    policy = DatasetModelRoutingWrite(
        dataset_id=1,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.PROMPTED_SEGMENTATION,
                label_id=None,
                model_registry_key="sam2-default",
            ),
            ModelRoutingBinding(
                task=ModelRoutingTask.INSTANCE_SEGMENTATION,
                label_id=None,
                model_registry_key="m2f-default",
            ),
        ],
    )
    assert len(policy.bindings) == 2

    # Invalid: duplicate defaults for the same task
    with pytest.raises(ValidationError, match="Each \\(task, label_id\\) pair may appear at most once"):
        DatasetModelRoutingWrite(
            dataset_id=1,
            bindings=[
                ModelRoutingBinding(
                    task=ModelRoutingTask.PROMPTED_SEGMENTATION,
                    label_id=None,
                    model_registry_key="sam2-default",
                ),
                ModelRoutingBinding(
                    task=ModelRoutingTask.PROMPTED_SEGMENTATION,
                    label_id=None,
                    model_registry_key="sam2-other",
                ),
            ],
        )


# --------------------------------------------------------------------------- #
# CRUD and Persistence Service Tests
# --------------------------------------------------------------------------- #

def test_get_routing_policy_returns_none_when_empty(ctx):
    """An unconfigured dataset returns None."""
    assert configuration.get_routing_policy(ctx["db"], ctx["dataset"].id) is None


def test_upsert_creates_and_updates_routing_policy(monkeypatch, ctx):
    """Upsert creates a new policy, and subsequent upsert replaces bindings atomically."""
    _stub_multi_task_catalog(monkeypatch)
    db = ctx["db"]
    ds_id = ctx["dataset"].id
    cell_id = ctx["cell"].id

    initial_body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.INSTANCE_SEGMENTATION,
                label_id=None,
                model_registry_key="m2f",
            ),
            ModelRoutingBinding(
                task=ModelRoutingTask.PROMPTED_SEGMENTATION,
                label_id=cell_id,
                model_registry_key="sam2-prompted",
            ),
        ],
    )

    created = configuration.upsert_routing_policy(
        db=db,
        dataset_id=ds_id,
        username="curator",
        body=initial_body,
    )

    assert created.dataset_id == ds_id
    assert len(created.bindings) == 2
    assert created.updated_by == "curator"
    assert created.created_at is not None
    assert created.updated_at is not None

    created_at = created.created_at

    # Add editor user for FK constraint
    editor = Users(username="editor", hashed_password="pwd")
    db.add(editor)
    db.commit()

    # Update policy with new bindings
    update_body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
                label_id=cell_id,
                model_registry_key="sam3-cross",
            )
        ],
    )

    updated = configuration.upsert_routing_policy(
        db=db,
        dataset_id=ds_id,
        username="editor",
        body=update_body,
    )

    assert updated.dataset_id == ds_id
    assert len(updated.bindings) == 1
    assert updated.bindings[0].task == ModelRoutingTask.CROSS_IMAGE_SUGGESTION
    assert updated.bindings[0].model_registry_key == "sam3-cross"
    assert updated.updated_by == "editor"
    # Created timestamp preserved
    assert updated.created_at == created_at
    assert updated.updated_at >= created_at

    # Database verification
    row = db.query(DatasetModelRoutingConfigs).filter_by(dataset_id=ds_id).first()
    assert row is not None
    assert len(row.bindings) == 1
    assert row.bindings[0]["task"] == "cross-image-suggestion"


def test_database_enforces_single_policy_per_dataset(ctx):
    """Database enforces dataset_id as primary key (single row per dataset)."""
    db = ctx["db"]
    ds_id = ctx["dataset"].id

    r1 = DatasetModelRoutingConfigs(dataset_id=ds_id, bindings=[])
    db.add(r1)
    db.commit()

    engine = db.get_bind()
    with engine.connect() as conn:
        with pytest.raises(Exception):
            conn.execute(
                DatasetModelRoutingConfigs.__table__.insert().values(
                    dataset_id=ds_id, bindings=[], created_at=r1.created_at, updated_at=r1.updated_at
                )
            )


def test_upsert_mismatched_dataset_id_raises_bad_request(monkeypatch, ctx):
    """Mismatched dataset ID in body vs path raises 400."""
    _stub_multi_task_catalog(monkeypatch)
    body = DatasetModelRoutingWrite(
        dataset_id=999,
        bindings=[],
    )
    with pytest.raises(HTTPException) as exc_info:
        configuration.upsert_routing_policy(
            db=ctx["db"],
            dataset_id=ctx["dataset"].id,
            username="curator",
            body=body,
        )
    assert exc_info.value.status_code == 400


def test_upsert_nonexistent_dataset_raises_not_found(monkeypatch, ctx):
    """Upsert for non-existent dataset raises 404."""
    _stub_multi_task_catalog(monkeypatch)
    body = DatasetModelRoutingWrite(
        dataset_id=9999,
        bindings=[],
    )
    with pytest.raises(HTTPException) as exc_info:
        configuration.upsert_routing_policy(
            db=ctx["db"],
            dataset_id=9999,
            username="curator",
            body=body,
        )
    assert exc_info.value.status_code == 404


def test_upsert_invalid_label_raises_not_found(monkeypatch, ctx):
    """Binding targeting a label from another dataset or missing label raises 404."""
    _stub_multi_task_catalog(monkeypatch)
    db = ctx["db"]

    body = DatasetModelRoutingWrite(
        dataset_id=ctx["dataset"].id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.PROMPTED_SEGMENTATION,
                label_id=99999,
                model_registry_key="sam2-prompted",
            )
        ],
    )
    with pytest.raises(HTTPException) as exc_info:
        configuration.upsert_routing_policy(
            db=db,
            dataset_id=ctx["dataset"].id,
            username="curator",
            body=body,
        )
    assert exc_info.value.status_code == 404
    assert "not part of dataset" in exc_info.value.detail


def test_upsert_unregistered_or_incompatible_task_model_is_rejected(monkeypatch, ctx):
    """Binding targeting an unknown model or a model that does not serve the task raises 404."""
    _stub_multi_task_catalog(monkeypatch)
    db = ctx["db"]
    ds_id = ctx["dataset"].id

    body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.PROMPTED_SEGMENTATION,
                label_id=ctx["cell"].id,
                model_registry_key="nonexistent-model",
            )
        ],
    )
    with pytest.raises(HTTPException) as exc_info:
        configuration.upsert_routing_policy(
            db=db,
            dataset_id=ds_id,
            username="curator",
            body=body,
        )
    assert exc_info.value.status_code == 404


def test_upsert_model_not_predicting_label_is_rejected(monkeypatch, ctx):
    """Class-aware model that declares label IDs must predict the binding's label."""
    # Stub catalog where Mask2Former only predicts nucleus (not cell)
    _stub_multi_task_catalog(monkeypatch, label_ids=[ctx["nucleus"].id])
    db = ctx["db"]
    ds_id = ctx["dataset"].id

    body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.INSTANCE_SEGMENTATION,
                label_id=ctx["cell"].id,  # not in [nucleus.id]
                model_registry_key="m2f",
            )
        ],
    )
    with pytest.raises(HTTPException) as exc_info:
        configuration.upsert_routing_policy(
            db=db,
            dataset_id=ds_id,
            username="curator",
            body=body,
        )
    assert exc_info.value.status_code == 400
    assert "does not predict label 'cell'" in exc_info.value.detail


def test_upsert_normalizes_contract_inputs(monkeypatch, ctx):
    """Model inputs are normalized according to the model's input contract."""
    _stub_multi_task_catalog(monkeypatch)
    db = ctx["db"]
    ds_id = ctx["dataset"].id

    body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
                label_id=ctx["cell"].id,
                model_registry_key="custom-cross",
                inputs={
                    "conditioning": {"count": 8},
                    "parameters": {"threshold": 0.8},
                },
            )
        ],
    )

    saved = configuration.upsert_routing_policy(
        db=db,
        dataset_id=ds_id,
        username="curator",
        body=body,
    )

    assert len(saved.bindings) == 1
    binding = saved.bindings[0]
    assert binding.inputs["conditioning"]["count"] == 8
    assert binding.inputs["parameters"]["threshold"] == 0.8


def test_upsert_invalid_inputs_rejected_and_leaves_previous_policy_unchanged(monkeypatch, ctx):
    """Validation failure raises 400 and preserves previous policy atomically."""
    _stub_multi_task_catalog(monkeypatch)
    db = ctx["db"]
    ds_id = ctx["dataset"].id

    # 1. Save valid initial policy
    valid_body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.PROMPTED_SEGMENTATION,
                label_id=ctx["cell"].id,
                model_registry_key="sam2-prompted",
            )
        ],
    )
    configuration.upsert_routing_policy(db=db, dataset_id=ds_id, username="curator", body=valid_body)

    # 2. Attempt invalid update with invalid threshold parameter type
    invalid_body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
                label_id=ctx["cell"].id,
                model_registry_key="custom-cross",
                inputs={
                    "parameters": {"threshold": "not-a-number"},
                },
            )
        ],
    )

    with pytest.raises(HTTPException) as exc_info:
        configuration.upsert_routing_policy(
            db=db,
            dataset_id=ds_id,
            username="editor",
            body=invalid_body,
        )
    assert exc_info.value.status_code == 400

    # 3. Verify previous policy is intact
    current = configuration.get_routing_policy(db, ds_id)
    assert current is not None
    assert len(current.bindings) == 1
    assert current.bindings[0].task == ModelRoutingTask.PROMPTED_SEGMENTATION
    assert current.updated_by == "curator"


# --------------------------------------------------------------------------- #
# Resolution tests
# --------------------------------------------------------------------------- #

def test_resolve_routing_binding_exact_override_over_task_default(monkeypatch, ctx):
    """Exact (task, label_id) override wins over the (task, None) default."""
    _stub_multi_task_catalog(monkeypatch)
    db = ctx["db"]
    ds_id = ctx["dataset"].id
    cell_id = ctx["cell"].id

    body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.PROMPTED_SEGMENTATION,
                label_id=None,
                model_registry_key="sam2-default",
            ),
            ModelRoutingBinding(
                task=ModelRoutingTask.PROMPTED_SEGMENTATION,
                label_id=cell_id,
                model_registry_key="sam2-large-override",
            ),
        ],
    )
    # Persist directly in table
    row = DatasetModelRoutingConfigs(
        dataset_id=ds_id,
        bindings=[b.model_dump() for b in body.bindings],
    )
    db.add(row)
    db.commit()

    # Resolve with label_id -> exact match
    resolved_cell = configuration.resolve_routing_binding(
        db, ds_id, task=ModelRoutingTask.PROMPTED_SEGMENTATION, label_id=cell_id
    )
    assert resolved_cell is not None
    assert resolved_cell["model_registry_key"] == "sam2-large-override"
    assert resolved_cell["label_id"] == cell_id


def test_resolve_routing_binding_falls_back_to_task_default(monkeypatch, ctx):
    """Missing label override falls back to the (task, None) default."""
    db = ctx["db"]
    ds_id = ctx["dataset"].id
    nucleus_id = ctx["nucleus"].id

    row = DatasetModelRoutingConfigs(
        dataset_id=ds_id,
        bindings=[
            {
                "task": "prompted-segmentation",
                "label_id": None,
                "model_registry_key": "sam2-default",
            }
        ],
    )
    db.add(row)
    db.commit()

    resolved_nucleus = configuration.resolve_routing_binding(
        db, ds_id, task="prompted-segmentation", label_id=nucleus_id
    )
    assert resolved_nucleus is not None
    assert resolved_nucleus["model_registry_key"] == "sam2-default"
    assert resolved_nucleus["label_id"] is None


def test_resolve_routing_binding_returns_none_when_no_match(ctx):
    """Returns None when no exact override and no task default exists."""
    db = ctx["db"]
    ds_id = ctx["dataset"].id

    row = DatasetModelRoutingConfigs(
        dataset_id=ds_id,
        bindings=[
            {
                "task": "instance-segmentation",
                "label_id": None,
                "model_registry_key": "m2f",
            }
        ],
    )
    db.add(row)
    db.commit()

    # Querying a different task returns None
    assert (
        configuration.resolve_routing_binding(
            db, ds_id, task="prompted-segmentation", label_id=ctx["cell"].id
        )
        is None
    )


def test_clear_routing_policy(monkeypatch, ctx):
    """clear_routing_policy deletes the policy row and returns True (or False if absent)."""
    _stub_multi_task_catalog(monkeypatch)
    db = ctx["db"]
    ds_id = ctx["dataset"].id

    body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.INSTANCE_SEGMENTATION,
                label_id=None,
                model_registry_key="m2f",
            )
        ],
    )
    configuration.upsert_routing_policy(db=db, dataset_id=ds_id, username="curator", body=body)

    assert configuration.clear_routing_policy(db, ds_id) is True
    assert configuration.get_routing_policy(db, ds_id) is None
    # Deleting again returns False
    assert configuration.clear_routing_policy(db, ds_id) is False


def test_cascade_delete_dataset_removes_policy(monkeypatch, ctx):
    """Deleting a dataset cascades to remove its model routing policy."""
    _stub_multi_task_catalog(monkeypatch)
    db = ctx["db"]
    ds_id = ctx["dataset"].id

    body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.INSTANCE_SEGMENTATION,
                label_id=None,
                model_registry_key="m2f",
            )
        ],
    )
    configuration.upsert_routing_policy(db=db, dataset_id=ds_id, username="curator", body=body)

    # Delete dataset
    dataset = db.get(Datasets, ds_id)
    db.delete(dataset)
    db.commit()

    # Policy row is gone
    row = db.query(DatasetModelRoutingConfigs).filter_by(dataset_id=ds_id).first()
    assert row is None


def test_user_delete_sets_updated_by_to_null(monkeypatch, ctx):
    """Deleting a user retains the dataset policy and sets updated_by to NULL."""
    _stub_multi_task_catalog(monkeypatch)
    db = ctx["db"]
    ds_id = ctx["dataset"].id

    editor = Users(username="editor_to_delete", hashed_password="pwd")
    db.add(editor)
    db.commit()

    body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.INSTANCE_SEGMENTATION,
                label_id=None,
                model_registry_key="m2f",
            )
        ],
    )
    configuration.upsert_routing_policy(db=db, dataset_id=ds_id, username="editor_to_delete", body=body)

    # Delete editor user
    user = db.get(Users, "editor_to_delete")
    db.delete(user)
    db.commit()

    policy = configuration.get_routing_policy(db, ds_id)
    assert policy is not None
    assert policy.updated_by is None


# --------------------------------------------------------------------------- #
# Suggest Step Execution Tests
# --------------------------------------------------------------------------- #

def test_execute_suggest_step_with_task_and_mask_id(monkeypatch, ctx):
    """execute_suggest_step executes with patch mode on target mask."""
    _stub_multi_task_catalog(monkeypatch)
    db = ctx["db"]
    ds_id = ctx["dataset"].id
    cell_id = ctx["cell"].id

    # Create image and mask
    image = Images(
        dataset_id=ds_id,
        file_name="test_img.png",
        file_path="/tmp/test_img.png",
        thumbnail_file_path="/tmp/thumb.png",
        width=100,
        height=100,
    )
    db.add(image)
    db.flush()
    mask = Masks(image_id=image.id, file_path="/tmp/mask.png")
    db.add(mask)
    db.commit()

    # Save routing policy
    body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
                label_id=cell_id,
                model_registry_key="sam3-cross",
            )
        ],
    )
    configuration.upsert_routing_policy(db=db, dataset_id=ds_id, username="curator", body=body)

    # Mock run_unit
    fake_unit_result = execution.UnitResult(created=2, suppressed=0, unparented=0, contour_ids=[101, 102])
    run_unit_mock = MagicMock(return_value=fake_unit_result)
    monkeypatch.setattr(execution, "run_unit", run_unit_mock)

    result = configuration.execute_suggest_step(
        db=db,
        dataset_id=ds_id,
        image_id=image.id,
        label_id=cell_id,
        username="curator",
        task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
        mask_id=mask.id,
    )

    assert result.contours_created == 2
    assert result.contour_ids == [101, 102]
    assert run_unit_mock.called

    _, kwargs = run_unit_mock.call_args
    assert kwargs["options"].write_mode == WriteMode.PATCH
    assert kwargs["options"].preserve_reviewed is True
    assert kwargs["mask"].id == mask.id


def test_execute_suggest_step_rejects_active_batch_before_model_call(monkeypatch, ctx):
    """execute_suggest_step raises 409 when a batch job is active."""
    _stub_multi_task_catalog(monkeypatch)
    db = ctx["db"]
    ds_id = ctx["dataset"].id
    cell_id = ctx["cell"].id

    image = Images(
        dataset_id=ds_id,
        file_name="test_img.png",
        file_path="/tmp/test_img.png",
        thumbnail_file_path="/tmp/thumb.png",
        width=100,
        height=100,
    )
    db.add(image)
    db.flush()
    mask = Masks(image_id=image.id, file_path="/tmp/mask.png")
    db.add(mask)

    # Create active batch job
    job = InferenceJobs(
        dataset_id=ds_id,
        status="running",
        created_by="curator",
        plan_steps=[],
        options={},
    )
    db.add(job)
    db.commit()

    with pytest.raises(HTTPException) as exc_info:
        configuration.execute_suggest_step(
            db=db,
            dataset_id=ds_id,
            image_id=image.id,
            label_id=cell_id,
            username="curator",
            task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
            mask_id=mask.id,
        )
    assert exc_info.value.status_code == 409
    assert "currently running" in exc_info.value.detail


def test_upsert_orphaned_label_is_rejected(monkeypatch, ctx):
    """upsert_routing_policy rejects labels whose parent pointer is invalid/orphaned."""
    _stub_multi_task_catalog(monkeypatch)
    db = ctx["db"]
    ds_id = ctx["dataset"].id

    # Create another dataset with a label to provide a valid FK target in another dataset
    other_ds = Datasets(name="other_ds", description="", dataset_type="image", folder_path="/tmp", created_by="curator")
    db.add(other_ds)
    db.flush()
    foreign_label = Labels(dataset_id=other_ds.id, name="foreign", value=1)
    db.add(foreign_label)
    db.flush()

    # Create an orphaned label in ds_id whose parent is in other_ds
    orphan = Labels(dataset_id=ds_id, name="orphan_label", value=99, parent_id=foreign_label.id)
    db.add(orphan)
    db.commit()

    body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.INSTANCE_SEGMENTATION,
                label_id=orphan.id,
                model_registry_key="m2f",
            )
        ],
    )
    with pytest.raises(HTTPException) as exc_info:
        configuration.upsert_routing_policy(db=db, dataset_id=ds_id, username="curator", body=body)

    assert exc_info.value.status_code == 404
    assert f"Label {orphan.id} is not part of dataset {ds_id}'s hierarchy." in exc_info.value.detail


def test_execute_suggest_step_unsupported_task_raises_400(monkeypatch, ctx):
    """execute_suggest_step rejects tasks not currently supported by interactive suggestion."""
    _stub_multi_task_catalog(monkeypatch)
    db = ctx["db"]
    ds_id = ctx["dataset"].id
    cell_id = ctx["cell"].id

    image = Images(
        dataset_id=ds_id,
        file_name="test_img.png",
        file_path="/tmp/test_img.png",
        thumbnail_file_path="/tmp/thumb.png",
        width=100,
        height=100,
    )
    db.add(image)
    db.flush()
    mask = Masks(image_id=image.id, file_path="/tmp/mask.png")
    db.add(mask)
    db.commit()

    with pytest.raises(HTTPException) as exc_info:
        configuration.execute_suggest_step(
            db=db,
            dataset_id=ds_id,
            image_id=image.id,
            label_id=cell_id,
            username="curator",
            task=ModelRoutingTask.PROMPTED_SEGMENTATION,
            mask_id=mask.id,
        )

    assert exc_info.value.status_code == 400
    assert "Interactive suggestion is not supported for task 'prompted-segmentation'" in exc_info.value.detail


def test_execute_suggest_step_orphaned_label_raises_404(monkeypatch, ctx):
    """execute_suggest_step rejects orphaned labels."""
    _stub_multi_task_catalog(monkeypatch)
    db = ctx["db"]
    ds_id = ctx["dataset"].id

    other_ds = Datasets(name="other_ds2", description="", dataset_type="image", folder_path="/tmp", created_by="curator")
    db.add(other_ds)
    db.flush()
    foreign_label = Labels(dataset_id=other_ds.id, name="foreign2", value=1)
    db.add(foreign_label)
    db.flush()

    orphan = Labels(dataset_id=ds_id, name="orphan_label", value=99, parent_id=foreign_label.id)
    db.add(orphan)
    db.flush()

    image = Images(
        dataset_id=ds_id,
        file_name="test_img.png",
        file_path="/tmp/test_img.png",
        thumbnail_file_path="/tmp/thumb.png",
        width=100,
        height=100,
    )
    db.add(image)
    db.flush()
    mask = Masks(image_id=image.id, file_path="/tmp/mask.png")
    db.add(mask)
    db.commit()

    # Manually store routing row with orphan label to test execution resolution check
    row = DatasetModelRoutingConfigs(
        dataset_id=ds_id,
        bindings=[{
            "task": "cross-image-suggestion",
            "label_id": orphan.id,
            "model_registry_key": "sam3-cross",
            "inputs": None,
        }],
        updated_by="curator",
    )
    db.add(row)
    db.commit()

    with pytest.raises(HTTPException) as exc_info:
        configuration.execute_suggest_step(
            db=db,
            dataset_id=ds_id,
            image_id=image.id,
            label_id=orphan.id,
            username="curator",
            task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
            mask_id=mask.id,
        )

    assert exc_info.value.status_code == 404
    assert f"Label {orphan.id} is not part of dataset {ds_id}'s hierarchy." in exc_info.value.detail


def test_get_routing_policy_returns_stored_policy_with_unavailable_model_for_repair_display(monkeypatch, ctx):
    """get_routing_policy returns saved bindings as-is even if a model is unavailable, allowing UI repair display."""
    db = ctx["db"]
    ds_id = ctx["dataset"].id
    cell_id = ctx["cell"].id

    row = DatasetModelRoutingConfigs(
        dataset_id=ds_id,
        bindings=[
            {
                "task": "instance-segmentation",
                "label_id": cell_id,
                "model_registry_key": "deleted-or-unregistered-model",
                "inputs": None,
            }
        ],
        updated_by="curator",
    )
    db.add(row)
    db.commit()

    policy = configuration.get_routing_policy(db, ds_id)
    assert policy is not None
    assert policy.dataset_id == ds_id
    assert len(policy.bindings) == 1
    assert policy.bindings[0].model_registry_key == "deleted-or-unregistered-model"
    assert policy.bindings[0].task == "instance-segmentation"
    assert policy.bindings[0].label_id == cell_id

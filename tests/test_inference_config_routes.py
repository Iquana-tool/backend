"""Integration and route tests for dataset model routing policy endpoints (`/inference/config`)."""
import pytest
from fastapi import HTTPException, Response
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.contours  # noqa: F401
import app.database.dataset_members  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.dataset_model_routing_configs  # noqa: F401
import app.database.images  # noqa: F401
import app.database.inference_jobs  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.users  # noqa: F401
from app.database.contours import Contours
from app.database.dataset_members import DatasetMembers
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.inference_jobs import InferenceJobs
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.routes.services import inference_router
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.inference import (
    DatasetModelRoutingRead,
    DatasetModelRoutingWrite,
    ModelCatalog,
    ModelOption,
    ModelRoutingBinding,
    ModelRoutingSuggestRequest,
    ModelRoutingSuggestResult,
    ModelRoutingTask,
)
from app.schemas.permissions import DatasetRole
from app.services.inference import execution, planning
from app.services.inference.contract_resolver import LEGACY_TASK_DEFAULTS
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract


@pytest.fixture
def ctx(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'config_routes.db'}")

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    database.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()

    curator = Users(username="curator", hashed_password="x")
    annotator = Users(username="annotator", hashed_password="x")
    viewer = Users(username="viewer", hashed_password="x")
    ai_only = Users(username="ai_only", hashed_password="x")
    db.add_all([curator, annotator, viewer, ai_only])

    ds = Datasets(
        name="ds",
        description="Dataset for testing routes",
        dataset_type="image",
        folder_path=str(tmp_path),
        created_by="curator",
    )
    other_ds = Datasets(
        name="other_ds",
        description="Other dataset",
        dataset_type="image",
        folder_path=str(tmp_path),
        created_by="curator",
    )
    db.add_all([ds, other_ds])
    db.flush()

    db.add(DatasetMembers(dataset_id=ds.id, username="curator", role=DatasetRole.CURATOR.value))
    db.add(DatasetMembers(dataset_id=ds.id, username="annotator", role=DatasetRole.ANNOTATOR.value))
    db.add(DatasetMembers(dataset_id=ds.id, username="viewer", role=DatasetRole.VIEWER.value))
    db.add(
        DatasetMembers(
            dataset_id=ds.id,
            username="ai_only",
            role=DatasetRole.ANNOTATOR.value,
            denied_permissions=["annotation.create"],
        )
    )

    cell = Labels(dataset_id=ds.id, name="cell", value=1)
    db.add(cell)
    db.flush()
    nucleus = Labels(dataset_id=ds.id, name="nucleus", value=2, parent_id=cell.id)
    db.add(nucleus)
    db.flush()

    img = Images(
        dataset_id=ds.id,
        file_name="img1.png",
        file_path=str(tmp_path / "img1.png"),
        thumbnail_file_path=str(tmp_path / "t1.png"),
        width=100,
        height=100,
        color_mode="RGB",
    )
    other_img = Images(
        dataset_id=other_ds.id,
        file_name="other.png",
        file_path=str(tmp_path / "other.png"),
        thumbnail_file_path=str(tmp_path / "t_other.png"),
        width=100,
        height=100,
        color_mode="RGB",
    )
    db.add_all([img, other_img])
    db.flush()

    mask = Masks(image_id=img.id, file_path=str(tmp_path / "m1.png"))
    other_mask = Masks(image_id=other_img.id, file_path=str(tmp_path / "m_other.png"))
    db.add_all([mask, other_mask])
    db.commit()

    def get_auth_user(username: str) -> AuthenticatedUser:
        user_row = db.query(Users).filter(Users.username == username).one()
        return AuthenticatedUser.from_query(user_row)

    return dict(
        db=db,
        ds=ds,
        other_ds=other_ds,
        cell=cell,
        nucleus=nucleus,
        img=img,
        other_img=other_img,
        mask=mask,
        curator=curator,
        annotator=annotator,
        viewer=viewer,
        ai_only=ai_only,
        get_auth_user=get_auth_user,
    )


def _stub_catalog(monkeypatch):
    contract_inst = LEGACY_TASK_DEFAULTS["instance-segmentation"]
    contract_cross = LEGACY_TASK_DEFAULTS["cross-image-suggestion"]
    contract_prompted = LEGACY_TASK_DEFAULTS["prompted-segmentation"]
    contract_intra = LEGACY_TASK_DEFAULTS["instance-suggestion"]

    stub_catalog_fn = lambda db, dataset_id, tasks=None: ModelCatalog(
        models=[
            ModelOption(
                registry_key="m2f",
                name="Mask2Former",
                task="instance-segmentation",
                label_ids=[],
                input_contract=contract_inst,
                provenance="legacy_default",
            ),
            ModelOption(
                registry_key="sam3-cross",
                name="SAM3 Cross",
                task="cross-image-suggestion",
                label_ids=[],
                input_contract=contract_cross,
                provenance="legacy_default",
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
    )
    monkeypatch.setattr(planning, "model_catalog", stub_catalog_fn)
    monkeypatch.setattr(inference_router.planning, "model_catalog", stub_catalog_fn)


def _stub_predictions(monkeypatch, contours):
    monkeypatch.setattr(
        execution,
        "predict",
        lambda db, step, image, username: [c.model_copy(deep=True) for c in contours],
    )


def test_get_config_returns_204_when_absent(ctx, monkeypatch):
    _stub_catalog(monkeypatch)
    ds_id = ctx["ds"].id
    db = ctx["db"]
    user = ctx["get_auth_user"]("curator")
    response = Response()

    res = inference_router.get_dataset_model_routing(
        dataset_id=ds_id,
        response=response,
        db=db,
        user=user,
    )
    assert res is None
    assert response.status_code == 204


def test_crud_lifecycle_and_permissions(ctx, monkeypatch):
    _stub_catalog(monkeypatch)
    ds_id = ctx["ds"].id
    cell_id = ctx["cell"].id
    db = ctx["db"]

    put_body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.INSTANCE_SEGMENTATION,
                label_id=cell_id,
                model_registry_key="m2f",
            ),
            ModelRoutingBinding(
                task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
                label_id=None,
                model_registry_key="sam3-cross",
            ),
        ],
    )

    # 1. Annotator cannot PUT (403)
    annotator_user = ctx["get_auth_user"]("annotator")
    with pytest.raises(HTTPException) as exc_info:
        inference_router.update_dataset_model_routing(
            body=put_body,
            db=db,
            user=annotator_user,
        )
    assert exc_info.value.status_code == 403

    # 2. Curator can PUT (200)
    curator_user = ctx["get_auth_user"]("curator")
    saved = inference_router.update_dataset_model_routing(
        body=put_body,
        db=db,
        user=curator_user,
    )
    assert saved.dataset_id == ds_id
    assert len(saved.bindings) == 2
    assert saved.updated_by == "curator"
    assert saved.created_at is not None
    assert saved.updated_at is not None

    # 3. Annotator CAN GET saved policy (200)
    response = Response()
    got = inference_router.get_dataset_model_routing(
        dataset_id=ds_id,
        response=response,
        db=db,
        user=annotator_user,
    )
    assert got is not None
    assert got.dataset_id == ds_id
    assert len(got.bindings) == 2

    # 4. Viewer CANNOT GET saved policy (403)
    viewer_user = ctx["get_auth_user"]("viewer")
    with pytest.raises(HTTPException) as exc_info:
        inference_router.get_dataset_model_routing(
            dataset_id=ds_id,
            response=response,
            db=db,
            user=viewer_user,
        )
    assert exc_info.value.status_code == 403

    # 5. Annotator cannot DELETE (403)
    with pytest.raises(HTTPException) as exc_info:
        inference_router.delete_dataset_model_routing(
            dataset_id=ds_id,
            db=db,
            user=annotator_user,
        )
    assert exc_info.value.status_code == 403

    # 6. Curator can DELETE (200)
    deleted = inference_router.delete_dataset_model_routing(
        dataset_id=ds_id,
        db=db,
        user=curator_user,
    )
    assert deleted["success"] is True

    # 7. GET after delete returns 204
    response = Response()
    res = inference_router.get_dataset_model_routing(
        dataset_id=ds_id,
        response=response,
        db=db,
        user=curator_user,
    )
    assert res is None
    assert response.status_code == 204

    # 8. Second DELETE returns 404
    with pytest.raises(HTTPException) as exc_info:
        inference_router.delete_dataset_model_routing(
            dataset_id=ds_id,
            db=db,
            user=curator_user,
        )
    assert exc_info.value.status_code == 404


def test_suggest_step_execution_and_patch_semantics(ctx, monkeypatch):
    _stub_catalog(monkeypatch)
    ds_id = ctx["ds"].id
    cell_id = ctx["cell"].id
    img_id = ctx["img"].id
    db = ctx["db"]

    # Save a routing policy for cross-image-suggestion
    curator_user = ctx["get_auth_user"]("curator")
    put_body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
                label_id=cell_id,
                model_registry_key="sam3-cross",
            )
        ],
    )
    inference_router.update_dataset_model_routing(body=put_body, db=db, user=curator_user)

    _stub_predictions(
        monkeypatch,
        [
            Contour(
                x=[0.1, 0.2, 0.2, 0.1],
                y=[0.1, 0.1, 0.2, 0.2],
                confidence=0.9,
                label_id=cell_id,
                added_by="model",
            )
        ],
    )

    # Annotator executes suggest
    annotator_user = ctx["get_auth_user"]("annotator")
    suggest_req = ModelRoutingSuggestRequest(
        dataset_id=ds_id,
        image_id=img_id,
        label_id=cell_id,
        task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
    )
    result = inference_router.suggest_model_routing_step(
        body=suggest_req,
        db=db,
        user=annotator_user,
    )
    assert result.dataset_id == ds_id
    assert result.image_id == img_id
    assert result.label_id == cell_id
    assert result.task == "cross-image-suggestion"
    assert result.contours_created == 1
    assert len(result.contour_ids) == 1

    # Verify contour was committed in DB
    assert db.query(Contours).filter(Contours.id == result.contour_ids[0]).count() == 1

    # Unsupported suggestion task is rejected (400)
    with pytest.raises(HTTPException) as exc_info:
        inference_router.suggest_model_routing_step(
            body=ModelRoutingSuggestRequest(
                dataset_id=ds_id,
                image_id=img_id,
                label_id=cell_id,
                task=ModelRoutingTask.PROMPTED_SEGMENTATION,
            ),
            db=db,
            user=annotator_user,
        )
    assert exc_info.value.status_code == 400

    # Cross-dataset image is rejected (404)
    with pytest.raises(HTTPException) as exc_info:
        inference_router.suggest_model_routing_step(
            body=ModelRoutingSuggestRequest(
                dataset_id=ds_id,
                image_id=ctx["other_img"].id,
                label_id=cell_id,
                task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
            ),
            db=db,
            user=annotator_user,
        )
    assert exc_info.value.status_code == 404

    # Unconfigured label is rejected (404)
    with pytest.raises(HTTPException) as exc_info:
        inference_router.suggest_model_routing_step(
            body=ModelRoutingSuggestRequest(
                dataset_id=ds_id,
                image_id=img_id,
                label_id=ctx["nucleus"].id,
                task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
            ),
            db=db,
            user=annotator_user,
        )
    assert exc_info.value.status_code == 404

    # User with interactive AI but missing annotation.create is rejected (403)
    ai_only_user = ctx["get_auth_user"]("ai_only")
    with pytest.raises(HTTPException) as exc_info:
        inference_router.suggest_model_routing_step(
            body=suggest_req,
            db=db,
            user=ai_only_user,
        )
    assert exc_info.value.status_code == 403

    # Viewer is rejected (403)
    viewer_user = ctx["get_auth_user"]("viewer")
    with pytest.raises(HTTPException) as exc_info:
        inference_router.suggest_model_routing_step(
            body=suggest_req,
            db=db,
            user=viewer_user,
        )
    assert exc_info.value.status_code == 403


def test_suggest_rejected_when_batch_job_active(ctx, monkeypatch):
    """Ensure suggestions return 409 Conflict if a batch inference job is currently active."""
    _stub_catalog(monkeypatch)
    ds_id = ctx["ds"].id
    cell_id = ctx["cell"].id
    img_id = ctx["img"].id
    db = ctx["db"]
    curator_user = ctx["get_auth_user"]("curator")
    annotator_user = ctx["get_auth_user"]("annotator")

    put_body = DatasetModelRoutingWrite(
        dataset_id=ds_id,
        bindings=[
            ModelRoutingBinding(
                task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
                label_id=cell_id,
                model_registry_key="sam3-cross",
            )
        ],
    )
    inference_router.update_dataset_model_routing(body=put_body, db=db, user=curator_user)

    # Insert an active batch inference job
    job = InferenceJobs(
        dataset_id=ds_id,
        name="Active Batch Run",
        status="running",
        total_units=5,
        created_by="curator",
    )
    db.add(job)
    db.commit()

    model_calls = []

    def _record_model_call(*args, **kwargs):
        model_calls.append(True)
        raise AssertionError("the model must not be called while a batch is active")

    monkeypatch.setattr(execution, "run_unit", _record_model_call)

    suggest_req = ModelRoutingSuggestRequest(
        dataset_id=ds_id,
        image_id=img_id,
        label_id=cell_id,
        task=ModelRoutingTask.CROSS_IMAGE_SUGGESTION,
    )

    with pytest.raises(HTTPException) as exc_info:
        inference_router.suggest_model_routing_step(
            body=suggest_req,
            db=db,
            user=annotator_user,
        )
    assert exc_info.value.status_code == 409
    assert "currently running on this dataset" in exc_info.value.detail or "is currently running" in exc_info.value.detail
    assert model_calls == []


@pytest.mark.anyio
async def test_route_cancel_job_lifecycle(ctx):
    """Test route-level cancel_job handler for pending, running, cancelling, and terminal jobs."""
    from datetime import datetime, timezone
    from app.database.inference_jobs import InferenceJobItems

    db = ctx["db"]
    ds_id = ctx["ds"].id
    curator_user = ctx["get_auth_user"]("curator")
    viewer_user = ctx["get_auth_user"]("viewer")

    # 1. Cancel an unstarted/pending job (started_at is None) -> immediately finalized as cancelled
    pending_job = InferenceJobs(
        dataset_id=ds_id,
        name="Unstarted Run",
        status="pending",
        started_at=None,
        total_units=2,
        created_by="curator",
    )
    db.add(pending_job)
    db.flush()
    item1 = InferenceJobItems(job_id=pending_job.id, image_id=ctx["img"].id, step_index=0, level=0, status="pending")
    item2 = InferenceJobItems(job_id=pending_job.id, image_id=ctx["img"].id, step_index=1, level=1, status="pending")
    db.add_all([item1, item2])
    db.commit()

    snap_pending = await inference_router.cancel_job(pending_job.id, db=db, user=curator_user)
    assert snap_pending.status == "cancelled"
    db.expire_all()
    assert db.get(InferenceJobs, pending_job.id).status == "cancelled"
    assert db.get(InferenceJobItems, item1.id).status == "skipped"

    # 2. Cancel a running job that has already claimed an item -> transitions to "cancelling"
    running_job = InferenceJobs(
        dataset_id=ds_id,
        name="Running Run",
        status="running",
        started_at=datetime.now(timezone.utc),
        total_units=2,
        created_by="curator",
    )
    db.add(running_job)
    db.flush()
    running_item = InferenceJobItems(
        job_id=running_job.id,
        image_id=ctx["img"].id,
        step_index=0,
        level=0,
        status="running",
    )
    db.add(running_item)
    db.commit()

    snap_running = await inference_router.cancel_job(running_job.id, db=db, user=curator_user)
    assert snap_running.status == "cancelling"
    db.expire_all()
    assert db.get(InferenceJobs, running_job.id).status == "cancelling"

    # 3. Second cancel on a job already in "cancelling" -> finalized immediately as "cancelled"
    snap_second_cancel = await inference_router.cancel_job(running_job.id, db=db, user=curator_user)
    assert snap_second_cancel.status == "cancelled"
    db.expire_all()
    assert db.get(InferenceJobs, running_job.id).status == "cancelled"

    # 4. Cancel on an already terminal job returns current snapshot directly
    snap_terminal = await inference_router.cancel_job(running_job.id, db=db, user=curator_user)
    assert snap_terminal.status == "cancelled"

    # 5. Cancel on non-existent job raises 404
    with pytest.raises(HTTPException) as exc_404:
        await inference_router.cancel_job(99999, db=db, user=curator_user)
    assert exc_404.value.status_code == 404

    # 6. Cancel without batch permission (viewer) raises 403
    with pytest.raises(HTTPException) as exc_403:
        await inference_router.cancel_job(running_job.id, db=db, user=viewer_user)
    assert exc_403.value.status_code == 403


@pytest.mark.anyio
async def test_route_cancel_claimed_job_with_no_started_item_finishes_immediately(ctx):
    """A lost run_next task must not require a second cancel to release the dataset."""
    from datetime import datetime, timezone
    from app.database.inference_jobs import InferenceJobItems

    db = ctx["db"]
    ds_id = ctx["ds"].id
    curator_user = ctx["get_auth_user"]("curator")

    job = InferenceJobs(
        dataset_id=ds_id,
        name="Lost follow-up task",
        status="running",
        started_at=datetime.now(timezone.utc),
        total_units=2,
        created_by="curator",
    )
    db.add(job)
    db.flush()
    items = [
        InferenceJobItems(
            job_id=job.id,
            image_id=ctx["img"].id,
            step_index=index,
            level=index,
            status="pending",
        )
        for index in range(2)
    ]
    db.add_all(items)
    db.commit()

    snapshot = await inference_router.cancel_job(job.id, db=db, user=curator_user)

    assert snapshot.status == "cancelled"
    db.expire_all()
    assert db.get(InferenceJobs, job.id).status == "cancelled"
    assert all(item.status == "skipped" for item in db.query(InferenceJobItems).filter_by(job_id=job.id))


@pytest.mark.anyio
async def test_route_delete_job_lifecycle(ctx):
    """Test route-level delete_job handler enforcing baseline rules (only actively running is blocked)."""
    db = ctx["db"]
    ds_id = ctx["ds"].id
    curator_user = ctx["get_auth_user"]("curator")
    viewer_user = ctx["get_auth_user"]("viewer")

    # 1. Delete a running job -> blocked with 409
    running_job = InferenceJobs(
        dataset_id=ds_id,
        name="Running Run",
        status="running",
        total_units=1,
        created_by="curator",
    )
    db.add(running_job)
    db.commit()

    with pytest.raises(HTTPException) as exc_running:
        await inference_router.delete_job(running_job.id, db=db, user=curator_user)
    assert exc_running.value.status_code == 409
    assert "still running" in exc_running.value.detail

    # 2. Delete a pending job -> allowed in baseline
    pending_job = InferenceJobs(
        dataset_id=ds_id,
        name="Pending Run",
        status="pending",
        total_units=1,
        created_by="curator",
    )
    db.add(pending_job)
    db.commit()
    pending_id = pending_job.id

    res_pending = await inference_router.delete_job(pending_id, db=db, user=curator_user)
    assert res_pending["success"] is True
    assert db.get(InferenceJobs, pending_id) is None

    # 3. Delete a cancelling job -> allowed in baseline
    cancelling_job = InferenceJobs(
        dataset_id=ds_id,
        name="Cancelling Run",
        status="cancelling",
        total_units=1,
        created_by="curator",
    )
    db.add(cancelling_job)
    db.commit()
    cancelling_id = cancelling_job.id

    res_cancelling = await inference_router.delete_job(cancelling_id, db=db, user=curator_user)
    assert res_cancelling["success"] is True
    assert db.get(InferenceJobs, cancelling_id) is None

    # 4. Delete a completed job -> allowed
    completed_job = InferenceJobs(
        dataset_id=ds_id,
        name="Completed Run",
        status="succeeded",
        total_units=1,
        created_by="curator",
    )
    db.add(completed_job)
    db.commit()
    completed_id = completed_job.id

    res_completed = await inference_router.delete_job(completed_id, db=db, user=curator_user)
    assert res_completed["success"] is True
    assert db.get(InferenceJobs, completed_id) is None

    # 5. Delete non-existent job -> 404
    with pytest.raises(HTTPException) as exc_404:
        await inference_router.delete_job(99999, db=db, user=curator_user)
    assert exc_404.value.status_code == 404

    # 6. Delete without permission -> 403
    another_job = InferenceJobs(
        dataset_id=ds_id,
        name="Another Run",
        status="succeeded",
        total_units=1,
        created_by="curator",
    )
    db.add(another_job)
    db.commit()

    with pytest.raises(HTTPException) as exc_403:
        await inference_router.delete_job(another_job.id, db=db, user=viewer_user)
    assert exc_403.value.status_code == 403


@pytest.mark.anyio
async def test_model_catalog_endpoint_permissions(ctx, monkeypatch):
    """Model catalog allows users with AI_BATCH_INFER or AI_INTERACTIVE, rejecting users with neither."""
    async def _inline_to_thread(func, *args, **kwargs):
        """Keep this route test on the fixture's thread-bound SQLite session."""
        return func(*args, **kwargs)

    monkeypatch.setattr(inference_router.asyncio, "to_thread", _inline_to_thread)
    _stub_catalog(monkeypatch)
    ds_id = ctx["ds"].id
    db = ctx["db"]

    # 1. Curator (AI_BATCH_INFER) can view model catalog (200)
    curator_user = ctx["get_auth_user"]("curator")
    catalog_curator = await inference_router.get_model_catalog(dataset_id=ds_id, db=db, user=curator_user)
    assert catalog_curator is not None
    assert len(catalog_curator.models) > 0

    # 2. Annotator (AI_INTERACTIVE) can also view model catalog (200)
    annotator_user = ctx["get_auth_user"]("annotator")
    catalog_annotator = await inference_router.get_model_catalog(dataset_id=ds_id, db=db, user=annotator_user)
    assert catalog_annotator is not None
    assert len(catalog_annotator.models) > 0

    # 3. Viewer (no AI permissions) gets 403 Forbidden
    viewer_user = ctx["get_auth_user"]("viewer")
    with pytest.raises(HTTPException) as exc_info:
        await inference_router.get_model_catalog(dataset_id=ds_id, db=db, user=viewer_user)
    assert exc_info.value.status_code == 403

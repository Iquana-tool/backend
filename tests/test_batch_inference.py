"""Unit tests for batch inference (`app.services.inference`).

Covers the decisions that make an orchestration behave predictably: hierarchy ordering of
the work list, filtering a multiclass model's output down to a step's label, nesting
child-level predictions under the right parent, patch-vs-replace semantics, and the counts
the replace warning is built from. Driven straight against the services with a temp SQLite
database; the AI service is stubbed, since none of this is about HTTP.
"""
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.contours  # noqa: F401
import app.database.dataset_members  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.inference_jobs  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.rejections  # noqa: F401
import app.database.users  # noqa: F401
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.inference_jobs import InferenceJobItems
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.schemas.inference import (
    ImageSelection,
    InferenceJobCreate,
    InferenceOptions,
    InferenceStepRequest,
    ModelCatalog,
    ModelOption,
    ResolvedStep,
    UnparentedPolicy,
    WriteMode,
)
from app.services.inference import execution, planning
from iquana_toolbox.schemas.database.contours import Contour


def box(x0, y0, x1, y1, *, label_id=None, confidence=1.0):
    return Contour(x=[x0, x1, x1, x0], y=[y0, y0, y1, y1],
                   label_id=label_id, confidence=confidence, added_by="model")


def db_contour(mask_id, contour: Contour, *, label_id=None, parent_id=None, author="cur"):
    return Contours(mask_id=mask_id, parent_id=parent_id, label_id=label_id,
                    added_by="manual", author_username=author, confidence_score=1.0,
                    area=1.0, perimeter=1.0, circularity=1.0, diameter=1.0,
                    x=contour.x, y=contour.y)


@pytest.fixture
def ctx(tmp_path):
    """A two-level dataset (cell > nucleus) with three images, one already annotated."""
    engine = create_engine(f"sqlite:///{tmp_path / 'inference.db'}")
    database.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()

    db.add(Users(username="cur", hashed_password="x"))
    dataset = Datasets(name="ds", description="", dataset_type="image",
                       folder_path=str(tmp_path), created_by="cur")
    db.add(dataset)
    db.flush()

    cell = Labels(dataset_id=dataset.id, name="cell", value=1)
    db.add(cell)
    db.flush()
    nucleus = Labels(dataset_id=dataset.id, name="nucleus", value=2, parent_id=cell.id)
    debris = Labels(dataset_id=dataset.id, name="debris", value=3)
    db.add_all([nucleus, debris])
    db.flush()

    images, masks = [], []
    for index in range(3):
        image = Images(dataset_id=dataset.id, file_name=f"i{index}.png",
                       file_path=str(tmp_path / f"i{index}.png"),
                       thumbnail_file_path=str(tmp_path / f"t{index}.png"),
                       width=100, height=100, color_mode="RGB")
        db.add(image)
        db.flush()
        mask = Masks(image_id=image.id, file_path=str(tmp_path / f"m{index}.png"))
        db.add(mask)
        db.flush()
        images.append(image)
        masks.append(mask)

    # Image 0 already carries one hand-drawn cell.
    existing = db_contour(masks[0].id, box(0.1, 0.1, 0.4, 0.4), label_id=cell.id)
    db.add(existing)
    db.commit()

    return dict(db=db, dataset=dataset, cell=cell, nucleus=nucleus, debris=debris,
                images=images, masks=masks, existing=existing)


def step(label_id, *, level=0, parent_label_id=None, model_label_ids=(), key="m2f"):
    return ResolvedStep(
        label_id=label_id, model_registry_key=key, task="instance-segmentation",
        level=level, parent_label_id=parent_label_id, label_name="l", model_name="M",
        model_label_ids=list(model_label_ids),
    )


# --------------------------------------------------------------------------- #
# Broker wiring
# --------------------------------------------------------------------------- #
def test_backend_tasks_never_publish_to_the_shared_default_queue():
    """The ai-service worker also consumes "celery"; a collision there is silent data loss.

    It has no `inference.*` task registered, so a message that lands on the shared default
    queue is discarded with "Received unregistered task of type ..." and the run stops dead
    with nothing to retry it.
    """
    from app.services.celery_app import BACKEND_QUEUE, celery_app

    assert BACKEND_QUEUE != "celery"
    assert celery_app.conf.task_default_queue == BACKEND_QUEUE

    from app.services.inference.tasks import run_job, run_next

    for task in (run_job, run_next):
        # The router resolves to a kombu Queue, whose name is what the broker actually sees.
        queue = celery_app.amqp.router.route({}, task.name, (1,), {})["queue"]
        assert getattr(queue, "name", queue) == BACKEND_QUEUE


# --------------------------------------------------------------------------- #
# Hierarchy
# --------------------------------------------------------------------------- #
def test_label_levels_are_depths_from_the_roots(ctx):
    levels = planning.label_levels(ctx["db"], ctx["dataset"].id)
    assert levels[ctx["cell"].id] == 0
    assert levels[ctx["debris"].id] == 0
    assert levels[ctx["nucleus"].id] == 1


def test_steps_are_ordered_root_level_first(ctx, monkeypatch):
    _stub_catalog(monkeypatch, ctx)
    resolved = planning.resolve_steps(ctx["db"], ctx["dataset"].id, [
        InferenceStepRequest(label_id=ctx["nucleus"].id, model_registry_key="m2f"),
        InferenceStepRequest(label_id=ctx["cell"].id, model_registry_key="m2f"),
    ])
    assert [s.label_name for s in resolved] == ["cell", "nucleus"]
    assert [s.level for s in resolved] == [0, 1]
    assert resolved[1].parent_label_id == ctx["cell"].id


def test_work_list_is_ordered_by_level_across_the_whole_dataset(ctx, monkeypatch):
    """Every root unit must precede every child unit -- not just per image."""
    _stub_catalog(monkeypatch, ctx)
    job = planning.create_job(ctx["db"], ctx["dataset"].id, "cur", _job_body(ctx))

    items = (
        ctx["db"].query(InferenceJobItems)
        .filter(InferenceJobItems.job_id == job.id)
        .order_by(InferenceJobItems.level, InferenceJobItems.step_index,
                  InferenceJobItems.image_id)
        .all()
    )
    assert job.total_units == 2 * 3
    levels = [item.level for item in items]
    assert levels == sorted(levels)
    assert levels == [0, 0, 0, 1, 1, 1]


def test_the_worker_never_reaches_a_child_unit_before_the_root_level_is_done(ctx, monkeypatch):
    """The guarantee, from the worker's side: drain the queue and watch the levels."""
    from app.services.inference.tasks import _next_item

    _stub_catalog(monkeypatch, ctx)
    job = planning.create_job(ctx["db"], ctx["dataset"].id, "cur", _job_body(ctx))

    seen = []
    while (item := _next_item(ctx["db"], job.id)) is not None:
        seen.append((item.level, item.image_id))
        item.status = "done"
        ctx["db"].commit()

    assert len(seen) == 6
    root_positions = [i for i, (level, _) in enumerate(seen) if level == 0]
    child_positions = [i for i, (level, _) in enumerate(seen) if level == 1]
    assert max(root_positions) < min(child_positions)


def test_cancelling_marks_the_untouched_units_skipped(ctx, monkeypatch):
    from app.services.inference.tasks import abandon_pending

    _stub_catalog(monkeypatch, ctx)
    job = planning.create_job(ctx["db"], ctx["dataset"].id, "cur", _job_body(ctx))
    first = ctx["db"].query(InferenceJobItems).filter_by(job_id=job.id).first()
    first.status = "done"
    ctx["db"].commit()

    abandon_pending(ctx["db"], job.id)
    ctx["db"].commit()

    statuses = [item.status for item in
                ctx["db"].query(InferenceJobItems).filter_by(job_id=job.id).all()]
    assert statuses.count("done") == 1
    assert statuses.count("skipped") == 5


def test_cancelling_a_run_no_worker_ever_claimed_frees_the_dataset(ctx, monkeypatch):
    """A job the broker never delivered must not block the dataset forever."""
    from app.services.inference.tasks import abandon_pending, finish

    _stub_catalog(monkeypatch, ctx)
    job = planning.create_job(ctx["db"], ctx["dataset"].id, "cur", _job_body(ctx))
    assert job.started_at is None  # no worker picked it up

    # What the cancel route does for an unreachable run: finalise it rather than asking a
    # worker that is not there to do it.
    abandon_pending(ctx["db"], job.id)
    finish(ctx["db"], job, "cancelled")

    assert planning.active_job(ctx["db"], ctx["dataset"].id) is None
    planning.create_job(ctx["db"], ctx["dataset"].id, "cur", _job_body(ctx))


def test_a_run_stuck_in_cancelling_is_finalised_on_the_second_ask(ctx, monkeypatch):
    from app.services.inference.tasks import abandon_pending, finish

    _stub_catalog(monkeypatch, ctx)
    job = planning.create_job(ctx["db"], ctx["dataset"].id, "cur", _job_body(ctx))
    job.status = "cancelling"
    ctx["db"].commit()
    # Still blocking: `cancelling` is a request, not a terminal state.
    assert planning.active_job(ctx["db"], ctx["dataset"].id) is not None

    abandon_pending(ctx["db"], job.id)
    finish(ctx["db"], job, "cancelled")
    assert planning.active_job(ctx["db"], ctx["dataset"].id) is None


def test_a_label_may_only_appear_once_in_a_plan(ctx):
    with pytest.raises(ValueError, match="at most once"):
        InferenceJobCreate(
            dataset_id=ctx["dataset"].id,
            steps=[InferenceStepRequest(label_id=1, model_registry_key="a"),
                   InferenceStepRequest(label_id=1, model_registry_key="b")],
        )


def test_a_second_job_on_one_dataset_is_refused(ctx, monkeypatch):
    _stub_catalog(monkeypatch, ctx)
    planning.create_job(ctx["db"], ctx["dataset"].id, "cur", _job_body(ctx))
    with pytest.raises(HTTPException) as excinfo:
        planning.create_job(ctx["db"], ctx["dataset"].id, "cur", _job_body(ctx))
    assert excinfo.value.status_code == 409


def test_replace_without_acknowledgement_is_refused(ctx, monkeypatch):
    _stub_catalog(monkeypatch, ctx)
    body = _job_body(ctx)
    body.options = InferenceOptions(write_mode=WriteMode.REPLACE)
    body.confirm_replace = False
    with pytest.raises(HTTPException) as excinfo:
        planning.create_job(ctx["db"], ctx["dataset"].id, "cur", body)
    assert excinfo.value.status_code == 400


def test_a_model_cannot_be_bound_to_a_label_it_does_not_predict(ctx, monkeypatch):
    _stub_catalog(monkeypatch, ctx, label_ids=[ctx["cell"].id])
    with pytest.raises(HTTPException) as excinfo:
        planning.resolve_steps(ctx["db"], ctx["dataset"].id, [
            InferenceStepRequest(label_id=ctx["nucleus"].id, model_registry_key="m2f"),
        ])
    assert excinfo.value.status_code == 400


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def test_multiclass_output_is_filtered_to_the_steps_label(ctx):
    """The orchestration promise: a shared model contributes only its bound class."""
    plan_step = step(ctx["cell"].id, model_label_ids=[ctx["cell"].id, ctx["debris"].id])
    kept = execution.filter_for_step([
        box(0.1, 0.1, 0.2, 0.2, label_id=ctx["cell"].id),
        box(0.5, 0.5, 0.6, 0.6, label_id=ctx["debris"].id),
    ], plan_step)
    assert len(kept) == 1
    assert kept[0].label_id == ctx["cell"].id
    assert kept[0].added_by == "m2f"


def test_class_agnostic_output_is_stamped_with_the_steps_label(ctx):
    kept = execution.filter_for_step([box(0.1, 0.1, 0.2, 0.2)], step(ctx["cell"].id))
    assert [c.label_id for c in kept] == [ctx["cell"].id]


def test_a_failed_prediction_names_the_model_and_the_services_reply(ctx, monkeypatch):
    """A bare httpx 500 names neither, which makes one broken model look like a broken page."""
    class FakeResponse:
        text = "boom"

        @staticmethod
        def json():
            return {"detail": "No module named 'models.mask2former'"}

    error = RuntimeError("Server error '500' for url '.../annotation_session/run'")
    error.response = FakeResponse()

    def explode(db, step, image, username):
        raise error

    monkeypatch.setattr(execution, "predict", explode)
    with pytest.raises(execution.InferenceUnitError) as excinfo:
        execution.run_unit(ctx["db"], step(ctx["cell"].id), ctx["images"][1],
                           InferenceOptions(), "cur")

    message = str(excinfo.value)
    assert "'m2f'" in message and "instance-segmentation" in message
    assert "No module named 'models.mask2former'" in message


def test_ids_echoed_back_by_a_model_are_discarded(ctx):
    """A model's idea of a contour/parent id must never reach the database."""
    prediction = box(0.1, 0.1, 0.2, 0.2)
    prediction.id = 999
    prediction.parent_id = 888
    kept = execution.filter_for_step([prediction], step(ctx["cell"].id))
    assert kept[0].id is None and kept[0].parent_id is None


def test_low_confidence_predictions_are_dropped(ctx):
    kept = execution.filter_for_step(
        [box(0.1, 0.1, 0.2, 0.2, confidence=0.3), box(0.5, 0.5, 0.6, 0.6, confidence=0.9)],
        step(ctx["cell"].id), min_confidence=0.5,
    )
    assert len(kept) == 1


# --------------------------------------------------------------------------- #
# Parent attachment
# --------------------------------------------------------------------------- #
def test_children_are_nested_under_the_containing_parent(ctx):
    parents = [ctx["existing"]]
    attached, dropped = execution.attach_parents(
        [box(0.15, 0.15, 0.25, 0.25)], parents, InferenceOptions()
    )
    assert dropped == 0
    assert attached[0].parent_id == ctx["existing"].id


def test_orphan_children_are_dropped_by_default(ctx):
    attached, dropped = execution.attach_parents(
        [box(0.7, 0.7, 0.8, 0.8)], [ctx["existing"]], InferenceOptions()
    )
    assert (attached, dropped) == ([], 1)


def test_orphan_children_can_be_kept_at_root(ctx):
    options = InferenceOptions(unparented=UnparentedPolicy.KEEP_AT_ROOT)
    attached, dropped = execution.attach_parents(
        [box(0.7, 0.7, 0.8, 0.8)], [ctx["existing"]], options
    )
    assert dropped == 0 and attached[0].parent_id is None


# --------------------------------------------------------------------------- #
# Merging into the dataset
# --------------------------------------------------------------------------- #
def test_patch_keeps_existing_work_and_drops_the_duplicate(ctx, monkeypatch):
    """A prediction that repeats a hand-drawn object is suppressed, not written twice."""
    duplicate = box(0.1, 0.1, 0.4, 0.4)
    fresh = box(0.6, 0.6, 0.9, 0.9)
    _stub_predictions(monkeypatch, [duplicate, fresh])

    result = execution.run_unit(
        ctx["db"], step(ctx["cell"].id), ctx["images"][0], InferenceOptions(), "cur"
    )
    ctx["db"].commit()

    assert (result.created, result.suppressed) == (1, 1)
    contours = ctx["db"].query(Contours).filter_by(mask_id=ctx["masks"][0].id).all()
    assert len(contours) == 2
    assert any(c.added_by == "manual" for c in contours)


def test_two_predictions_of_the_same_object_collapse_to_one(ctx, monkeypatch):
    _stub_predictions(monkeypatch, [
        box(0.6, 0.6, 0.9, 0.9, confidence=0.6),
        box(0.61, 0.61, 0.9, 0.9, confidence=0.95),
    ])
    result = execution.run_unit(
        ctx["db"], step(ctx["cell"].id), ctx["images"][1], InferenceOptions(), "cur"
    )
    ctx["db"].commit()

    assert (result.created, result.suppressed) == (1, 1)
    # The survivor is the confident one.
    written = ctx["db"].query(Contours).filter_by(mask_id=ctx["masks"][1].id).one()
    assert written.confidence_score == pytest.approx(0.95)


def test_predictions_under_different_parents_never_suppress_each_other(ctx, monkeypatch):
    """Two nuclei at the same coordinates in different cells are two objects."""
    other_cell = db_contour(ctx["masks"][0].id, box(0.5, 0.5, 0.9, 0.9), label_id=ctx["cell"].id)
    ctx["db"].add(other_cell)
    ctx["db"].commit()

    _stub_predictions(monkeypatch, [box(0.15, 0.15, 0.3, 0.3), box(0.6, 0.6, 0.75, 0.75)])
    child_step = step(ctx["nucleus"].id, level=1, parent_label_id=ctx["cell"].id)
    result = execution.run_unit(
        ctx["db"], child_step, ctx["images"][0], InferenceOptions(), "cur"
    )
    ctx["db"].commit()

    assert (result.created, result.suppressed) == (2, 0)
    written = ctx["db"].query(Contours).filter_by(label_id=ctx["nucleus"].id).all()
    assert {c.parent_id for c in written} == {ctx["existing"].id, other_cell.id}


def test_written_contours_are_attributed_and_unreviewed(ctx, monkeypatch):
    _stub_predictions(monkeypatch, [box(0.6, 0.6, 0.9, 0.9)])
    execution.run_unit(
        ctx["db"], step(ctx["cell"].id), ctx["images"][1], InferenceOptions(), "cur"
    )
    ctx["db"].commit()

    written = ctx["db"].query(Contours).filter_by(mask_id=ctx["masks"][1].id).one()
    assert written.added_by == "m2f"          # what produced the geometry
    assert written.author_username == "cur"   # who started the run
    assert written.reviewed_by == []          # goes to the review queue like any annotation


# --------------------------------------------------------------------------- #
# Replace
# --------------------------------------------------------------------------- #
def test_replace_preview_counts_what_would_be_destroyed(ctx):
    nucleus = db_contour(ctx["masks"][0].id, box(0.15, 0.15, 0.25, 0.25),
                         label_id=ctx["nucleus"].id, parent_id=ctx["existing"].id)
    ctx["db"].add(nucleus)
    ctx["db"].commit()

    preview = planning.replace_preview(
        ctx["db"], [image.id for image in ctx["images"]], preserve_reviewed=False
    )
    assert preview.images == 3
    assert preview.contours == 2
    assert preview.root_contours == 1
    assert preview.reviewed_contours == 0


def test_replace_deletes_contours_with_their_children(ctx):
    nucleus = db_contour(ctx["masks"][0].id, box(0.15, 0.15, 0.25, 0.25),
                         label_id=ctx["nucleus"].id, parent_id=ctx["existing"].id)
    ctx["db"].add(nucleus)
    ctx["masks"][0].fully_annotated = True
    ctx["db"].commit()

    deleted = execution.wipe_images(ctx["db"], [ctx["images"][0].id], preserve_reviewed=False)
    assert deleted == 2
    assert ctx["db"].query(Contours).count() == 0
    assert ctx["db"].get(Masks, ctx["masks"][0].id).fully_annotated is False


def test_preserve_reviewed_spares_approved_objects_and_their_subtree(ctx):
    reviewer = ctx["db"].query(Users).filter_by(username="cur").one()
    ctx["existing"].reviewed_by = [reviewer]
    nucleus = db_contour(ctx["masks"][0].id, box(0.15, 0.15, 0.25, 0.25),
                         label_id=ctx["nucleus"].id, parent_id=ctx["existing"].id)
    unreviewed = db_contour(ctx["masks"][0].id, box(0.6, 0.6, 0.9, 0.9), label_id=ctx["cell"].id)
    ctx["db"].add_all([nucleus, unreviewed])
    ctx["db"].commit()

    deleted = execution.wipe_images(ctx["db"], [ctx["images"][0].id], preserve_reviewed=True)
    assert deleted == 1
    survivors = {c.id for c in ctx["db"].query(Contours).all()}
    assert survivors == {ctx["existing"].id, nucleus.id}


# --------------------------------------------------------------------------- #
from app.services.inference.contract_resolver import LEGACY_TASK_DEFAULTS


def _stub_catalog(monkeypatch, ctx, label_ids=()):
    """Pretend one ready instance-segmentation model exists, without touching MLflow."""
    contract = LEGACY_TASK_DEFAULTS["instance-segmentation"]
    monkeypatch.setattr(planning, "model_catalog", lambda db, dataset_id: ModelCatalog(
        models=[ModelOption(registry_key="m2f", name="Mask2Former",
                            task="instance-segmentation", label_ids=list(label_ids),
                            input_contract=contract, provenance="legacy_default")],
    ))


def _stub_predictions(monkeypatch, contours):
    monkeypatch.setattr(execution, "predict", lambda db, step, image, username: [
        contour.model_copy(deep=True) for contour in contours
    ])


def _job_body(ctx):
    return InferenceJobCreate(
        dataset_id=ctx["dataset"].id,
        steps=[
            InferenceStepRequest(label_id=ctx["nucleus"].id, model_registry_key="m2f"),
            InferenceStepRequest(label_id=ctx["cell"].id, model_registry_key="m2f"),
        ],
        image_selection=ImageSelection.ALL,
    )

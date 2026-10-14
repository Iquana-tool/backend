"""Mapping a model's label space onto a dataset's labels (app/services/model_label_mapping.py)."""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.datasets  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.model_label_mappings  # noqa: F401
import app.database.users  # noqa: F401
from app.database.datasets import Datasets
from app.database.labels import Labels
from app.database.model_label_mappings import ModelLabelMappings
from app.database.users import Users
from app.schemas.inference import ResolvedStep
from app.schemas.model_label_mapping import ClassMappingWrite
from app.services import model_label_mapping as mlm
from app.services.inference import planning
from app.services.inference.execution import filter_for_step
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract

KEY = "choroidalyzer"
INFO = {
    "registry_key": KEY,
    "name": "Choroidalyzer",
    "classes": [
        {"key": "region", "name": "Choroid region"},
        {"key": "vessel", "name": "Choroidal vessels", "part_of": "region"},
        {"key": "fovea", "name": "Fovea"},
    ],
    "input_contracts": [InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[],
    ).model_dump()],
}


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'mapping.db'}")
    database.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(Users(username="tester", hashed_password="pw"))
    dataset = Datasets(name="oct", description="", dataset_type="image",
                       folder_path=str(tmp_path), created_by="tester")
    db.add(dataset)
    db.flush()
    choroid = Labels(dataset_id=dataset.id, name="Choroid", value=1)
    fovea = Labels(dataset_id=dataset.id, name="fovea", value=2)  # same name, other case
    db.add_all([choroid, fovea])
    db.commit()
    monkeypatch.setattr(mlm, "_model_info", lambda key: INFO)
    return dict(db=db, dataset_id=dataset.id, choroid=choroid, fovea=fovea)


def _by_class(mapping):
    return {entry.model_class.key: (entry.action, entry.label_id, entry.source) for entry in mapping.classes}


def _prediction():
    """A choroid with two vessels inside, and a fovea -- as Choroidalyzer returns them."""
    region = Contour(x=[0, 1, 1], y=[0, 0, 1], model_class="region")
    region.children = [Contour(x=[0, .5, .5], y=[0, 0, .5], model_class="vessel") for _ in range(2)]
    return [region, Contour(x=[0, .1, .1], y=[0, 0, .1], model_class="fovea")]


def test_unsaved_classes_fall_back_to_same_name_labels(ctx):
    mapping = mlm.get_mapping(ctx["db"], ctx["dataset_id"], KEY)
    assert _by_class(mapping) == {
        "region": ("unmapped", None, "none"),       # "Choroid" is not "Choroid region"
        "vessel": ("unmapped", None, "none"),
        "fovea": ("map", ctx["fovea"].id, "suggested"),
    }


def test_saved_choices_win_and_can_be_cleared(ctx):
    db, ds = ctx["db"], ctx["dataset_id"]
    mlm.save_mapping(db, ds, KEY, [
        ClassMappingWrite(model_class="region", action="map", label_id=ctx["choroid"].id),
        ClassMappingWrite(model_class="fovea", action="discard"),
    ], "tester")
    assert _by_class(mlm.get_mapping(db, ds, KEY))["fovea"] == ("discard", None, "saved")

    mlm.save_mapping(db, ds, KEY, [ClassMappingWrite(model_class="fovea", action="unmapped")], "tester")
    by_class = _by_class(mlm.get_mapping(db, ds, KEY))
    assert by_class["fovea"] == ("map", ctx["fovea"].id, "suggested")
    assert by_class["region"] == ("map", ctx["choroid"].id, "saved")


def test_created_labels_follow_the_model_hierarchy(ctx):
    db, ds = ctx["db"], ctx["dataset_id"]
    # Child listed first on purpose: parents are processed first anyway.
    mlm.save_mapping(db, ds, KEY, [
        ClassMappingWrite(model_class="vessel", action="create"),
        ClassMappingWrite(model_class="region", action="map", label_id=ctx["choroid"].id),
    ], "tester")
    vessels = db.query(Labels).filter_by(dataset_id=ds, name="Choroidal vessels").one()
    assert vessels.parent_id == ctx["choroid"].id
    assert vessels.value == 3
    assert _by_class(mlm.get_mapping(db, ds, KEY))["vessel"] == ("map", vessels.id, "saved")


def test_creating_a_label_that_exists_points_at_mapping_instead(ctx):
    with pytest.raises(HTTPException) as raised:
        mlm.save_mapping(ctx["db"], ctx["dataset_id"], KEY, [
            ClassMappingWrite(model_class="region", action="create", label_name="choroid"),
        ], "tester")
    assert raised.value.status_code == 409


def test_unknown_classes_and_foreign_labels_are_rejected(ctx):
    db, ds = ctx["db"], ctx["dataset_id"]
    with pytest.raises(HTTPException, match="no class"):
        mlm.save_mapping(db, ds, KEY, [ClassMappingWrite(model_class="retina", action="discard")], None)
    with pytest.raises(HTTPException, match="not a label of dataset"):
        mlm.save_mapping(db, ds, KEY, [ClassMappingWrite(model_class="region", action="map", label_id=999)], None)


def test_deleting_a_label_unmaps_its_classes(ctx):
    db, ds = ctx["db"], ctx["dataset_id"]
    mlm.save_mapping(db, ds, KEY, [
        ClassMappingWrite(model_class="vessel", action="create"),
    ], "tester")
    assert db.query(ModelLabelMappings).count() == 1
    # SQLite enforces the ON DELETE CASCADE only with foreign keys on; emulate it the
    # way Postgres does it, by deleting the row with its label.
    label = db.query(Labels).filter_by(name="Choroidal vessels").one()
    db.query(ModelLabelMappings).filter_by(label_id=label.id).delete()
    db.delete(label)
    db.commit()
    assert _by_class(mlm.get_mapping(db, ds, KEY))["vessel"] == ("unmapped", None, "none")


def test_apply_labels_predictions_and_discards_keep_their_children(ctx):
    db, ds = ctx["db"], ctx["dataset_id"]
    mlm.save_mapping(db, ds, KEY, [
        ClassMappingWrite(model_class="region", action="discard"),
        ClassMappingWrite(model_class="vessel", action="create"),
    ], "tester")
    vessels = db.query(Labels).filter_by(name="Choroidal vessels").one()

    out = mlm.apply_mapping(db, ds, KEY, _prediction())
    # The discarded choroid's vessels take its place; the fovea is mapped by name.
    assert [(c.model_class, c.label_id) for c in out] == [
        ("vessel", vessels.id), ("vessel", vessels.id), ("fovea", ctx["fovea"].id),
    ]


def test_apply_leaves_unmapped_classes_unlabelled_and_class_agnostic_output_alone(ctx):
    db, ds = ctx["db"], ctx["dataset_id"]
    region, fovea = mlm.apply_mapping(db, ds, KEY, _prediction())
    assert region.label_id is None and [v.label_id for v in region.children] == [None, None]
    assert fovea.label_id == ctx["fovea"].id

    plain = [Contour(x=[0, 1, 1], y=[0, 0, 1], label_id=42)]
    assert mlm.apply_mapping(db, ds, "sam2", plain)[0].label_id == 42


def test_flatten_cuts_the_tree():
    flat = mlm.flatten(_prediction())
    assert [c.model_class for c in flat] == ["region", "vessel", "vessel", "fovea"]
    assert all(c.children == [] for c in flat)


def test_catalog_offers_classed_models_for_mapped_labels_only(ctx, monkeypatch):
    db, ds = ctx["db"], ctx["dataset_id"]
    monkeypatch.setattr(planning, "list_available_models", lambda task: {"result": [INFO]})
    monkeypatch.setattr(planning, "strategy_options", lambda db, dataset_id: [])
    (option,) = planning.model_catalog(db, ds, tasks=("instance-segmentation",)).models
    assert [c.key for c in option.classes] == ["region", "vessel", "fovea"]
    assert option.label_ids == [ctx["fovea"].id]
    assert option.predicts(ctx["fovea"].id)
    # Unmapped: a classed model is never class-agnostic, even with no label mapped.
    assert not option.predicts(ctx["choroid"].id)
    assert "Model Orchestration" in option.not_predicted_reason("Choroid")
    assert not option.trained_on_dataset


def test_batch_filter_keeps_only_the_steps_label_for_classed_output():
    step = ResolvedStep(
        label_id=7, model_registry_key=KEY, task="instance-segmentation",
        level=0, parent_label_id=None, label_name="Choroid", model_name="M",
        model_label_ids=[],  # a plan made before the mapping existed
    )
    kept = filter_for_step([
        Contour(x=[0, 1, 1], y=[0, 0, 1], model_class="region", label_id=7),
        Contour(x=[0, 1, 1], y=[0, 0, 1], model_class="fovea", label_id=8),
        Contour(x=[0, 1, 1], y=[0, 0, 1], model_class="vessel", label_id=None),
    ], step)
    assert [c.model_class for c in kept] == ["region"]

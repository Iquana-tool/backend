"""Unit tests for the annotation queue service (`app.services.annotation_queue`).

Covers the plain sorting methods, persistence (upsert per dataset+user), settings
validation, the method registry's self-description and the summary. Driven directly
against the service with a temp SQLite database — the HTTP wiring is the same
`require()` dependency already covered by the permission-route tests.
"""
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.annotation_queues  # noqa: F401
import app.database.contours  # noqa: F401
import app.database.dataset_members  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.rejections  # noqa: F401
import app.database.users  # noqa: F401
from app.database.annotation_queues import AnnotationQueues
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.masks import Masks
from app.database.users import Users
from app.schemas.annotation_queue import QueueSettings
from app.services.annotation_queue import (
    build_and_save_queue,
    get_saved_queue,
    method_options,
    summarize,
    trained_models_for_dataset,
)


def settings(method, **kwargs):
    return QueueSettings(method=method, **kwargs)


@pytest.fixture
def ctx(tmp_path):
    """A dataset with three images (each with a mask) and one annotator."""
    engine = create_engine(f"sqlite:///{tmp_path / 'queue.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    db.add(Users(username="ann", hashed_password="x"))
    ds = Datasets(name="ds", description="", dataset_type="image",
                  folder_path="/tmp/ds", created_by="ann")
    db.add(ds)
    db.flush()

    image_ids = []
    for name in ("a.png", "b.png", "c.png"):
        img = Images(dataset_id=ds.id, file_name=name, file_path=f"/tmp/{name}",
                     thumbnail_file_path=f"/tmp/t_{name}", width=10, height=10,
                     color_mode="RGB")
        db.add(img)
        db.flush()
        db.add(Masks(image_id=img.id, fully_annotated=False, file_path=f"/tmp/m_{name}"))
        image_ids.append(img.id)
    db.commit()

    yield db, ds, image_ids
    db.close()


def test_as_uploaded_preserves_upload_order_and_flips_descending(ctx):
    db, ds, image_ids = ctx
    queue = build_and_save_queue(ds.id, "ann", settings("as_uploaded"), db)
    assert queue.image_ids == image_ids
    assert queue.total == len(image_ids)
    assert queue.settings.method == "as_uploaded"
    flipped = build_and_save_queue(ds.id, "ann", settings("as_uploaded", direction="descending"), db)
    assert flipped.image_ids == image_ids[::-1]


def test_random_is_a_permutation(ctx):
    db, ds, image_ids = ctx
    queue = build_and_save_queue(ds.id, "ann", settings("random"), db)
    assert sorted(queue.image_ids) == sorted(image_ids)


def test_build_persists_settings_and_upserts(ctx):
    db, ds, image_ids = ctx
    assert get_saved_queue(ds.id, "ann", db) is None

    build_and_save_queue(ds.id, "ann", settings("as_uploaded", show_builder=False), db)
    saved = get_saved_queue(ds.id, "ann", db)
    assert saved.image_ids == image_ids
    assert saved.settings.show_builder is False

    # Rebuilding with another method overwrites the single row, not appends.
    build_and_save_queue(ds.id, "ann", settings("random"), db)
    assert db.query(AnnotationQueues).count() == 1
    assert get_saved_queue(ds.id, "ann", db).settings.method == "random"
    assert get_saved_queue(ds.id, "ann", db).settings.show_builder is True


@pytest.mark.parametrize("bad", [
    settings("nope"),
    settings("random", direction="descending"),        # Random has no direction
    settings("uncertainty"),                            # needs a trained model
    settings("diversity"),                              # needs an embedding model
    settings("as_uploaded", max_confidence=0.5),        # the filter needs a trained model
    settings("as_uploaded", round_robin_with="nope"),
])
def test_invalid_settings_are_rejected_and_nothing_is_saved(ctx, bad):
    db, ds, _ = ctx
    with pytest.raises(HTTPException) as exc:
        build_and_save_queue(ds.id, "ann", bad, db)
    assert exc.value.status_code == 400
    assert db.query(AnnotationQueues).count() == 0


def test_summary_reports_total_and_saved_settings(ctx):
    db, ds, image_ids = ctx
    before = summarize(ds.id, "ann", db)
    assert before.total == len(image_ids)
    assert before.has_saved_queue is False
    assert before.saved_settings is None

    build_and_save_queue(ds.id, "ann", settings("as_uploaded"), db)
    after = summarize(ds.id, "ann", db)
    assert after.has_saved_queue is True
    assert after.saved_settings.method == "as_uploaded"


def test_methods_describe_their_needs_and_directions():
    options = {option.key: option for option in method_options()}
    assert list(options) == ["as_uploaded", "random", "diversity", "density", "uncertainty"]
    assert options["diversity"].needs == ["embedding_model"]
    assert options["density"].needs == ["embedding_model"]
    assert options["uncertainty"].needs == ["trained_model"]
    assert options["random"].directions == ["ascending"]
    assert options["uncertainty"].directions[0] == "descending"
    # Every supported direction says what it puts first, and the sentences differ.
    for option in options.values():
        assert list(option.descriptions) == option.directions
        assert len(set(option.descriptions.values())) == len(option.directions)


def test_a_registered_method_appears_without_other_changes(ctx, monkeypatch):
    from app.services import queue_methods

    score = lambda c: {u.key: u.key for u in c.units}  # noqa: E731
    monkeypatch.setitem(queue_methods.METHODS, "reverse_id",
                        queue_methods.QueueMethod("reverse_id", "Reverse",
                                                  {"descending": "Highest id first."}, score))
    assert "reverse_id" in {o.key for o in method_options()}

    db, ds, image_ids = ctx
    assert build_and_save_queue(ds.id, "ann", settings("reverse_id"), db).image_ids \
        == image_ids[::-1]


def test_trained_models_are_models_trained_on_this_dataset():
    models = [
        {"registry_key": "trained-a", "name": "Corals v3", "tags": {"trained_on_dataset_id": "4"}},
        {"registry_key": "trained-b", "name": "Other", "tags": {"trained_on_dataset_id": "5"}},
        {"registry_key": "mask2former", "name": "Mask2Former", "tags": {"trainable": "true"}},
        {"registry_key": "legacy", "tags": {"dataset_id": "4"}},
    ]
    assert [(m.registry_key, m.name) for m in trained_models_for_dataset(models, 4)] == [
        ("trained-a", "Corals v3"), ("legacy", "legacy")]

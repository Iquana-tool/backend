"""Unit tests for the annotation queue service (`app.services.annotation_queue`).

Covers the ordering strategies, persistence (upsert per dataset+user), strategy
validation (unknown / unavailable rejected), and the summary. Driven directly
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
from app.services.annotation_queue import (
    build_and_save_queue,
    get_saved_queue,
    strategy_options,
    summarize,
)


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


def test_as_uploaded_preserves_upload_order(ctx):
    db, ds, image_ids = ctx
    queue = build_and_save_queue(ds.id, "ann", "as_uploaded", db)
    assert queue.image_ids == image_ids
    assert queue.total == len(image_ids)
    assert queue.strategy == "as_uploaded"


def test_random_is_a_permutation(ctx):
    db, ds, image_ids = ctx
    queue = build_and_save_queue(ds.id, "ann", "random", db)
    assert sorted(queue.image_ids) == sorted(image_ids)


def test_build_persists_and_upserts(ctx):
    db, ds, image_ids = ctx
    assert get_saved_queue(ds.id, "ann", db) is None

    build_and_save_queue(ds.id, "ann", "as_uploaded", db)
    saved = get_saved_queue(ds.id, "ann", db)
    assert saved is not None
    assert saved.strategy == "as_uploaded"
    assert saved.image_ids == image_ids

    # Rebuilding with another strategy overwrites the single row, not appends.
    build_and_save_queue(ds.id, "ann", "random", db)
    assert db.query(AnnotationQueues).count() == 1
    assert get_saved_queue(ds.id, "ann", db).strategy == "random"


def test_unavailable_strategy_is_rejected(ctx):
    db, ds, _ = ctx
    with pytest.raises(HTTPException) as exc:
        build_and_save_queue(ds.id, "ann", "diversity", db)
    assert exc.value.status_code == 400
    # Nothing persisted for a rejected build.
    assert db.query(AnnotationQueues).count() == 0


def test_unknown_strategy_is_rejected(ctx):
    db, ds, _ = ctx
    with pytest.raises(HTTPException) as exc:
        build_and_save_queue(ds.id, "ann", "nope", db)
    assert exc.value.status_code == 400


def test_summary_reports_total_and_saved_state(ctx):
    db, ds, image_ids = ctx
    before = summarize(ds.id, "ann", db)
    assert before.total == len(image_ids)
    assert before.has_saved_queue is False
    assert before.saved_strategy is None
    assert any(s.key == "as_uploaded" for s in before.strategies)

    build_and_save_queue(ds.id, "ann", "as_uploaded", db)
    after = summarize(ds.id, "ann", db)
    assert after.has_saved_queue is True
    assert after.saved_strategy == "as_uploaded"


def test_diversity_option_is_a_placeholder():
    options = {option.key: option for option in strategy_options()}
    assert options["as_uploaded"].available is True
    assert options["random"].available is True
    assert options["diversity"].available is False

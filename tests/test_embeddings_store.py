"""Store-layer tests for the embeddings table (app.database.embeddings).

Runs on a temp-file SQLite database wired to a fresh engine/session -- the same house
pattern as the other DB tests. SQLite exercises the portable-column fallback (vectors
stored as JSON) and the NumPy brute-force branch of ``search_similar``; the pgvector
``<=>`` path only runs on PostgreSQL and is not covered here.

Covers: upsert insert + idempotent replace, the exactly-one-subject CHECK, cosine search
ordering, dataset scoping, id exclusion, and ON DELETE CASCADE from both subjects.
"""
import numpy as np
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.contours  # noqa: F401
import app.database.users  # noqa: F401
import app.database.embeddings  # noqa: F401
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.embeddings import (
    EMBEDDING_DIM,
    SUBJECT_CONTOUR,
    SUBJECT_IMAGE,
    Embeddings,
    search_similar,
    upsert_embedding,
)
from app.database.images import Images
from app.database.masks import Masks
from app.database.users import Users

DIM = EMBEDDING_DIM


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


def _axis(*components: tuple[int, float]) -> list[float]:
    """A DIM-vector that is zero except for the given ``(index, value)`` pairs."""
    v = np.zeros(DIM, dtype=np.float64)
    for i, val in components:
        v[i] = val
    return v.tolist()


def _contour(session, mask_id: int) -> Contours:
    c = Contours(mask_id=mask_id, added_by="u", confidence_score=1.0,
                 area=0.0, perimeter=0.0, circularity=0.0, diameter=0.0,
                 x=[0.0, 0.1, 0.1], y=[0.0, 0.0, 0.1])
    session.add(c)
    session.flush()
    return c


def _seed_dataset(session, name: str) -> tuple[Datasets, Images, Masks]:
    ds = Datasets(name=name, description="", dataset_type="image",
                  folder_path=f"/tmp/{name}", created_by="u")
    session.add(ds)
    session.flush()
    img = Images(dataset_id=ds.id, file_name="a.png", file_path="/tmp/a.png",
                 thumbnail_file_path="/tmp/t.png", width=100, height=100,
                 color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    session.add(img)
    session.flush()
    mask = Masks(image_id=img.id, fully_annotated=False, file_path="/tmp/m.png")
    session.add(mask)
    session.flush()
    return ds, img, mask


@pytest.fixture
def seed(session):
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    session.flush()
    ds, img, mask = _seed_dataset(session, "A")
    return session, ds, img, mask


def test_upsert_inserts_then_replaces_in_place(seed):
    session, ds, img, mask = seed

    row = upsert_embedding(session, image_id=img.id, kind="image_cls",
                           model_id="dinov3-b", vector=_axis((0, 1.0)))
    assert row.id is not None
    assert row.dim == DIM
    assert session.query(Embeddings).count() == 1

    # Re-embedding the same (subject, kind, model) overwrites in place -- no duplicate row.
    again = upsert_embedding(session, image_id=img.id, kind="image_cls",
                             model_id="dinov3-b", vector=_axis((1, 1.0)))
    assert again.id == row.id
    assert session.query(Embeddings).count() == 1
    assert session.get(Embeddings, row.id).vector[1] == pytest.approx(1.0)

    # A different model_id is a distinct generation -> a new row.
    upsert_embedding(session, image_id=img.id, kind="image_cls",
                     model_id="dinov3-l", vector=_axis((2, 1.0)))
    assert session.query(Embeddings).count() == 2


def test_upsert_requires_exactly_one_subject(seed):
    session, ds, img, mask = seed
    c = _contour(session, mask.id)
    with pytest.raises(ValueError):
        upsert_embedding(session, image_id=img.id, contour_id=c.id,
                         kind="x", model_id="m", vector=_axis((0, 1.0)))
    with pytest.raises(ValueError):
        upsert_embedding(session, kind="x", model_id="m", vector=_axis((0, 1.0)))


def test_wrong_dimension_rejected(seed):
    session, ds, img, mask = seed
    with pytest.raises(ValueError):
        upsert_embedding(session, image_id=img.id, kind="x", model_id="m",
                         vector=[1.0, 2.0, 3.0])


def test_check_constraint_blocks_two_subjects(seed):
    session, ds, img, mask = seed
    c = _contour(session, mask.id)
    # Bypass the upsert guard and hit the DB CHECK directly.
    session.add(Embeddings(image_id=img.id, contour_id=c.id, kind="x",
                           model_id="m", dim=DIM, vector=_axis((0, 1.0))))
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def test_search_orders_by_cosine_distance(seed):
    session, ds, img, mask = seed
    near = _contour(session, mask.id)
    mid = _contour(session, mask.id)
    far = _contour(session, mask.id)

    upsert_embedding(session, contour_id=near.id, kind="region_mean",
                     model_id="m", vector=_axis((0, 1.0)))                 # ~ query
    upsert_embedding(session, contour_id=mid.id, kind="region_mean",
                     model_id="m", vector=_axis((0, 1.0), (1, 1.0)))       # 45 deg off
    upsert_embedding(session, contour_id=far.id, kind="region_mean",
                     model_id="m", vector=_axis((1, 1.0)))                 # orthogonal

    hits = search_similar(session, _axis((0, 1.0)), subject_type=SUBJECT_CONTOUR,
                          kind="region_mean", model_id="m", top_k=10)

    assert [h.subject_id for h in hits] == [near.id, mid.id, far.id]
    assert hits[0].distance == pytest.approx(0.0, abs=1e-9)
    assert hits[2].distance == pytest.approx(1.0, abs=1e-9)
    assert hits[0].distance <= hits[1].distance <= hits[2].distance

    # top_k caps the result set.
    assert len(search_similar(session, _axis((0, 1.0)), subject_type=SUBJECT_CONTOUR,
                              kind="region_mean", model_id="m", top_k=2)) == 2


def test_search_filters_by_kind_and_model(seed):
    session, ds, img, mask = seed
    c = _contour(session, mask.id)
    upsert_embedding(session, contour_id=c.id, kind="region_mean",
                     model_id="m", vector=_axis((0, 1.0)))

    # Wrong kind and wrong model both yield no candidates.
    assert search_similar(session, _axis((0, 1.0)), subject_type=SUBJECT_CONTOUR,
                          kind="other", model_id="m") == []
    assert search_similar(session, _axis((0, 1.0)), subject_type=SUBJECT_CONTOUR,
                          kind="region_mean", model_id="other") == []
    # Right subject type matters too: an image_cls query must not see contour rows.
    assert search_similar(session, _axis((0, 1.0)), subject_type=SUBJECT_IMAGE,
                          kind="region_mean", model_id="m") == []


def test_search_dataset_scoping_and_exclusion(session):
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    session.flush()
    ds_a, img_a, mask_a = _seed_dataset(session, "A")
    ds_b, img_b, mask_b = _seed_dataset(session, "B")

    ca = _contour(session, mask_a.id)
    cb = _contour(session, mask_b.id)
    upsert_embedding(session, contour_id=ca.id, kind="region_mean",
                     model_id="m", vector=_axis((0, 1.0)))
    upsert_embedding(session, contour_id=cb.id, kind="region_mean",
                     model_id="m", vector=_axis((0, 1.0)))

    # Scoped to dataset A: only A's contour (reached via mask -> image -> dataset).
    hits = search_similar(session, _axis((0, 1.0)), subject_type=SUBJECT_CONTOUR,
                          kind="region_mean", model_id="m", dataset_id=ds_a.id)
    assert [h.subject_id for h in hits] == [ca.id]

    # Excluding the only in-scope subject yields nothing.
    assert search_similar(session, _axis((0, 1.0)), subject_type=SUBJECT_CONTOUR,
                          kind="region_mean", model_id="m", dataset_id=ds_a.id,
                          exclude_ids=[ca.id]) == []


def test_cascade_delete_removes_embeddings(seed):
    session, ds, img, mask = seed
    c = _contour(session, mask.id)
    upsert_embedding(session, image_id=img.id, kind="image_cls",
                     model_id="m", vector=_axis((0, 1.0)))
    upsert_embedding(session, contour_id=c.id, kind="region_mean",
                     model_id="m", vector=_axis((0, 1.0)))
    session.commit()
    assert session.query(Embeddings).count() == 2

    # Deleting the contour cascades to its embedding (DB-level ON DELETE CASCADE).
    session.delete(c)
    session.commit()
    assert session.query(Embeddings).count() == 1
    assert session.query(Embeddings).filter(Embeddings.contour_id.isnot(None)).count() == 0

    # Deleting the image cascades to its embedding too.
    session.delete(session.get(Images, img.id))
    session.commit()
    assert session.query(Embeddings).count() == 0

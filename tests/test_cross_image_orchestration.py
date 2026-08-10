"""Tests for the cross-image orchestration core (build_cross_image_request).

Temp-file SQLite. Exercises the sync retrieve -> resolve -> assemble step end-to-end over a
seeded embedding store, without the ai-service (the HTTP call lives in the route). Needs the
toolbox's new cross-image schemas, so run with the local toolbox on PYTHONPATH:

    PYTHONPATH=<repo>/iquana-toolbox/src backend/.venv/Scripts/python.exe -m pytest \
        tests/test_cross_image_orchestration.py
"""
import numpy as np
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
import app.database.embeddings  # noqa: F401
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.embeddings import EMBEDDING_DIM, upsert_embedding
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.services.cross_image_orchestration import build_cross_image_request

MODEL = "dinov3-b"


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


def _axis(*components):
    v = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    for i, val in components:
        v[i] = val
    return v.tolist()


def _image(session, dataset_id, name):
    img = Images(dataset_id=dataset_id, file_name=f"{name}.png", file_path=f"/data/{name}.png",
                 thumbnail_file_path="/tmp/t.png", width=100, height=100, color_mode="RGB",
                 scale_x=1.0, scale_y=1.0, unit="px")
    session.add(img)
    session.flush()
    mask = Masks(image_id=img.id, fully_annotated=False, file_path=f"/tmp/{name}_m.png")
    session.add(mask)
    session.flush()
    return img, mask


def _contour(session, mask_id, label_id=None):
    c = Contours(mask_id=mask_id, added_by="u", confidence_score=1.0, label_id=label_id,
                 area=0.0, perimeter=0.0, circularity=0.0, diameter=0.0,
                 x=[0.1, 0.6, 0.6], y=[0.1, 0.1, 0.6])
    session.add(c)
    session.flush()
    return c


@pytest.fixture
def world(session):
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    session.flush()
    ds = Datasets(name="A", description="", dataset_type="image", folder_path="/tmp/A", created_by="u")
    session.add(ds)
    session.flush()
    label = Labels(dataset_id=ds.id, parent_id=None, name="coral", value=1)
    session.add(label)
    session.flush()

    target, _ = _image(session, ds.id, "target")
    near, near_mask = _image(session, ds.id, "near")
    far, far_mask = _image(session, ds.id, "far")
    # Scene embeddings: target query; near close, far orthogonal.
    upsert_embedding(session, image_id=target.id, kind="image_cls", model_id=MODEL, vector=_axis((0, 1.0)))
    upsert_embedding(session, image_id=near.id, kind="image_cls", model_id=MODEL, vector=_axis((0, 1.0), (1, 0.1)))
    upsert_embedding(session, image_id=far.id, kind="image_cls", model_id=MODEL, vector=_axis((1, 1.0)))

    return dict(session=session, ds=ds, label=label, target=target,
                near=near, near_mask=near_mask, far=far, far_mask=far_mask)


def test_global_scene_builds_request_with_resolved_exemplars(world):
    s = world["session"]
    cb = _contour(s, world["near_mask"].id, label_id=world["label"].id)
    _contour(s, world["far_mask"].id, label_id=world["label"].id)

    request, matches = build_cross_image_request(
        s, target_image_id=world["target"].id, strategy="global_scene",
        concept_label_id=world["label"].id, model_id=MODEL,
        cross_image_model_key="sam3",
    )
    # One exemplar image by default, and it is the best-ranked one (near scene, not far).
    assert [m.contour_id for m in matches] == [cb.id]
    # Target + exemplar resolved to their image URLs; concept carried as a text prompt.
    assert request.image_url == "/data/target.png"
    assert request.model_registry_key == "sam3"
    assert [ex.image_url for ex in request.exemplars] == ["/data/near.png"]
    assert request.concept is not None and request.concept.name == "coral"
    # Each exemplar carries a decodable mask.
    assert request.exemplars[0].mask.mask.shape == (100, 100)


def test_max_exemplar_images_admits_more_images_best_first(world):
    s = world["session"]
    cb = _contour(s, world["near_mask"].id, label_id=world["label"].id)
    cc = _contour(s, world["far_mask"].id, label_id=world["label"].id)

    request, matches = build_cross_image_request(
        s, target_image_id=world["target"].id, strategy="global_scene",
        concept_label_id=world["label"].id, max_exemplar_images=2, model_id=MODEL,
    )
    assert [m.contour_id for m in matches] == [cb.id, cc.id]
    assert [ex.image_url for ex in request.exemplars] == ["/data/near.png", "/data/far.png"]


def test_second_object_in_the_same_image_is_dropped(world):
    """One tile per exemplar: a second object in an already-picked image adds no image."""
    s = world["session"]
    first = _contour(s, world["near_mask"].id, label_id=world["label"].id)
    _contour(s, world["near_mask"].id, label_id=world["label"].id)

    request, matches = build_cross_image_request(
        s, target_image_id=world["target"].id, strategy="global_scene",
        concept_label_id=world["label"].id, max_exemplar_images=2, model_id=MODEL,
    )
    # Both objects rank, but the far image has none -- so only the near image's best survives.
    assert [m.contour_id for m in matches] == [first.id]
    assert [ex.image_url for ex in request.exemplars] == ["/data/near.png"]


def test_no_exemplars_returns_none(world):
    s = world["session"]
    # No contours seeded -> retrieval finds images but no exemplar contours.
    request, matches = build_cross_image_request(
        s, target_image_id=world["target"].id, strategy="global_scene", model_id=MODEL,
    )
    assert request is None and matches == []


def test_missing_target_image_raises_404(world):
    s = world["session"]
    with pytest.raises(HTTPException) as exc:
        build_cross_image_request(s, target_image_id=999999, strategy="global_scene", model_id=MODEL)
    assert exc.value.status_code == 404


def test_concept_region_uses_query_contour_embedding(world):
    s = world["session"]
    # The candidate sits in a different image than the query source, so the one-per-image
    # thinning cannot confuse the two.
    ca = _contour(s, world["far_mask"].id, label_id=world["label"].id)
    src = _contour(s, world["near_mask"].id, label_id=world["label"].id)  # the query source
    upsert_embedding(s, contour_id=ca.id, kind="region_mean", model_id=MODEL, vector=_axis((0, 1.0)))
    upsert_embedding(s, contour_id=src.id, kind="region_mean", model_id=MODEL, vector=_axis((0, 1.0)))

    request, matches = build_cross_image_request(
        s, target_image_id=world["target"].id, strategy="concept_region",
        concept_label_id=world["label"].id, query_contour_id=src.id,
        max_exemplar_images=2, model_id=MODEL,
    )
    assert request is not None
    assert ca.id in {m.contour_id for m in matches}


def test_concept_region_without_query_embedding_raises_400(world):
    s = world["session"]
    orphan = _contour(s, world["near_mask"].id, label_id=world["label"].id)  # no region_mean stored
    with pytest.raises(HTTPException) as exc:
        build_cross_image_request(
            s, target_image_id=world["target"].id, strategy="concept_region",
            query_contour_id=orphan.id, model_id=MODEL,
        )
    assert exc.value.status_code == 400

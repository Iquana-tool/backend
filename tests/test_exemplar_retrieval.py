"""Tests for the exemplar-retrieval strategy registry (app.services.exemplar_retrieval).

Temp-file SQLite (house pattern). Seeds a dataset with a target image plus candidate images,
their contours/labels, and precomputed embeddings, then exercises both concrete strategies,
the concept filter, the missing-input errors, and the dispatcher/options surface. The SQLite
NumPy branch of the store's cosine search is what runs here (pgvector's is Postgres-only).
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
from app.services.exemplar_retrieval import (
    RETRIEVAL_STRATEGIES,
    RetrievalQuery,
    retrieve_exemplars,
    strategy_options,
)

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


def _axis(*components: tuple[int, float]) -> list[float]:
    v = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    for i, val in components:
        v[i] = val
    return v.tolist()


def _image(session, dataset_id: int, name: str) -> tuple[Images, Masks]:
    img = Images(dataset_id=dataset_id, file_name=f"{name}.png", file_path=f"/tmp/{name}.png",
                 thumbnail_file_path="/tmp/t.png", width=100, height=100, color_mode="RGB",
                 scale_x=1.0, scale_y=1.0, unit="px")
    session.add(img)
    session.flush()
    mask = Masks(image_id=img.id, fully_annotated=False, file_path=f"/tmp/{name}_m.png")
    session.add(mask)
    session.flush()
    return img, mask


def _contour(session, mask_id: int, label_id: int | None = None) -> Contours:
    c = Contours(mask_id=mask_id, added_by="u", confidence_score=1.0, label_id=label_id,
                 area=0.0, perimeter=0.0, circularity=0.0, diameter=0.0,
                 x=[0.0, 0.1, 0.1], y=[0.0, 0.0, 0.1])
    session.add(c)
    session.flush()
    return c


@pytest.fixture
def world(session):
    """A dataset with target image A, candidates B (near A) and C (far), plus two labels."""
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    session.flush()
    ds = Datasets(name="A", description="", dataset_type="image", folder_path="/tmp/A", created_by="u")
    session.add(ds)
    session.flush()
    l1 = Labels(dataset_id=ds.id, parent_id=None, name="coral", value=1)
    l2 = Labels(dataset_id=ds.id, parent_id=None, name="sponge", value=2)
    session.add_all([l1, l2])
    session.flush()

    img_a, _ = _image(session, ds.id, "A")
    img_b, mask_b = _image(session, ds.id, "B")
    img_c, mask_c = _image(session, ds.id, "C")

    # Scene embeddings: A is the query; B is close to A, C is orthogonal (far).
    upsert_embedding(session, image_id=img_a.id, kind="image_cls", model_id=MODEL, vector=_axis((0, 1.0)))
    upsert_embedding(session, image_id=img_b.id, kind="image_cls", model_id=MODEL, vector=_axis((0, 1.0), (1, 0.1)))
    upsert_embedding(session, image_id=img_c.id, kind="image_cls", model_id=MODEL, vector=_axis((1, 1.0)))

    return dict(session=session, ds=ds, l1=l1, l2=l2, img_a=img_a, img_b=img_b, img_c=img_c,
                mask_b=mask_b, mask_c=mask_c)


# --- global_scene ---------------------------------------------------------- #
def test_global_scene_orders_by_source_image_similarity(world):
    s, ds = world["session"], world["ds"]
    cb = _contour(s, world["mask_b"].id, label_id=world["l1"].id)
    cc = _contour(s, world["mask_c"].id, label_id=world["l1"].id)

    matches = retrieve_exemplars(
        s, "global_scene",
        RetrievalQuery(dataset_id=ds.id, target_image_id=world["img_a"].id, top_k=5),
        model_id=MODEL,
    )
    assert [m.contour_id for m in matches] == [cb.id, cc.id]
    assert matches[0].image_id == world["img_b"].id
    assert matches[0].score > matches[1].score  # B's scene beats C's

    # top_k caps the shortlist.
    capped = retrieve_exemplars(
        s, "global_scene",
        RetrievalQuery(dataset_id=ds.id, target_image_id=world["img_a"].id, top_k=1),
        model_id=MODEL,
    )
    assert [m.contour_id for m in capped] == [cb.id]


def test_global_scene_concept_filter(world):
    s, ds = world["session"], world["ds"]
    cb = _contour(s, world["mask_b"].id, label_id=world["l1"].id)
    _contour(s, world["mask_c"].id, label_id=world["l2"].id)  # different concept

    matches = retrieve_exemplars(
        s, "global_scene",
        RetrievalQuery(dataset_id=ds.id, target_image_id=world["img_a"].id,
                       concept_label_id=world["l1"].id, top_k=5),
        model_id=MODEL,
    )
    assert [m.contour_id for m in matches] == [cb.id]  # C's sponge is filtered out


def test_global_scene_requires_target_and_embedding(world):
    s, ds = world["session"], world["ds"]
    with pytest.raises(ValueError):
        retrieve_exemplars(s, "global_scene", RetrievalQuery(dataset_id=ds.id), model_id=MODEL)
    # An image with no image_cls embedding is a clear error, not an empty result.
    img_d, _ = _image(s, ds.id, "D")
    with pytest.raises(ValueError):
        retrieve_exemplars(
            s, "global_scene",
            RetrievalQuery(dataset_id=ds.id, target_image_id=img_d.id), model_id=MODEL,
        )


# --- concept_region -------------------------------------------------------- #
def test_concept_region_ranks_and_filters_by_concept(world):
    s, ds = world["session"], world["ds"]
    ca = _contour(s, world["mask_b"].id, label_id=world["l1"].id)
    cb = _contour(s, world["mask_b"].id, label_id=world["l1"].id)
    cc = _contour(s, world["mask_c"].id, label_id=world["l2"].id)  # other concept, identical vec
    upsert_embedding(s, contour_id=ca.id, kind="region_mean", model_id=MODEL, vector=_axis((0, 1.0)))
    upsert_embedding(s, contour_id=cb.id, kind="region_mean", model_id=MODEL, vector=_axis((0, 1.0), (1, 1.0)))
    upsert_embedding(s, contour_id=cc.id, kind="region_mean", model_id=MODEL, vector=_axis((0, 1.0)))

    # Filtered to concept l1: ca (nearest) then cb; cc excluded despite matching the query.
    filtered = retrieve_exemplars(
        s, "concept_region",
        RetrievalQuery(dataset_id=ds.id, concept_label_id=world["l1"].id,
                       query_vector=_axis((0, 1.0)), top_k=5),
        model_id=MODEL,
    )
    assert [m.contour_id for m in filtered] == [ca.id, cb.id]
    assert filtered[0].image_id == world["img_b"].id
    assert filtered[0].score > filtered[1].score

    # Unfiltered: cc is now eligible and ties ca at the top.
    unfiltered = retrieve_exemplars(
        s, "concept_region",
        RetrievalQuery(dataset_id=ds.id, query_vector=_axis((0, 1.0)), top_k=5),
        model_id=MODEL,
    )
    assert {m.contour_id for m in unfiltered} == {ca.id, cb.id, cc.id}
    assert unfiltered[-1].contour_id == cb.id  # the 45-degree one is last


def test_concept_region_requires_query_vector(world):
    s, ds = world["session"], world["ds"]
    with pytest.raises(ValueError):
        retrieve_exemplars(
            s, "concept_region", RetrievalQuery(dataset_id=ds.id), model_id=MODEL
        )


def test_concept_region_unknown_concept_returns_empty(world):
    s, ds = world["session"], world["ds"]
    matches = retrieve_exemplars(
        s, "concept_region",
        RetrievalQuery(dataset_id=ds.id, concept_label_id=99999, query_vector=_axis((0, 1.0))),
        model_id=MODEL,
    )
    assert matches == []


# --- registry / dispatch --------------------------------------------------- #
def test_strategy_options_expose_required_kinds_and_placeholder():
    options = {o.key: o for o in strategy_options()}
    assert options["global_scene"].required_kinds == ["image_cls"]
    assert options["concept_region"].required_kinds == ["region_mean"]
    assert options["global_scene"].available is True
    assert options["hybrid"].available is False


def test_registry_declared_kinds():
    assert RETRIEVAL_STRATEGIES["global_scene"].required_kinds == ("image_cls",)
    assert RETRIEVAL_STRATEGIES["concept_region"].required_kinds == ("region_mean",)


def test_dispatch_unknown_and_unavailable(world):
    s, ds = world["session"], world["ds"]
    with pytest.raises(HTTPException) as unknown:
        retrieve_exemplars(s, "nope", RetrievalQuery(dataset_id=ds.id), model_id=MODEL)
    assert unknown.value.status_code == 400

    with pytest.raises(HTTPException) as unavailable:
        retrieve_exemplars(s, "hybrid", RetrievalQuery(dataset_id=ds.id), model_id=MODEL)
    assert unavailable.value.status_code == 400

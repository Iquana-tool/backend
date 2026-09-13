"""Tests for the embedding lifecycle engine + backfill (app.services.embedding_lifecycle).

Temp-file SQLite. A fake embed client stands in for the ai-service ``embed`` surface, so the
orchestration -- request building, response filtering, contour->region mapping, idempotent
upsert, backfill missing-only selection, and the gated on-write enqueue -- is verified without
the model, HTTP, or a Celery broker.
"""
import numpy as np
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from iquana_toolbox.schemas.networking.http.services import EmbeddingVector

from app.database import database
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.contours  # noqa: F401
import app.database.users  # noqa: F401
import app.database.embeddings  # noqa: F401
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.embeddings import EMBEDDING_DIM, Embeddings, upsert_embedding
from app.database.images import Images
from app.database.masks import Masks
from app.database.users import Users
from app.services import embedding_lifecycle as lifecycle


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


class FakeEmbedClient:
    """Stands in for EmbeddingService: echoes back a vector per requested image kind / region."""

    def __init__(self, model_id="facebook/dinov3-vitb16"):
        self.model_id = model_id
        self.requests = []

    def request_embeddings(self, request):
        self.requests.append(request)
        out = []
        for kind in request.image_kinds:
            out.append(EmbeddingVector(kind=kind, region_id=None, model_id=self.model_id,
                                       dim=EMBEDDING_DIM, vector=[0.0] * EMBEDDING_DIM))
        for region in request.regions:
            vec = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
            out.append(EmbeddingVector(kind="region_mean", region_id=region.region_id,
                                       model_id=self.model_id, dim=EMBEDDING_DIM, vector=vec))
        return out


def _seed(session, name="A"):
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    session.flush()
    ds = Datasets(name=name, description="", dataset_type="image", folder_path=f"/tmp/{name}", created_by="u")
    session.add(ds)
    session.flush()
    return ds


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


def _contour(session, mask_id, temporary=False):
    c = Contours(mask_id=mask_id, added_by="u", confidence_score=1.0, temporary=temporary,
                 area=0.0, perimeter=0.0, circularity=0.0, diameter=0.0,
                 x=[0.1, 0.6, 0.6], y=[0.1, 0.1, 0.6])
    session.add(c)
    session.flush()
    return c


def test_required_kinds_from_available_strategies():
    kinds = lifecycle.required_kinds()
    assert kinds == {"image_cls", "region_mean"}  # global_scene + concept_region; hybrid excluded


def test_embed_image_upserts_image_cls(session):
    ds = _seed(session)
    img, _ = _image(session, ds.id, "A")
    client = FakeEmbedClient()

    stored = lifecycle.embed_image(session, img, client=client)
    assert len(stored) == 1
    row = session.query(Embeddings).filter_by(image_id=img.id, kind="image_cls").one()
    assert row.model_id == client.model_id and row.dim == EMBEDDING_DIM
    # The request asked only for image kinds, no regions.
    assert client.requests[0].image_kinds == ["image_cls"]
    assert client.requests[0].regions == []
    assert client.requests[0].image_url == "/data/A.png"


def test_embed_contours_maps_regions_and_batches_by_image(session):
    ds = _seed(session)
    img, mask = _image(session, ds.id, "A")
    c1 = _contour(session, mask.id)
    c2 = _contour(session, mask.id)

    client = FakeEmbedClient()
    stored = lifecycle.embed_contours(session, [c1, c2], client=client)

    assert len(stored) == 2
    got = {r.contour_id for r in session.query(Embeddings).filter_by(kind="region_mean").all()}
    assert got == {c1.id, c2.id}
    # One request for the shared image, carrying both regions tagged by contour id.
    assert len(client.requests) == 1
    assert {r.region_id for r in client.requests[0].regions} == {c1.id, c2.id}


def test_embed_contours_skips_temporary_via_caller(session):
    # embed_contours embeds whatever it is given; the write hook is what filters temporary.
    ds = _seed(session)
    img, mask = _image(session, ds.id, "A")
    c = _contour(session, mask.id)
    client = FakeEmbedClient()
    lifecycle.embed_contours(session, [c], client=client)
    assert session.query(Embeddings).filter_by(kind="region_mean").count() == 1


def test_upsert_is_idempotent_on_reembed(session):
    ds = _seed(session)
    img, _ = _image(session, ds.id, "A")
    client = FakeEmbedClient()
    lifecycle.embed_image(session, img, client=client)
    lifecycle.embed_image(session, img, client=client)  # re-embed
    assert session.query(Embeddings).filter_by(image_id=img.id, kind="image_cls").count() == 1


def test_backfill_embeds_only_missing(session):
    ds = _seed(session)
    img1, mask1 = _image(session, ds.id, "img1")
    img2, mask2 = _image(session, ds.id, "img2")
    c1 = _contour(session, mask1.id)
    c2 = _contour(session, mask2.id)
    # Pre-seed embeddings for img1 (image_cls) and c1 (region_mean) under an older model.
    upsert_embedding(session, image_id=img1.id, kind="image_cls", model_id="old", vector=[0.0] * EMBEDDING_DIM)
    upsert_embedding(session, contour_id=c1.id, kind="region_mean", model_id="old", vector=[0.0] * EMBEDDING_DIM)

    client = FakeEmbedClient()
    counts = lifecycle.backfill_embeddings(session, dataset_id=ds.id, client=client)

    assert counts == {"images": 1, "contours": 1}  # only img2 and c2 were missing
    # img2/c2 now embedded; img1/c1 untouched (still the 'old' model row, not duplicated).
    assert session.query(Embeddings).filter_by(image_id=img2.id, kind="image_cls").count() == 1
    assert session.query(Embeddings).filter_by(contour_id=c2.id, kind="region_mean").count() == 1
    assert session.query(Embeddings).filter_by(image_id=img1.id).one().model_id == "old"


def test_enqueue_is_noop_when_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle.embed_image_task, "delay", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(lifecycle, "EMBEDDING_LIFECYCLE_ENABLED", False)
    lifecycle.enqueue_embed_image(1)
    assert calls == []


def test_enqueue_dispatches_when_enabled(monkeypatch):
    img_calls, contour_calls = [], []
    monkeypatch.setattr(lifecycle.embed_image_task, "delay", lambda *a, **k: img_calls.append(a))
    monkeypatch.setattr(lifecycle.embed_contours_task, "delay", lambda *a, **k: contour_calls.append(a))
    monkeypatch.setattr(lifecycle, "EMBEDDING_LIFECYCLE_ENABLED", True)

    lifecycle.enqueue_embed_image(7)
    lifecycle.enqueue_embed_contours([3, 4])
    lifecycle.enqueue_embed_contours([])  # empty -> no dispatch

    assert img_calls == [(7,)]
    assert contour_calls == [([3, 4],)]

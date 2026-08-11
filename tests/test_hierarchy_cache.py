"""Tests for the contour-hierarchy cache that image switching reads through.

Switching images used to rebuild a mask's whole hierarchy from scratch on every visit.
The cache makes a revisit free, which is only safe if it can never hand back a hierarchy
that no longer matches the database. The tests here are almost all about that second
half:

  * a repeat read is served from the cache (the point of the exercise),
  * every write through the database-access layer drops the entry -- including the
    in-place edits (label, approval) that leave the row count untouched,
  * a write made *outside* that layer, as batch inference's Celery worker does, is still
    caught, because the fingerprint checked on every read no longer matches,
  * the client payload carries the contour tree without the two index copies of it that
    ``model_dump`` would also have written.
"""
import asyncio

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.users  # noqa: F401
import app.database.contours  # noqa: F401  (also pulls in contour_metrics)
from app.database.contours import Contours, save_contour_tree
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.services import hierarchy_cache
from app.services.database_access import contours as contours_db
from app.services.database_access import masks as masks_db

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.user import User

WIDTH, HEIGHT = 1000, 1000


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
    # The cache is module-level state shared by every test in the process.
    hierarchy_cache.clear()
    try:
        yield s
    finally:
        hierarchy_cache.clear()
        s.close()
        engine.dispose()


def _rect(cx_px, cy_px, half=20):
    x_px = [cx_px - half, cx_px + half, cx_px + half, cx_px - half]
    y_px = [cy_px - half, cy_px - half, cy_px + half, cy_px + half]
    return ([x / WIDTH for x in x_px], [y / HEIGHT for y in y_px])


def _contour(cx_px, cy_px, label_id=None, half=20):
    x, y = _rect(cx_px, cy_px, half=half)
    return Contour(x=x, y=y, added_by="u", confidence=1.0, label_id=label_id)


def _seed(session):
    """One dataset / image / mask with two root contours, and a user to review with."""
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    ds = Datasets(name="cache", description="", dataset_type="image",
                  folder_path="/tmp/cache", created_by="u")
    session.add(ds)
    session.flush()

    label = Labels(dataset_id=ds.id, name="cell", value=1)
    session.add(label)

    img = Images(dataset_id=ds.id, file_name="a.png", file_path="/tmp/a.png",
                 thumbnail_file_path="/tmp/t.png", width=WIDTH, height=HEIGHT,
                 color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    session.add(img)
    session.flush()

    mask = Masks(image_id=img.id, fully_annotated=False, file_path="/tmp/m.png")
    session.add(mask)
    session.flush()

    for contour in (_contour(100, 100), _contour(500, 500)):
        save_contour_tree(session, contour, mask.id, author_username="u")
    session.commit()
    return mask, label


def _read(mask_id, session):
    return asyncio.run(masks_db.get_cached_contour_hierarchy_of_mask(mask_id, session))


# --------------------------------------------------------------------------- #
# The cache itself
# --------------------------------------------------------------------------- #
def test_repeat_read_is_served_from_cache(session):
    mask, _ = _seed(session)

    first_hierarchy, first_payload = _read(mask.id, session)
    second_hierarchy, second_payload = _read(mask.id, session)

    # Identity, not equality: a rebuilt hierarchy would compare equal but be a new object.
    assert second_hierarchy is first_hierarchy
    assert second_payload is first_payload
    assert len(first_payload["root_contours"]) == 2


def test_payload_carries_the_tree_without_the_index_copies(session):
    mask, _ = _seed(session)

    _, payload = _read(mask.id, session)

    assert set(payload) == {"root_contours"}
    # What ``model_dump`` would also have serialized, at least twice over.
    assert "id_to_contour" not in payload
    assert "label_id_to_contours" not in payload


def test_children_still_reach_the_client(session):
    """The lean payload must not cost nested objects: they ride inside their parent."""
    mask, _ = _seed(session)
    parent_id = session.query(Contours.id).filter_by(mask_id=mask.id).first()[0]
    child = _contour(100, 100, half=5)
    child.parent_id = parent_id
    save_contour_tree(session, child, mask.id, parent_id=parent_id, author_username="u")
    session.commit()

    _, payload = _read(mask.id, session)

    roots = payload["root_contours"]
    assert len(roots) == 2
    nested = [c for root in roots for c in root["children"]]
    assert len(nested) == 1


# --------------------------------------------------------------------------- #
# Invalidation through the database-access layer
# --------------------------------------------------------------------------- #
def test_adding_a_contour_invalidates(session):
    mask, _ = _seed(session)
    _, first = _read(mask.id, session)

    asyncio.run(masks_db.add_contour_to_mask(mask.id, _contour(800, 800), session,
                                             check_hierarchy=False, author_username="u"))
    _, second = _read(mask.id, session)

    assert len(first["root_contours"]) == 2
    assert len(second["root_contours"]) == 3


def test_deleting_a_contour_invalidates(session):
    mask, _ = _seed(session)
    _read(mask.id, session)
    victim = session.query(Contours.id).filter_by(mask_id=mask.id).first()[0]

    asyncio.run(contours_db.delete_contour(victim, session))
    _, payload = _read(mask.id, session)

    assert len(payload["root_contours"]) == 1


def test_label_change_invalidates_even_though_the_rows_are_unchanged(session):
    """The fingerprint cannot see this one -- only the explicit invalidation can."""
    mask, label = _seed(session)
    _read(mask.id, session)
    contour_id = session.query(Contours.id).filter_by(mask_id=mask.id).first()[0]

    asyncio.run(contours_db.modify_contour(contour_id, db=session, label_id=label.id))
    _, payload = _read(mask.id, session)

    labels = {c["label_id"] for c in payload["root_contours"]}
    assert label.id in labels


def test_approval_invalidates_even_though_the_rows_are_unchanged(session):
    mask, _ = _seed(session)
    _read(mask.id, session)
    contour_id = session.query(Contours.id).filter_by(mask_id=mask.id).first()[0]

    user = User(username="u", is_admin=False, owned_datasets=[], accessible_datasets=[])
    asyncio.run(contours_db.review_contour(contour_id, user=user, db=session))
    _, payload = _read(mask.id, session)

    reviewed = {c["id"]: c["reviewed_by"] for c in payload["root_contours"]}
    assert reviewed[contour_id] == ["u"]


def test_wiping_the_mask_invalidates(session):
    mask, _ = _seed(session)
    _read(mask.id, session)

    asyncio.run(masks_db.delete_all_contours_of_mask(mask.id, session))
    _, payload = _read(mask.id, session)

    assert payload["root_contours"] == []


# --------------------------------------------------------------------------- #
# Writes the cache never hears about
# --------------------------------------------------------------------------- #
def test_out_of_band_insert_is_caught_by_the_fingerprint(session):
    """Batch inference writes from a Celery worker, whose invalidations never arrive.

    Simulated by writing straight to the database and skipping the invalidation the
    access layer would normally have done in this process.
    """
    mask, _ = _seed(session)
    _read(mask.id, session)

    save_contour_tree(session, _contour(800, 800), mask.id, author_username="u")
    session.commit()
    # Deliberately no hierarchy_cache.invalidate() -- that is the whole point.

    _, payload = _read(mask.id, session)

    assert len(payload["root_contours"]) == 3


def test_out_of_band_delete_is_caught_by_the_fingerprint(session):
    mask, _ = _seed(session)
    _read(mask.id, session)

    victim = session.query(Contours.id).filter_by(mask_id=mask.id).first()[0]
    session.query(Contours).filter_by(id=victim).delete()
    session.commit()

    _, payload = _read(mask.id, session)

    assert len(payload["root_contours"]) == 1


def test_two_masks_do_not_share_an_entry(session):
    mask, _ = _seed(session)
    image_id = session.query(Masks.image_id).filter_by(id=mask.id).scalar()
    other = Masks(image_id=image_id, fully_annotated=False, file_path="/tmp/m2.png")
    session.add(other)
    session.commit()

    _, first = _read(mask.id, session)
    _, second = _read(other.id, session)

    assert len(first["root_contours"]) == 2
    assert second["root_contours"] == []


def test_cache_evicts_the_least_recently_used_entry(monkeypatch):
    """The cache is bounded, so a long navigation cannot grow it without limit."""
    monkeypatch.setattr(hierarchy_cache, "MAX_ENTRIES", 2)
    hierarchy_cache.clear()
    from iquana_toolbox.schemas.database.contour_hierarchy import ContourHierarchy

    for mask_id in (1, 2, 3):
        hierarchy_cache.put(mask_id, (0, 0), ContourHierarchy())

    assert hierarchy_cache.get(1, (0, 0)) is None
    assert hierarchy_cache.get(2, (0, 0)) is not None
    assert hierarchy_cache.get(3, (0, 0)) is not None
    hierarchy_cache.clear()

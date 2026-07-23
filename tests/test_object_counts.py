"""Tests for the per-class object-count census (total / reviewed / unreviewed) that
``get_quantification_summary`` returns under ``object_counts_per_label_id``.

Uses a temp-file SQLite database. Verifies that the counts:
  * split a label's contours into reviewed (>=1 reviewer) vs unreviewed correctly,
  * are a FULL census that ignores both exclude filters (so ``exclude_unreviewed=True``
    does not zero out the unreviewed count).
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
from app.database.contours import save_contour_tree
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users

from app.services.database_access.datasets import get_quantification_summary

from iquana_toolbox.schemas.database.contours import Contour

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
    try:
        yield s
    finally:
        s.close()
        engine.dispose()


def _rect(cx_px, cy_px, half=5):
    x_px = [cx_px - half, cx_px + half, cx_px + half, cx_px - half]
    y_px = [cy_px - half, cy_px - half, cy_px + half, cy_px + half]
    return ([x / WIDTH for x in x_px], [y / HEIGHT for y in y_px])


def _seed(session):
    ds = Datasets(name="counts-smoke", description="", dataset_type="image",
                  folder_path="/tmp/counts-smoke", created_by="u")
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    session.add(ds)
    session.flush()

    img = Images(dataset_id=ds.id, file_name="a.png", file_path="/tmp/a.png",
                 thumbnail_file_path="/tmp/t.png", width=WIDTH, height=HEIGHT,
                 color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    session.add(img)
    session.flush()

    # One fully-annotated mask, one not, to prove exclude_not_fully_annotated is ignored.
    mask_done = Masks(image_id=img.id, fully_annotated=True, file_path="/tmp/m1.png")
    mask_wip = Masks(image_id=img.id, fully_annotated=False, file_path="/tmp/m2.png")
    session.add_all([mask_done, mask_wip])
    session.flush()

    label = Labels(dataset_id=ds.id, parent_id=None, name="cell", value=1)
    session.add(label)
    session.flush()
    return ds, img, mask_done, mask_wip, label


def _add(session, mask_id, label_id, cx, reviewed):
    px, py = _rect(cx, 500)
    schema = Contour(x=px, y=py, label_id=label_id, added_by="u",
                     reviewed_by=(["u"] if reviewed else []))
    save_contour_tree(session, schema, mask_id)


def test_object_counts_split_reviewed_unreviewed(session):
    ds, img, mask_done, mask_wip, label = _seed(session)

    # 2 reviewed + 1 unreviewed on the finished mask.
    _add(session, mask_done.id, label.id, 100, reviewed=True)
    _add(session, mask_done.id, label.id, 200, reviewed=True)
    _add(session, mask_done.id, label.id, 300, reviewed=False)
    # 1 unreviewed on the not-fully-annotated mask (must still be counted).
    _add(session, mask_wip.id, label.id, 400, reviewed=False)
    session.commit()

    # Even with both filters ON, the census is complete: total 4, reviewed 2, unreviewed 2.
    summary = asyncio.run(get_quantification_summary(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session,
    ))
    counts = summary["object_counts_per_label_id"][str(label.id)]
    assert counts == {"total": 4, "reviewed": 2, "unreviewed": 2}


def test_object_counts_ignore_filters_match_unfiltered(session):
    ds, img, mask_done, mask_wip, label = _seed(session)
    _add(session, mask_done.id, label.id, 100, reviewed=True)
    _add(session, mask_wip.id, label.id, 200, reviewed=False)
    session.commit()

    filtered = asyncio.run(get_quantification_summary(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session,
    ))
    unfiltered = asyncio.run(get_quantification_summary(
        ds.id, exclude_not_fully_annotated=False, exclude_unreviewed=False, db=session,
    ))
    key = str(label.id)
    assert filtered["object_counts_per_label_id"][key] == {"total": 2, "reviewed": 1, "unreviewed": 1}
    assert unfiltered["object_counts_per_label_id"][key] == {"total": 2, "reviewed": 1, "unreviewed": 1}

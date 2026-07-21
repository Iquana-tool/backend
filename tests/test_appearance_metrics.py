"""End-to-end smoke test for Step 3: appearance-tier metrics (mean color / intensity).

Uses a temp-file SQLite database wired to a fresh engine/session (does not touch the real
database) plus a real small PNG written to a temp file, referenced by ``Images.file_path``.
Exercises the full lazy/batched path:
  * ``compute_appearance_metrics_for_dataset`` computes and stores mean_color_rgb /
    mean_color_lab / mean_intensity rows for every contour of a dataset,
  * modifying a contour's geometry marks its appearance rows stale, and a follow-up
    ``only_stale=True`` recompute clears that staleness,
  * ``get_quantification_summary`` surfaces the color metrics with a 3-component
    ``components`` array, exactly like any other metric.
"""
import asyncio

import numpy as np
import pytest
from PIL import Image as PILImage
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# Import the shared declarative base and ALL model modules so metadata is complete.
from app.database import database
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.users  # noqa: F401
import app.database.contours  # noqa: F401  (also pulls in contour_metrics)
from app.database.contour_metrics import ContourMetrics
from app.database.contours import Contours, save_contour_tree
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users

from app.services.quantification import (
    APPEARANCE_METRIC_KEYS,
    compute_appearance_metrics_for_dataset,
)
from app.services.database_access.contours import modify_contour

from iquana_toolbox.schemas.database.contours import Contour

# A non-square image so pixel-space geometry (used to build the mask) matters.
WIDTH, HEIGHT = 100, 60
SCALE_X, SCALE_Y = 1.0, 1.0
UNIT = "px"


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


@pytest.fixture
def image_file(tmp_path):
    """A real small solid-color (green) PNG on disk."""
    path = tmp_path / "solid_green.png"
    arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    arr[:, :, 1] = 200  # solid green-ish
    PILImage.fromarray(arr).save(path)
    return str(path)


def _rect(x_px, y_px):
    return ([x / WIDTH for x in x_px], [y / HEIGHT for y in y_px])


def _seed(session, image_file):
    ds = Datasets(name="appearance-smoke", description="", dataset_type="image",
                  folder_path="/tmp/appearance-smoke", created_by="u")
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    session.add(ds)
    session.flush()

    img = Images(dataset_id=ds.id, file_name="solid_green.png", file_path=image_file,
                 thumbnail_file_path=image_file, width=WIDTH, height=HEIGHT,
                 color_mode="RGB", scale_x=SCALE_X, scale_y=SCALE_Y, unit=UNIT)
    session.add(img)
    session.flush()

    mask = Masks(image_id=img.id, fully_annotated=True, file_path="/tmp/m.png")
    session.add(mask)
    session.flush()

    label = Labels(dataset_id=ds.id, parent_id=None, name="blob", value=1)
    session.add(label)
    session.flush()

    return ds, img, mask, label


def test_compute_stale_invalidation_and_summary(session, image_file):
    ds, img, mask, label = _seed(session, image_file)

    # A rectangle comfortably inside the 100x60 image.
    px, py = _rect([10, 40, 40, 10], [10, 10, 40, 40])
    contour_schema = Contour(x=px, y=py, label_id=label.id, added_by="u", reviewed_by=["u"])
    db_contour = save_contour_tree(session, contour_schema, mask.id)
    session.commit()
    contour_id = db_contour.id

    # --- 1. Batch compute appearance metrics for the whole dataset ------------------
    computed = compute_appearance_metrics_for_dataset(session, ds.id, only_stale=True)
    assert computed > 0

    rows = (
        session.query(ContourMetrics)
        .filter(ContourMetrics.contour_id == contour_id,
                ContourMetrics.metric_key.in_(APPEARANCE_METRIC_KEYS))
        .all()
    )
    by_key = {}
    for row in rows:
        by_key.setdefault(row.metric_key, {})[row.component] = row

    assert set(by_key["mean_color_rgb"]) == {0, 1, 2}
    assert set(by_key["mean_color_lab"]) == {0, 1, 2}
    assert set(by_key["mean_intensity"]) == {0}

    # Solid green (R=0, G=200, B=0) fills the whole contour -> exact means.
    assert by_key["mean_color_rgb"][0].value == pytest.approx(0.0)
    assert by_key["mean_color_rgb"][1].value == pytest.approx(200.0)
    assert by_key["mean_color_rgb"][2].value == pytest.approx(0.0)
    assert all(not r.stale for r in rows)

    # --- 2. Re-running with only_stale=True is a no-op (already fresh) --------------
    recomputed = compute_appearance_metrics_for_dataset(session, ds.id, only_stale=True)
    assert recomputed == 0

    # --- 3. Modifying geometry marks appearance rows stale ---------------------------
    new_px, new_py = _rect([5, 60, 60, 5], [5, 5, 50, 50])
    asyncio.run(modify_contour(contour_id, db=session, x=new_px, y=new_py))

    session.expire_all()
    stale_rows = (
        session.query(ContourMetrics)
        .filter(ContourMetrics.contour_id == contour_id,
                ContourMetrics.metric_key.in_(APPEARANCE_METRIC_KEYS))
        .all()
    )
    assert len(stale_rows) == len(rows)  # rows kept, just flagged
    assert all(r.stale for r in stale_rows)

    # --- 4. Recompute clears staleness (only_stale=True picks it up) -----------------
    recomputed_after_edit = compute_appearance_metrics_for_dataset(session, ds.id, only_stale=True)
    assert recomputed_after_edit > 0
    session.expire_all()
    fresh_rows = (
        session.query(ContourMetrics)
        .filter(ContourMetrics.contour_id == contour_id,
                ContourMetrics.metric_key.in_(APPEARANCE_METRIC_KEYS))
        .all()
    )
    assert all(not r.stale for r in fresh_rows)

    # --- 5. get_quantification_summary surfaces color with a 3-component array -------
    from app.services.database_access import datasets as datasets_db
    summary = asyncio.run(datasets_db.get_quantification_summary(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session
    ))
    label_key = str(label.id)
    assert label_key in summary["metrics"]
    lab_entry = summary["metrics"][label_key]["mean_color_lab"]
    assert len(lab_entry["components"]) == 3
    assert lab_entry["components"][0]["count"] == 1

    rgb_entry = summary["metrics"][label_key]["mean_color_rgb"]
    assert len(rgb_entry["components"]) == 3
    # Green channel (component 1) should be ~200 for the solid-green image.
    assert rgb_entry["components"][1]["mean"] == pytest.approx(200.0, abs=1.0)


def test_new_contour_without_appearance_rows_is_picked_up_by_only_stale(session, image_file):
    """A brand new contour (no appearance rows at all) must count as 'needs compute'."""
    ds, img, mask, label = _seed(session, image_file)
    px, py = _rect([20, 30, 30, 20], [20, 20, 30, 30])
    contour_schema = Contour(x=px, y=py, label_id=label.id, added_by="u", reviewed_by=["u"])
    db_contour = save_contour_tree(session, contour_schema, mask.id)
    session.commit()

    # No appearance rows exist yet (only geometry was dual-written by save_contour_tree).
    assert session.query(ContourMetrics).filter(
        ContourMetrics.contour_id == db_contour.id,
        ContourMetrics.metric_key.in_(APPEARANCE_METRIC_KEYS),
    ).count() == 0

    computed = compute_appearance_metrics_for_dataset(session, ds.id, only_stale=True)
    assert computed > 0
    assert session.query(ContourMetrics).filter(
        ContourMetrics.contour_id == db_contour.id,
        ContourMetrics.metric_key.in_(APPEARANCE_METRIC_KEYS),
    ).count() == 7  # 3 (rgb) + 3 (lab) + 1 (intensity)


def test_missing_image_file_does_not_crash_batch(session, tmp_path):
    """load_image_rgb logs and returns None for a missing file; the batch just skips it."""
    ds = Datasets(name="missing-file", description="", dataset_type="image",
                  folder_path=str(tmp_path), created_by="u")
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    session.add(ds)
    session.flush()

    missing_path = str(tmp_path / "does_not_exist.png")
    img = Images(dataset_id=ds.id, file_name="missing.png", file_path=missing_path,
                 thumbnail_file_path=missing_path, width=WIDTH, height=HEIGHT,
                 color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    session.add(img)
    session.flush()
    mask = Masks(image_id=img.id, fully_annotated=True, file_path="/tmp/m.png")
    session.add(mask)
    session.flush()
    label = Labels(dataset_id=ds.id, parent_id=None, name="blob", value=1)
    session.add(label)
    session.flush()

    px, py = _rect([10, 40, 40, 10], [10, 10, 40, 40])
    contour_schema = Contour(x=px, y=py, label_id=label.id, added_by="u", reviewed_by=["u"])
    save_contour_tree(session, contour_schema, mask.id)
    session.commit()

    # Must not raise; ctx.image is None -> metrics degrade to zeros (documented behavior).
    computed = compute_appearance_metrics_for_dataset(session, ds.id, only_stale=True)
    assert computed > 0
    rows = session.query(ContourMetrics).filter(
        ContourMetrics.metric_key == "mean_color_rgb"
    ).all()
    assert all(r.value == pytest.approx(0.0) for r in rows)

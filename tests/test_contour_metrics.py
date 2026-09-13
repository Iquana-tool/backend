"""End-to-end smoke test for the contour_metrics dual-write, the aggregation endpoint's
core function, and the backfill script.

Uses a temp-file SQLite database wired to fresh engine/session so it does not touch the
real database. Builds dataset -> non-square mm-scaled image -> mask -> a contour tree,
saves it via ``save_contour_tree``, and asserts:
  * the tall ``contour_metrics`` rows are PIXEL-native (stored scale-independent),
  * the legacy ``contours`` columns are in the image's PHYSICAL unit,
  * ``get_quantification_summary`` converts pixels -> physical on read (the dataset's one
    image is scaled to mm, so the summary is consistent and reports mm) with correct
    count/mean/std/min/max + child counts,
  * the backfill repairs both stores after a legacy column is zeroed.

The scale is isotropic (scale_x == scale_y), matching what the calibration UI produces:
area converts exactly (px_area * scale_x * scale_y) and length exactly (px_len * scale_x).
"""
import asyncio
import math

import numpy as np
import pytest
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

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.quantification import QuantificationModel

# Non-square image (so pixel-space projection matters) with an isotropic mm scale (what the
# calibration UI produces, so the read-time pixel->physical conversion is exact).
WIDTH, HEIGHT = 1000, 500
SCALE_X, SCALE_Y = 0.5, 0.5
UNIT = "mm"


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


def _rect(x_px, y_px):
    return ([x / WIDTH for x in x_px], [y / HEIGHT for y in y_px])


def _seed(session):
    """Create dataset, image, mask, two labels and a parent/child contour tree."""
    ds = Datasets(name="smoke", description="", dataset_type="image",
                  folder_path="/tmp/smoke", created_by="u")
    # Users FK: create the owner so created_by is valid.
    from app.database.users import Users
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    session.add(ds)
    session.flush()

    img = Images(dataset_id=ds.id, file_name="a.png", file_path="/tmp/a.png",
                 thumbnail_file_path="/tmp/t.png", width=WIDTH, height=HEIGHT,
                 color_mode="RGB", scale_x=SCALE_X, scale_y=SCALE_Y, unit=UNIT)
    session.add(img)
    session.flush()

    mask = Masks(image_id=img.id, fully_annotated=True, file_path="/tmp/m.png")
    session.add(mask)
    session.flush()

    parent_label = Labels(dataset_id=ds.id, parent_id=None, name="cell", value=1)
    child_label = Labels(dataset_id=ds.id, parent_id=None, name="nucleus", value=2)
    session.add_all([parent_label, child_label])
    session.flush()
    # Make child_label a child of parent_label in the hierarchy.
    child_label.parent_id = parent_label.id
    session.flush()

    return ds, img, mask, parent_label, child_label


def test_dual_write_and_aggregation_and_backfill(session):
    ds, img, mask, parent_label, child_label = _seed(session)

    # Two top-level rectangles (label=cell) each with one child rectangle (label=nucleus).
    px1, py1 = _rect([100, 200, 200, 100], [100, 100, 150, 150])  # 100x50 px
    px2, py2 = _rect([300, 500, 500, 300], [200, 200, 300, 300])  # 200x100 px
    cpx, cpy = _rect([120, 160, 160, 120], [110, 110, 130, 130])  # 40x20 px child

    tree1 = Contour(x=px1, y=py1, label_id=parent_label.id, added_by="u",
                    children=[Contour(x=cpx, y=cpy, label_id=child_label.id, added_by="u")])
    tree2 = Contour(x=px2, y=py2, label_id=parent_label.id, added_by="u")

    # Reviewer so exclude_unreviewed=True keeps them.
    from app.database.users import Users
    reviewer = session.query(Users).filter_by(username="u").one()
    tree1.reviewed_by = ["u"]
    tree2.reviewed_by = ["u"]
    tree1.children[0].reviewed_by = ["u"]

    save_contour_tree(session, tree1, mask.id)
    save_contour_tree(session, tree2, mask.id)
    session.commit()

    # --- Dual-write assertions ---------------------------------------------
    metric_rows = session.query(ContourMetrics).all()
    # 3 contours * 4 geometry metrics = 12 rows.
    assert len(metric_rows) == 12

    # The parent rectangle #1 is 100x50 px. The tall table stores PIXELS; the legacy columns
    # store PHYSICAL (scaled by SCALE_X/SCALE_Y).
    ref_px = QuantificationModel.from_contour(
        np.stack([np.array(px1) * WIDTH, np.array(py1) * HEIGHT], axis=-1),
        scale_x=1.0, scale_y=1.0, unit="px",
    )
    ref_phys = QuantificationModel.from_contour(
        np.stack([np.array(px1) * WIDTH, np.array(py1) * HEIGHT], axis=-1),
        scale_x=SCALE_X, scale_y=SCALE_Y, unit=UNIT,
    )
    # Find contour #1's id (top-level, label=cell, matches x length).
    parent1 = (
        session.query(Contours)
        .filter(Contours.label_id == parent_label.id, Contours.parent_id.is_(None))
        .order_by(Contours.id)
        .first()
    )
    area_row = session.query(ContourMetrics).filter_by(contour_id=parent1.id, metric_key="area").one()
    peri_row = session.query(ContourMetrics).filter_by(contour_id=parent1.id, metric_key="perimeter").one()
    # Tall table: pixel-native.
    assert area_row.value == pytest.approx(ref_px.area)
    assert area_row.value == pytest.approx(100.0 * 50.0)  # px^2
    assert area_row.unit == "px²"
    assert peri_row.value == pytest.approx(ref_px.perimeter)
    assert peri_row.value == pytest.approx(2 * (100.0 + 50.0))  # px
    assert peri_row.unit == "px"
    # Legacy columns: physical (image scale applied).
    assert parent1.area == pytest.approx(ref_phys.area)
    assert parent1.area == pytest.approx(50.0 * 25.0)  # mm^2

    # --- Aggregation function ----------------------------------------------
    # The dataset's single image is scaled to mm, so the summary is scale-consistent and
    # converts the stored pixels to physical mm on read.
    from app.services.database_access import datasets as datasets_db
    summary = asyncio.run(datasets_db.get_quantification_summary(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session
    ))
    assert summary["scale_status"]["consistent"] is True
    assert summary["scale_status"]["display_unit"] == "mm"
    metrics = summary["metrics"]
    cell_key = str(parent_label.id)
    assert cell_key in metrics
    area_stats = metrics[cell_key]["area"]["components"][0]
    assert area_stats["count"] == 2  # two cell contours
    assert metrics[cell_key]["area"]["unit"] == "mm²"

    # Verify mean/std/min/max against a manual numpy computation. The stored values are px;
    # the summary reports physical, so convert with area's factor (scale_x * scale_y).
    cell_ids = [
        c.id for c in session.query(Contours)
        .filter(Contours.label_id == parent_label.id, Contours.parent_id.is_(None)).all()
    ]
    areas = np.array([
        session.query(ContourMetrics).filter_by(contour_id=cid, metric_key="area").one().value
        for cid in cell_ids
    ]) * (SCALE_X * SCALE_Y)
    assert area_stats["mean"] == pytest.approx(float(areas.mean()))
    assert area_stats["std"] == pytest.approx(float(areas.std()))  # population std
    assert area_stats["min"] == pytest.approx(float(areas.min()))
    assert area_stats["max"] == pytest.approx(float(areas.max()))

    # Child counts: parent label 'cell' has 1 child of label 'nucleus'.
    ccpl = summary["child_counts_per_label_id"]
    assert ccpl.get(cell_key, {}).get(str(child_label.id)) == 1

    # --- Backfill repairs both stores --------------------------------------
    # Corrupt a legacy column and delete a metric row, then backfill.
    parent1.area = 0.0
    session.query(ContourMetrics).filter_by(contour_id=parent1.id, metric_key="area").delete()
    session.commit()

    # Run the backfill's core against this session by monkeypatching the session source.
    from app.database.contours import _resolve_metric_unit
    import scripts.backfill_contour_metrics as bf

    class _CtxSession:
        def __init__(self, s):
            self._s = s
        def __enter__(self):
            return self._s
        def __exit__(self, *a):
            return False

    original = bf.get_context_session
    bf.get_context_session = lambda: _CtxSession(session)
    try:
        result = bf.backfill(dataset_id=ds.id, dry_run=False)
    finally:
        bf.get_context_session = original

    assert result["processed"] == 3
    session.expire_all()
    repaired = session.query(Contours).filter_by(id=parent1.id).one()
    assert repaired.area == pytest.approx(ref_phys.area)  # legacy column: physical
    repaired_area_row = session.query(ContourMetrics).filter_by(contour_id=parent1.id, metric_key="area").one()
    assert repaired_area_row.value == pytest.approx(ref_px.area)  # tall table: pixels
    assert repaired_area_row.unit == "px²"

    from app.services.quantification import compute_geometry_metrics_for_dataset

    session.query(ContourMetrics).filter(
        ContourMetrics.metric_key.in_(["area", "perimeter", "circularity", "max_diameter"])
    ).update({"stale": True})
    session.commit()

    rows_written = compute_geometry_metrics_for_dataset(session, ds.id, only_stale=True)
    assert rows_written > 0

    stale_count = session.query(ContourMetrics).filter(
        ContourMetrics.metric_key.in_(["area", "perimeter", "circularity", "max_diameter"]),
        ContourMetrics.stale.is_(True),
    ).count()
    assert stale_count == 0


def test_mixed_scale_dataset_falls_back_to_pixels(session):
    """The reported bug: a dataset where one image is scaled (mm) and another is not (px)
    must NOT pool mm and px into one meaningless statistic. The summary reports pixels and
    flags the mix so the frontend can warn; when every image later shares one unit it
    reports that physical unit instead.

    Both images hold an identical 100x50 px rectangle (5000 px^2), so the pooled pixel mean
    is unambiguous regardless of the differing scales.
    """
    from app.database.users import Users
    from app.services.database_access import datasets as datasets_db

    ds, img_mm, mask_mm, parent_label, child_label = _seed(session)  # img_mm: 0.5 mm/px

    # A second image in the same dataset with NO scale (unit px, scale 1).
    img_px = Images(dataset_id=ds.id, file_name="b.png", file_path="/tmp/b.png",
                    thumbnail_file_path="/tmp/tb.png", width=WIDTH, height=HEIGHT,
                    color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    session.add(img_px)
    session.flush()
    mask_px = Masks(image_id=img_px.id, fully_annotated=True, file_path="/tmp/mb.png")
    session.add(mask_px)
    session.flush()

    # Identical 100x50 px rectangle on each image, same label, both reviewed.
    rect = _rect([100, 200, 200, 100], [100, 100, 150, 150])
    for mask in (mask_mm, mask_px):
        tree = Contour(x=rect[0], y=rect[1], label_id=parent_label.id, added_by="u",
                       reviewed_by=["u"])
        save_contour_tree(session, tree, mask.id)
    session.commit()

    summary = asyncio.run(datasets_db.get_quantification_summary(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session
    ))

    status = summary["scale_status"]
    assert status["consistent"] is False          # mm + px -> mixed
    assert status["display_unit"] == "px"          # so numbers stay pixel-native
    assert status["images_scaled"] == 1
    assert status["images_unscaled"] == 1

    area = summary["metrics"][str(parent_label.id)]["area"]
    assert area["unit"] == "px²"                    # NOT mm² / not clobbered
    comp = area["components"][0]
    assert comp["count"] == 2
    assert comp["mean"] == pytest.approx(100.0 * 50.0)   # both objects are 5000 px^2
    assert comp["min"] == pytest.approx(5000.0)
    assert comp["max"] == pytest.approx(5000.0)

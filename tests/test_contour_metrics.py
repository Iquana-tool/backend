"""End-to-end smoke test for Step 2: contour_metrics dual-write, the aggregation
endpoint's core function, and the backfill script.

Uses a temp-file SQLite database wired to fresh engine/session so it does not touch the
real database. Builds dataset -> non-square mm-scaled image -> mask -> a contour tree,
saves it via ``save_contour_tree``, and asserts:
  * the tall ``contour_metrics`` rows exist with the right values and units,
  * ``get_quantification_summary`` returns correct count/mean/std/min/max + child counts,
  * the backfill repairs both stores after a legacy column is zeroed.
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

# Non-square image with anisotropic mm scale, so units and pixel-space geometry matter.
WIDTH, HEIGHT = 1000, 500
SCALE_X, SCALE_Y = 0.5, 0.25
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

    # The parent rectangle #1 is 100x50 px -> physical 100*0.5 x 50*0.25 = 50mm x 12.5mm.
    ref = QuantificationModel.from_contour(
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
    assert area_row.value == pytest.approx(ref.area)
    assert area_row.value == pytest.approx(50.0 * 12.5)  # mm^2
    assert area_row.unit == "mm²"
    assert peri_row.value == pytest.approx(ref.perimeter)
    assert peri_row.unit == "mm"
    # Legacy columns dual-written identically.
    assert parent1.area == pytest.approx(area_row.value)

    # --- Aggregation function ----------------------------------------------
    from app.services.database_access import datasets as datasets_db
    summary = asyncio.run(datasets_db.get_quantification_summary(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session
    ))
    metrics = summary["metrics"]
    cell_key = str(parent_label.id)
    assert cell_key in metrics
    area_stats = metrics[cell_key]["area"]["components"][0]
    assert area_stats["count"] == 2  # two cell contours
    assert metrics[cell_key]["area"]["unit"] == "mm²"

    # Verify mean/std/min/max against a manual numpy computation.
    cell_ids = [
        c.id for c in session.query(Contours)
        .filter(Contours.label_id == parent_label.id, Contours.parent_id.is_(None)).all()
    ]
    areas = np.array([
        session.query(ContourMetrics).filter_by(contour_id=cid, metric_key="area").one().value
        for cid in cell_ids
    ])
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
    assert repaired.area == pytest.approx(ref.area)
    repaired_area_row = session.query(ContourMetrics).filter_by(contour_id=parent1.id, metric_key="area").one()
    assert repaired_area_row.value == pytest.approx(ref.area)
    assert repaired_area_row.unit == "mm²"

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

"""End-to-end smoke test for Step 4: contextual-tier metrics (nn_distance / mean_knn_distance).

Uses a temp-file SQLite database wired to a fresh engine/session (does not touch the real
database). Exercises the full lazy/batched + group-invalidation path:
  * ``compute_contextual_metrics_for_dataset`` computes and stores nn_distance /
    mean_knn_distance rows for every contour that has at least one same-parent sibling,
  * deleting one of three siblings recomputes the remaining pair (each other's nearest
    neighbor) and REMOVES the row for a contour that becomes an only-child,
  * moving a contour (modify_contour x/y) marks its whole parent group stale and a
    follow-up recompute changes the value,
  * ``get_quantification_summary`` surfaces nn_distance with a count that EXCLUDES
    only-child contours (they have no row at all).
"""
import asyncio

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
from app.database.users import Users

from app.services.quantification import (
    CONTEXTUAL_METRIC_KEYS,
    compute_contextual_metrics_for_dataset,
)
from app.services.database_access.contours import modify_contour, delete_contour

from iquana_toolbox.schemas.database.contours import Contour

# A non-square, mm-scaled image so physical-space distances differ from raw pixels.
WIDTH, HEIGHT = 1000, 1000
SCALE_X, SCALE_Y = 2.0, 2.0
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


def _point_rect(cx_px, cy_px, half=5):
    """A small square contour centered at (cx_px, cy_px), normalized to [0, 1]."""
    x_px = [cx_px - half, cx_px + half, cx_px + half, cx_px - half]
    y_px = [cy_px - half, cy_px - half, cy_px + half, cy_px + half]
    return ([x / WIDTH for x in x_px], [y / HEIGHT for y in y_px])


def _seed(session):
    ds = Datasets(name="contextual-smoke", description="", dataset_type="image",
                  folder_path="/tmp/contextual-smoke", created_by="u")
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

    label = Labels(dataset_id=ds.id, parent_id=None, name="blob", value=1)
    session.add(label)
    session.flush()

    return ds, img, mask, label


def _make_sibling_contours(session, mask, label, centers):
    """Save three top-level (parent_id=None) sibling contours at given pixel centers."""
    ids = []
    for cx, cy in centers:
        px, py = _point_rect(cx, cy)
        schema = Contour(x=px, y=py, label_id=label.id, added_by="u", reviewed_by=["u"])
        db_contour = save_contour_tree(session, schema, mask.id)
        ids.append(db_contour.id)
    session.commit()
    return ids


def test_three_siblings_compute_and_delete_removes_only_child_row(session):
    ds, img, mask, label = _seed(session)

    # Three siblings in a line: A(100,100) --100px-- B(200,100) --300px-- C(500,100).
    # In physical space (scale=2.0mm/px): A-B = 200mm, B-C = 600mm.
    ids = _make_sibling_contours(session, mask, label, [(100, 100), (200, 100), (500, 100)])
    id_a, id_b, id_c = ids

    # --- 1. Batch compute contextual metrics for the whole dataset ------------------
    computed = compute_contextual_metrics_for_dataset(session, ds.id, only_stale=True)
    assert computed > 0

    def _nn_distance(contour_id):
        row = session.query(ContourMetrics).filter_by(
            contour_id=contour_id, metric_key="nn_distance", component=0
        ).one_or_none()
        return row

    row_a, row_b, row_c = _nn_distance(id_a), _nn_distance(id_b), _nn_distance(id_c)
    assert row_a is not None and row_b is not None and row_c is not None
    assert row_a.value == pytest.approx(200.0)  # A -> B
    assert row_b.value == pytest.approx(200.0)  # B -> A (closer than B -> C = 600)
    assert row_c.value == pytest.approx(600.0)  # C -> B
    assert row_a.unit == "mm"
    assert not any(r.stale for r in (row_a, row_b, row_c))

    # mean_knn_distance rows also exist (k clamps to group_size - 1 = 2 here).
    knn_a = session.query(ContourMetrics).filter_by(
        contour_id=id_a, metric_key="mean_knn_distance", component=0
    ).one()
    assert knn_a.value == pytest.approx((200.0 + 800.0) / 2)  # A->B=200mm, A->C=800mm

    # --- 2. Re-running with only_stale=True is a no-op (already fresh) --------------
    recomputed = compute_contextual_metrics_for_dataset(session, ds.id, only_stale=True)
    assert recomputed == 0

    # --- 3. Delete B: A and C survive as an only-remaining pair ----------------------
    asyncio.run(delete_contour(id_b, db=session))

    session.expire_all()
    # B's own rows are gone via CASCADE.
    assert session.query(ContourMetrics).filter_by(contour_id=id_b).count() == 0
    # A and C's rows must be marked stale (deleting a sibling invalidates the survivors).
    stale_a = session.query(ContourMetrics).filter_by(
        contour_id=id_a, metric_key="nn_distance", component=0
    ).one()
    assert stale_a.stale is True

    recomputed_after_delete = compute_contextual_metrics_for_dataset(session, ds.id, only_stale=True)
    assert recomputed_after_delete > 0

    session.expire_all()
    fresh_a = session.query(ContourMetrics).filter_by(
        contour_id=id_a, metric_key="nn_distance", component=0
    ).one()
    fresh_c = session.query(ContourMetrics).filter_by(
        contour_id=id_c, metric_key="nn_distance", component=0
    ).one()
    # A and C are now each other's only neighbor: distance A(100,100) -> C(500,100) =
    # 400px * 2mm/px = 800mm.
    assert fresh_a.value == pytest.approx(800.0)
    assert fresh_c.value == pytest.approx(800.0)
    assert not fresh_a.stale and not fresh_c.stale


def test_deleting_down_to_an_only_child_removes_its_row(session):
    """Two siblings -> delete one -> the survivor becomes an only-child and its
    nn_distance row must be REMOVED entirely (not left stale, not zero)."""
    ds, img, mask, label = _seed(session)
    id_a, id_b = _make_sibling_contours(session, mask, label, [(100, 100), (300, 100)])

    compute_contextual_metrics_for_dataset(session, ds.id, only_stale=True)
    assert session.query(ContourMetrics).filter_by(contour_id=id_a, metric_key="nn_distance").count() == 1
    assert session.query(ContourMetrics).filter_by(contour_id=id_b, metric_key="nn_distance").count() == 1

    asyncio.run(delete_contour(id_b, db=session))
    compute_contextual_metrics_for_dataset(session, ds.id, only_stale=True)

    session.expire_all()
    # A is now an only-child: nn_distance / mean_knn_distance have no meaningful value
    # and must be OMITTED, so no row should exist at all (not a stale row, not a 0).
    assert session.query(ContourMetrics).filter_by(
        contour_id=id_a, metric_key="nn_distance"
    ).count() == 0
    assert session.query(ContourMetrics).filter_by(
        contour_id=id_a, metric_key="mean_knn_distance"
    ).count() == 0


def test_moving_a_contour_marks_group_stale_and_recompute_changes_value(session):
    ds, img, mask, label = _seed(session)
    id_a, id_b = _make_sibling_contours(session, mask, label, [(100, 100), (300, 100)])

    compute_contextual_metrics_for_dataset(session, ds.id, only_stale=True)
    session.expire_all()
    before = session.query(ContourMetrics).filter_by(contour_id=id_a, metric_key="nn_distance").one()
    assert before.value == pytest.approx(400.0)  # 200px * 2mm/px

    # Move A far away from B: (100,100) -> (900, 900).
    new_px, new_py = _point_rect(900, 900)
    asyncio.run(modify_contour(id_a, db=session, x=new_px, y=new_py))

    session.expire_all()
    stale_a = session.query(ContourMetrics).filter_by(contour_id=id_a, metric_key="nn_distance").one()
    stale_b = session.query(ContourMetrics).filter_by(contour_id=id_b, metric_key="nn_distance").one()
    assert stale_a.stale and stale_b.stale  # whole group marked stale, not just the mover

    recomputed = compute_contextual_metrics_for_dataset(session, ds.id, only_stale=True)
    assert recomputed > 0

    session.expire_all()
    after_a = session.query(ContourMetrics).filter_by(contour_id=id_a, metric_key="nn_distance").one()
    after_b = session.query(ContourMetrics).filter_by(contour_id=id_b, metric_key="nn_distance").one()
    # New distance: (900,900) -> (300,100) = sqrt(600^2 + 800^2) = 1000px * 2mm/px = 2000mm.
    assert after_a.value == pytest.approx(2000.0)
    assert after_b.value == pytest.approx(2000.0)
    assert not after_a.stale and not after_b.stale


def test_new_contour_joining_existing_group_marks_it_stale(session):
    """A brand new sibling must invalidate the group's EXISTING members too (they gained
    a new potential neighbor), not just get its own row computed."""
    ds, img, mask, label = _seed(session)
    id_a, id_b = _make_sibling_contours(session, mask, label, [(100, 100), (300, 100)])
    compute_contextual_metrics_for_dataset(session, ds.id, only_stale=True)
    session.expire_all()

    # Adding a third sibling much closer to A should invalidate A's existing row.
    px, py = _point_rect(110, 100)
    new_schema = Contour(x=px, y=py, label_id=label.id, added_by="u", reviewed_by=["u"])
    new_contour = save_contour_tree(session, new_schema, mask.id)
    session.commit()

    session.expire_all()
    stale_a = session.query(ContourMetrics).filter_by(contour_id=id_a, metric_key="nn_distance").one()
    assert stale_a.stale is True

    recomputed = compute_contextual_metrics_for_dataset(session, ds.id, only_stale=True)
    assert recomputed > 0
    session.expire_all()
    after_a = session.query(ContourMetrics).filter_by(contour_id=id_a, metric_key="nn_distance").one()
    # A(100,100) now has a much closer neighbor at (110,100): 10px * 2mm/px = 20mm.
    assert after_a.value == pytest.approx(20.0)
    assert not after_a.stale


def test_quantification_summary_surfaces_nn_distance_excluding_only_children(session):
    ds, img, mask, label = _seed(session)
    # Separate label for the (root-level) parent containers, so their own nn_distance
    # rows (they ARE siblings of each other at root level, per the documented root-level
    # behavior) land under a different label_id and don't affect ``label``'s count below.
    container_label = Labels(dataset_id=ds.id, parent_id=None, name="container", value=2)
    session.add(container_label)
    session.flush()

    # A container contour, large enough to hold A/B.
    parent_px, parent_py = _point_rect(500, 500, half=490)
    parent_schema = Contour(x=parent_px, y=parent_py, label_id=container_label.id, added_by="u",
                            reviewed_by=["u"])
    parent_contour = save_contour_tree(session, parent_schema, mask.id)
    session.commit()

    # Two siblings UNDER that container (get a meaningful nn_distance, label = `label`).
    px_a, py_a = _point_rect(100, 100)
    px_b, py_b = _point_rect(300, 100)
    schema_a = Contour(x=px_a, y=py_a, label_id=label.id, added_by="u", reviewed_by=["u"])
    schema_b = Contour(x=px_b, y=py_b, label_id=label.id, added_by="u", reviewed_by=["u"])
    contour_a = save_contour_tree(session, schema_a, mask.id, parent_id=parent_contour.id)
    contour_b = save_contour_tree(session, schema_b, mask.id, parent_id=parent_contour.id)
    session.commit()
    id_a, id_b = contour_a.id, contour_b.id

    # A second, SEPARATE container with a single (only-)child of label `label`.
    other_parent_px, other_parent_py = _point_rect(900, 900, half=90)
    other_parent_schema = Contour(x=other_parent_px, y=other_parent_py, label_id=container_label.id,
                                  added_by="u", reviewed_by=["u"])
    other_parent_contour = save_contour_tree(session, other_parent_schema, mask.id)
    session.commit()

    lone_px, lone_py = _point_rect(900, 900)
    lone_schema = Contour(x=lone_px, y=lone_py, label_id=label.id, added_by="u", reviewed_by=["u"])
    lone_contour = save_contour_tree(session, lone_schema, mask.id, parent_id=other_parent_contour.id)
    session.commit()

    compute_contextual_metrics_for_dataset(session, ds.id, only_stale=True)

    # Sanity: the lone contour has no nn_distance row at all (only-child, omitted).
    assert session.query(ContourMetrics).filter_by(
        contour_id=lone_contour.id, metric_key="nn_distance"
    ).count() == 0

    from app.services.database_access import datasets as datasets_db
    summary = asyncio.run(datasets_db.get_quantification_summary(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session
    ))
    label_key = str(label.id)
    assert label_key in summary["metrics"]
    nn_entry = summary["metrics"][label_key]["nn_distance"]
    # Only A and B (label=`label`) have a meaningful nn_distance; the lone child (also
    # label=`label`, but an only-child under its own container) is OMITTED entirely ->
    # count == 2, not 3.
    assert nn_entry["components"][0]["count"] == 2
    assert nn_entry["unit"] == "mm"
    assert nn_entry["components"][0]["mean"] == pytest.approx(400.0)

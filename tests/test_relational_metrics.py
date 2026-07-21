"""End-to-end tests for the RELATIONAL-tier metric ``n_children`` and its parent-targeted
staleness wiring, plus profile-scoped aggregation.

Uses a temp-file SQLite database (does not touch the real database). Exercises:
  * ``compute_relational_metrics_for_dataset`` writes a count row for EVERY contour
    (including 0 for leaves), a parent with 3 children -> 3,
  * deleting a child marks the parent stale -> recompute -> 2,
  * re-parenting a child marks BOTH old and new parent stale -> recompute -> correct counts,
  * ``get_quantification_summary`` with a profile scoping ``n_children`` to the parent label
    only reports it just for that label.
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
from app.database.contour_metrics import ContourMetrics
from app.database.contours import save_contour_tree
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users

from app.services.quantification import compute_relational_metrics_for_dataset
from app.services.database_access.contours import modify_contour, delete_contour

from iquana_toolbox.schemas.database.contours import Contour

WIDTH, HEIGHT = 1000, 1000
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


def _rect(cx_px, cy_px, half=5):
    x_px = [cx_px - half, cx_px + half, cx_px + half, cx_px - half]
    y_px = [cy_px - half, cy_px - half, cy_px + half, cy_px + half]
    return ([x / WIDTH for x in x_px], [y / HEIGHT for y in y_px])


def _seed(session):
    ds = Datasets(name="relational-smoke", description="", dataset_type="image",
                  folder_path="/tmp/relational-smoke", created_by="u")
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
    session.add(parent_label)
    session.flush()
    child_label = Labels(dataset_id=ds.id, parent_id=parent_label.id, name="nucleus", value=2)
    session.add(child_label)
    session.flush()

    return ds, img, mask, parent_label, child_label


def _n_children(session, contour_id):
    row = session.query(ContourMetrics).filter_by(
        contour_id=contour_id, metric_key="n_children", component=0
    ).one_or_none()
    return row


def test_parent_with_children_and_leaf_counts(session):
    ds, img, mask, parent_label, child_label = _seed(session)

    px, py = _rect(500, 500, half=200)
    parent_schema = Contour(x=px, y=py, label_id=parent_label.id, added_by="u", reviewed_by=["u"])
    parent = save_contour_tree(session, parent_schema, mask.id)
    session.commit()

    child_ids = []
    for i, cx in enumerate((400, 500, 600)):
        cpx, cpy = _rect(cx, 500)
        cs = Contour(x=cpx, y=cpy, label_id=child_label.id, added_by="u", reviewed_by=["u"])
        c = save_contour_tree(session, cs, mask.id, parent_id=parent.id)
        child_ids.append(c.id)
    session.commit()

    computed = compute_relational_metrics_for_dataset(session, ds.id, only_stale=True)
    assert computed > 0

    # Parent has 3 children.
    assert _n_children(session, parent.id).value == pytest.approx(3.0)
    assert _n_children(session, parent.id).unit == ""  # COUNT resolves to empty unit
    # Leaves each have 0 (a real, stored value - not omitted like contextual metrics).
    for cid in child_ids:
        row = _n_children(session, cid)
        assert row is not None
        assert row.value == pytest.approx(0.0)

    # Re-running only_stale is a no-op.
    assert compute_relational_metrics_for_dataset(session, ds.id, only_stale=True) == 0


def test_delete_child_marks_parent_stale_and_recompute_decrements(session):
    ds, img, mask, parent_label, child_label = _seed(session)
    px, py = _rect(500, 500, half=200)
    parent = save_contour_tree(
        session, Contour(x=px, y=py, label_id=parent_label.id, added_by="u", reviewed_by=["u"]), mask.id
    )
    session.commit()
    child_ids = []
    for cx in (400, 500, 600):
        cpx, cpy = _rect(cx, 500)
        c = save_contour_tree(
            session, Contour(x=cpx, y=cpy, label_id=child_label.id, added_by="u", reviewed_by=["u"]),
            mask.id, parent_id=parent.id,
        )
        child_ids.append(c.id)
    session.commit()
    compute_relational_metrics_for_dataset(session, ds.id, only_stale=True)
    assert _n_children(session, parent.id).value == pytest.approx(3.0)

    # Delete one child: the parent lost a child -> its n_children row is stale.
    asyncio.run(delete_contour(child_ids[0], db=session))
    session.expire_all()
    assert _n_children(session, parent.id).stale is True

    recomputed = compute_relational_metrics_for_dataset(session, ds.id, only_stale=True)
    assert recomputed > 0
    session.expire_all()
    assert _n_children(session, parent.id).value == pytest.approx(2.0)
    assert not _n_children(session, parent.id).stale


def test_reparent_child_marks_both_parents_stale(session):
    ds, img, mask, parent_label, child_label = _seed(session)
    # Two parents A and B; A starts with one child.
    pax, pay = _rect(300, 300, half=100)
    pbx, pby = _rect(700, 700, half=100)
    parent_a = save_contour_tree(
        session, Contour(x=pax, y=pay, label_id=parent_label.id, added_by="u", reviewed_by=["u"]), mask.id
    )
    parent_b = save_contour_tree(
        session, Contour(x=pbx, y=pby, label_id=parent_label.id, added_by="u", reviewed_by=["u"]), mask.id
    )
    session.commit()
    cpx, cpy = _rect(300, 300)
    child = save_contour_tree(
        session, Contour(x=cpx, y=cpy, label_id=child_label.id, added_by="u", reviewed_by=["u"]),
        mask.id, parent_id=parent_a.id,
    )
    session.commit()
    compute_relational_metrics_for_dataset(session, ds.id, only_stale=True)
    assert _n_children(session, parent_a.id).value == pytest.approx(1.0)
    assert _n_children(session, parent_b.id).value == pytest.approx(0.0)

    # Re-parent the child from A to B: BOTH parents must be marked stale.
    asyncio.run(modify_contour(child.id, db=session, parent_id=parent_b.id))
    session.expire_all()
    assert _n_children(session, parent_a.id).stale is True
    assert _n_children(session, parent_b.id).stale is True

    compute_relational_metrics_for_dataset(session, ds.id, only_stale=True)
    session.expire_all()
    assert _n_children(session, parent_a.id).value == pytest.approx(0.0)
    assert _n_children(session, parent_b.id).value == pytest.approx(1.0)


def test_summary_with_profile_scoping_n_children_to_parent_label(session):
    ds, img, mask, parent_label, child_label = _seed(session)
    px, py = _rect(500, 500, half=200)
    parent = save_contour_tree(
        session, Contour(x=px, y=py, label_id=parent_label.id, added_by="u", reviewed_by=["u"]), mask.id
    )
    session.commit()
    for cx in (400, 600):
        cpx, cpy = _rect(cx, 500)
        save_contour_tree(
            session, Contour(x=cpx, y=cpy, label_id=child_label.id, added_by="u", reviewed_by=["u"]),
            mask.id, parent_id=parent.id,
        )
    session.commit()
    compute_relational_metrics_for_dataset(session, ds.id, only_stale=True)

    from app.services.database_access import datasets as datasets_db

    # Profile scopes n_children to the PARENT label only.
    scoping = {"n_children": [parent_label.id]}
    summary = asyncio.run(datasets_db.get_quantification_summary(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session,
        metric_scoping=scoping,
    ))
    metrics = summary["metrics"]
    # Parent label reports n_children == 2 (one parent contour, 2 children).
    assert str(parent_label.id) in metrics
    parent_entry = metrics[str(parent_label.id)]["n_children"]
    assert parent_entry["components"][0]["count"] == 1  # one parent contour measured
    assert parent_entry["components"][0]["mean"] == pytest.approx(2.0)
    # Child label is NOT reported for n_children (scoped out), even though children have
    # stored 0-rows.
    child_metrics = metrics.get(str(child_label.id), {})
    assert "n_children" not in child_metrics

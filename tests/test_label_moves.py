"""Tests for moving a label to a different parent.

Uses a temp-file SQLite database (does not touch the real database). Exercises:
  * a move that strands no annotations succeeds outright,
  * a move that would strand annotated objects is refused and writes nothing,
  * the same move with ``detach_affected`` demotes those objects to root level, keeps
    their label, and marks the parent they left stale so ``n_children`` recomputes,
  * promoting a label to the top level strands its nested objects the same way,
  * cycles and self-parenting are rejected,
  * objects labelled with the moved label's CHILDREN are untouched.
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
from app.database.contours import Contours, save_contour_tree
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users

from app.services.database_access.label_moves import (
    LabelMoveBlocked,
    LabelMoveError,
    move_label,
    plan_move,
)
from app.services.quantification import compute_relational_metrics_for_dataset

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
    """A dataset whose label space is ``cell > nucleus`` plus a second root ``tissue``."""
    ds = Datasets(name="label-move", description="", dataset_type="image",
                  folder_path="/tmp/label-move", created_by="u")
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    session.add(ds)
    session.flush()

    img = Images(dataset_id=ds.id, file_name="a.png", file_path="/tmp/a.png",
                 thumbnail_file_path="/tmp/t.png", width=WIDTH, height=HEIGHT,
                 color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    session.add(img)
    session.flush()

    mask = Masks(image_id=img.id, fully_annotated=True, file_path="/tmp/m.png")
    session.add(mask)
    session.flush()

    cell = Labels(dataset_id=ds.id, parent_id=None, name="cell", value=1)
    tissue = Labels(dataset_id=ds.id, parent_id=None, name="tissue", value=2)
    session.add_all([cell, tissue])
    session.flush()
    nucleus = Labels(dataset_id=ds.id, parent_id=cell.id, name="nucleus", value=3)
    session.add(nucleus)
    session.flush()

    return ds, mask, cell, tissue, nucleus


def _annotate_nucleus_in_cell(session, mask, cell, nucleus):
    """One ``cell`` object with one ``nucleus`` object inside it."""
    px, py = _rect(500, 500, half=200)
    parent = save_contour_tree(
        session, Contour(x=px, y=py, label_id=cell.id, added_by="u", reviewed_by=["u"]), mask.id
    )
    session.commit()
    cpx, cpy = _rect(500, 500)
    child = save_contour_tree(
        session, Contour(x=cpx, y=cpy, label_id=nucleus.id, added_by="u", reviewed_by=["u"]),
        mask.id, parent_id=parent.id,
    )
    session.commit()
    return parent, child


def _n_children(session, contour_id):
    return session.query(ContourMetrics).filter_by(
        contour_id=contour_id, metric_key="n_children", component=0
    ).one_or_none()


def test_move_without_annotations_succeeds(session):
    ds, mask, cell, tissue, nucleus = _seed(session)

    impact = asyncio.run(move_label(session, cell.id, tissue.id))

    assert impact.count == 0
    session.refresh(cell)
    assert cell.parent_id == tissue.id


def test_move_is_blocked_when_it_would_strand_annotations(session):
    ds, mask, cell, tissue, nucleus = _seed(session)
    parent, child = _annotate_nucleus_in_cell(session, mask, cell, nucleus)

    with pytest.raises(LabelMoveBlocked) as excinfo:
        asyncio.run(move_label(session, nucleus.id, tissue.id))

    impact = excinfo.value.impact
    assert impact.count == 1
    assert impact.affected[0].contour_id == child.id
    assert impact.affected[0].old_parent_id == parent.id

    # Nothing was written: neither the label nor the annotation moved.
    session.refresh(nucleus)
    session.refresh(child)
    assert nucleus.parent_id == cell.id
    assert child.parent_id == parent.id


def test_detach_affected_moves_the_label_and_demotes_the_objects(session):
    ds, mask, cell, tissue, nucleus = _seed(session)
    parent, child = _annotate_nucleus_in_cell(session, mask, cell, nucleus)

    # Baseline: the cell object has one child.
    compute_relational_metrics_for_dataset(session, ds.id, only_stale=True)
    assert _n_children(session, parent.id).value == pytest.approx(1.0)

    impact = asyncio.run(move_label(session, nucleus.id, tissue.id, detach_affected=True))

    assert impact.count == 1
    session.refresh(nucleus)
    session.refresh(child)
    assert nucleus.parent_id == tissue.id
    # The class the annotator asserted survives; only the containment link is dropped.
    assert child.parent_id is None
    assert child.label_id == nucleus.id

    # The parent it left was marked stale, so recomputing corrects its child count.
    compute_relational_metrics_for_dataset(session, ds.id, only_stale=True)
    assert _n_children(session, parent.id).value == pytest.approx(0.0)


def test_promoting_to_top_level_strands_nested_objects(session):
    ds, mask, cell, tissue, nucleus = _seed(session)
    parent, child = _annotate_nucleus_in_cell(session, mask, cell, nucleus)

    # A top-level label is a direct part of nothing, so a nested object cannot carry it.
    impact = plan_move(session, nucleus.id, None)
    assert impact.count == 1

    with pytest.raises(LabelMoveBlocked):
        asyncio.run(move_label(session, nucleus.id, None))

    asyncio.run(move_label(session, nucleus.id, None, detach_affected=True))
    session.refresh(nucleus)
    session.refresh(child)
    assert nucleus.parent_id is None
    assert child.parent_id is None


def test_objects_labelled_with_a_child_of_the_moved_label_are_untouched(session):
    ds, mask, cell, tissue, nucleus = _seed(session)
    parent, child = _annotate_nucleus_in_cell(session, mask, cell, nucleus)

    # Moving `cell` does not affect the nucleus object: its validity depends on its
    # container still being labelled `cell`, which the move does not change.
    impact = asyncio.run(move_label(session, cell.id, tissue.id, detach_affected=True))

    assert impact.count == 0
    session.refresh(child)
    assert child.parent_id == parent.id


def test_cycles_and_self_parenting_are_rejected(session):
    ds, mask, cell, tissue, nucleus = _seed(session)

    with pytest.raises(LabelMoveError):
        asyncio.run(move_label(session, cell.id, nucleus.id))

    with pytest.raises(LabelMoveError):
        asyncio.run(move_label(session, cell.id, cell.id))

    with pytest.raises(LabelMoveError):
        asyncio.run(move_label(session, cell.id, 9999))

    session.refresh(cell)
    assert cell.parent_id is None


def test_no_op_move_is_accepted_and_changes_nothing(session):
    ds, mask, cell, tissue, nucleus = _seed(session)
    parent, child = _annotate_nucleus_in_cell(session, mask, cell, nucleus)

    # Re-stating the current parent must not be reported as stranding its own objects.
    impact = asyncio.run(move_label(session, nucleus.id, cell.id))

    assert impact.count == 0
    session.refresh(child)
    assert child.parent_id == parent.id

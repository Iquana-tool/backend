"""Tests for the annotator's undo / redo history.

The feature's whole promise is that undoing a delete gives back *the object you
deleted* -- not a lookalike. So most of what is checked here is identity and
completeness of the restore: the same contour id, the nested children, the label,
the author, the approvals. A client-side undo stack could satisfy none of that,
which is why the history lives in the database at all.

The rest covers the stack semantics people expect from Ctrl+Z: a new edit kills
the redo branch, one annotator's undo cannot touch another's work, a fan-out
operation costs one step rather than thirty, and the log stays bounded.
"""
import asyncio

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.annotation_actions  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.users  # noqa: F401
import app.database.contours  # noqa: F401  (also pulls in contour_metrics)
from app.database.annotation_actions import MAX_HISTORY_ENTRIES, AnnotationActions
from app.database.contours import Contours, save_contour_tree
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.services import hierarchy_cache
from app.services.database_access import annotation_history as history_db
from app.services.database_access import contours as contours_db
from app.services.database_access import masks as masks_db

from iquana_toolbox.schemas.database.contours import Contour

WIDTH, HEIGHT = 1000, 1000
ANNOTATOR = "ann"
OTHER = "other"


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
    hierarchy_cache.clear()
    try:
        yield s
    finally:
        hierarchy_cache.clear()
        s.close()
        engine.dispose()


def _contour(cx_px, cy_px, half=20, label_id=None, parent_id=None):
    x_px = [cx_px - half, cx_px + half, cx_px + half, cx_px - half]
    y_px = [cy_px - half, cy_px - half, cy_px + half, cy_px + half]
    return Contour(
        x=[x / WIDTH for x in x_px],
        y=[y / HEIGHT for y in y_px],
        added_by="User",
        confidence=1.0,
        label_id=label_id,
        parent_id=parent_id,
    )


@pytest.fixture
def world(session):
    """One dataset / image / mask, two labels, and two annotators."""
    session.add_all([
        Users(username=ANNOTATOR, hashed_password="x", is_admin=False),
        Users(username=OTHER, hashed_password="x", is_admin=False),
    ])
    dataset = Datasets(name="hist", description="", dataset_type="image",
                       folder_path="/tmp/hist", created_by=ANNOTATOR)
    session.add(dataset)
    session.flush()

    coral = Labels(dataset_id=dataset.id, name="coral", value=1)
    algae = Labels(dataset_id=dataset.id, name="algae", value=2)
    session.add_all([coral, algae])

    image = Images(dataset_id=dataset.id, file_name="a.png", file_path="/tmp/a.png",
                   thumbnail_file_path="/tmp/t.png", width=WIDTH, height=HEIGHT,
                   color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    session.add(image)
    session.flush()

    mask = Masks(image_id=image.id, fully_annotated=False, file_path="/tmp/m.png")
    session.add(mask)
    session.commit()
    return {"mask": mask, "image": image, "coral": coral, "algae": algae}


def _add(session, mask, cx=100, cy=100, half=20, label_id=None, parent_id=None,
         author=ANNOTATOR):
    """Save a contour and return its row."""
    row = save_contour_tree(session, _contour(cx, cy, half, label_id, parent_id),
                            mask.id, parent_id=parent_id, author_username=author)
    session.commit()
    return row


def _delete_with_history(session, mask, contour_id, username=ANNOTATOR):
    """The delete path as the handlers run it: snapshot, delete, record."""
    snapshot = history_db.snapshot_subtree(contour_id, session)
    asyncio.run(contours_db.delete_contour(contour_id, session))
    history_db.record_delete(session, mask.id, username, snapshot)


def _ids(session, mask):
    return {row.id for row in session.query(Contours).filter_by(mask_id=mask.id).all()}


# --------------------------------------------------------------------------- #
# Undoing a delete -- the case in the issue
# --------------------------------------------------------------------------- #
def test_undoing_a_delete_restores_the_same_contour(session, world):
    mask = world["mask"]
    contour = _add(session, mask, label_id=world["coral"].id)
    contour_id = contour.id

    _delete_with_history(session, mask, contour_id)
    assert _ids(session, mask) == set()

    history_db.undo(mask.id, ANNOTATOR, session)

    restored = session.query(Contours).filter_by(id=contour_id).one()
    # The id is the point: everything else in the system references the object by
    # it, so a restore under a fresh id would be a different object wearing its face.
    assert restored.id == contour_id
    assert restored.label_id == world["coral"].id
    assert restored.author_username == ANNOTATOR
    assert restored.mask_id == mask.id


def test_undoing_a_delete_restores_nested_children(session, world):
    """The delete cascades to descendants, so the undo has to bring them all back."""
    mask = world["mask"]
    # Plain ints, not ORM rows: the cascade delete below detaches the instances.
    parent_id = _add(session, mask, half=100).id
    child_id = _add(session, mask, half=10, parent_id=parent_id).id
    grandchild_id = _add(session, mask, half=4, parent_id=child_id).id

    _delete_with_history(session, mask, parent_id)
    assert _ids(session, mask) == set()

    history_db.undo(mask.id, ANNOTATOR, session)

    assert _ids(session, mask) == {parent_id, child_id, grandchild_id}
    assert session.query(Contours).filter_by(id=child_id).one().parent_id == parent_id
    assert session.query(Contours).filter_by(id=grandchild_id).one().parent_id == child_id


def test_undoing_a_delete_restores_approvals(session, world):
    mask = world["mask"]
    contour = _add(session, mask)
    reviewer = session.query(Users).filter_by(username=OTHER).one()
    contour.reviewed_by = [reviewer]
    session.commit()
    contour_id = contour.id

    _delete_with_history(session, mask, contour_id)
    history_db.undo(mask.id, ANNOTATOR, session)

    restored = session.query(Contours).filter_by(id=contour_id).one()
    assert [user.username for user in restored.reviewed_by] == [OTHER]


def test_redo_deletes_the_restored_contour_again(session, world):
    mask = world["mask"]
    contour_id = _add(session, mask).id

    _delete_with_history(session, mask, contour_id)
    history_db.undo(mask.id, ANNOTATOR, session)
    assert _ids(session, mask) == {contour_id}

    history_db.redo(mask.id, ANNOTATOR, session)
    assert _ids(session, mask) == set()


def test_restore_survives_a_parent_that_is_gone(session, world):
    """A vanished parent must degrade to a top-level restore, not a foreign-key error."""
    mask = world["mask"]
    parent = _add(session, mask, half=100)
    child = _add(session, mask, half=10, parent_id=parent.id)
    child_id = child.id

    _delete_with_history(session, mask, child_id)
    # The parent goes away between the delete and the undo, without history.
    asyncio.run(contours_db.delete_contour(parent.id, session))

    history_db.undo(mask.id, ANNOTATOR, session)

    restored = session.query(Contours).filter_by(id=child_id).one()
    assert restored.parent_id is None


# --------------------------------------------------------------------------- #
# Undoing a create
# --------------------------------------------------------------------------- #
def test_undoing_a_create_removes_the_object_and_redo_brings_it_back(session, world):
    mask = world["mask"]
    contour_id = _add(session, mask, label_id=world["algae"].id).id
    history_db.record_create(session, mask.id, ANNOTATOR, contour_id)

    history_db.undo(mask.id, ANNOTATOR, session)
    assert _ids(session, mask) == set()

    history_db.redo(mask.id, ANNOTATOR, session)
    restored = session.query(Contours).filter_by(id=contour_id).one()
    assert restored.label_id == world["algae"].id


def test_undoing_a_create_refuses_when_work_was_nested_inside(session, world):
    """Undo must not cascade away objects the user added inside it afterwards."""
    mask = world["mask"]
    parent_id = _add(session, mask, half=100).id
    history_db.record_create(session, mask.id, ANNOTATOR, parent_id)
    child_id = _add(session, mask, half=10, parent_id=parent_id).id

    with pytest.raises(history_db.HistoryError):
        history_db.undo(mask.id, ANNOTATOR, session)

    assert _ids(session, mask) == {parent_id, child_id}


# --------------------------------------------------------------------------- #
# Label changes
# --------------------------------------------------------------------------- #
def test_label_change_round_trips(session, world):
    mask = world["mask"]
    contour = _add(session, mask, label_id=world["coral"].id)
    contour.label_id = world["algae"].id
    session.commit()
    history_db.record_label_change(session, mask.id, ANNOTATOR, contour.id,
                                   world["coral"].id, world["algae"].id)

    history_db.undo(mask.id, ANNOTATOR, session)
    assert session.query(Contours).filter_by(id=contour.id).one().label_id == world["coral"].id

    history_db.redo(mask.id, ANNOTATOR, session)
    assert session.query(Contours).filter_by(id=contour.id).one().label_id == world["algae"].id


def test_a_label_change_that_changes_nothing_is_not_recorded(session, world):
    mask = world["mask"]
    contour = _add(session, mask, label_id=world["coral"].id)

    history_db.record_label_change(session, mask.id, ANNOTATOR, contour.id,
                                   world["coral"].id, world["coral"].id)

    assert history_db.get_status(mask.id, ANNOTATOR, session)["can_undo"] is False


# --------------------------------------------------------------------------- #
# Stack semantics
# --------------------------------------------------------------------------- #
def test_a_new_action_discards_the_redo_branch(session, world):
    mask = world["mask"]
    first_id = _add(session, mask).id
    _delete_with_history(session, mask, first_id)
    history_db.undo(mask.id, ANNOTATOR, session)
    assert history_db.get_status(mask.id, ANNOTATOR, session)["can_redo"] is True

    second_id = _add(session, mask, cx=500, cy=500).id
    history_db.record_create(session, mask.id, ANNOTATOR, second_id)

    status = history_db.get_status(mask.id, ANNOTATOR, session)
    assert status["can_redo"] is False
    assert status["can_undo"] is True


def test_one_annotators_stack_is_invisible_to_another(session, world):
    mask = world["mask"]
    contour_id = _add(session, mask).id
    _delete_with_history(session, mask, contour_id, username=ANNOTATOR)

    assert history_db.get_status(mask.id, OTHER, session)["can_undo"] is False
    with pytest.raises(history_db.HistoryError):
        history_db.undo(mask.id, OTHER, session)
    # And the other annotator's own undo is unaffected.
    assert history_db.get_status(mask.id, ANNOTATOR, session)["can_undo"] is True


def test_a_grouped_run_is_one_undo_step(session, world):
    """Thirty suggested instances should cost one Ctrl+Z, not thirty."""
    mask = world["mask"]
    group = history_db.new_group_id()
    ids = set()
    for index in range(5):
        contour_id = _add(session, mask, cx=100 + index * 100, cy=100, half=20).id
        history_db.record_create(session, mask.id, ANNOTATOR, contour_id, group_id=group)
        ids.add(contour_id)

    history_db.undo(mask.id, ANNOTATOR, session)
    assert _ids(session, mask) == set()

    history_db.redo(mask.id, ANNOTATOR, session)
    assert _ids(session, mask) == ids


def test_history_is_capped(session, world):
    mask = world["mask"]
    for index in range(MAX_HISTORY_ENTRIES + 4):
        contour_id = _add(session, mask, cx=50 + index * 30, cy=50, half=10).id
        history_db.record_create(session, mask.id, ANNOTATOR, contour_id)

    kept = session.query(AnnotationActions).filter_by(mask_id=mask.id,
                                                      username=ANNOTATOR).count()
    assert kept == MAX_HISTORY_ENTRIES


def test_status_describes_the_next_step(session, world):
    mask = world["mask"]
    contour_id = _add(session, mask).id
    _delete_with_history(session, mask, contour_id)

    status = history_db.get_status(mask.id, ANNOTATOR, session)
    assert status["undo_label"] == "delete object"
    assert status["redo_label"] is None


def test_undo_on_an_empty_stack_is_an_error(session, world):
    with pytest.raises(history_db.HistoryError):
        history_db.undo(world["mask"].id, ANNOTATOR, session)


# --------------------------------------------------------------------------- #
# Operations the history cannot span
# --------------------------------------------------------------------------- #
def test_wiping_a_mask_discards_the_history(session, world):
    """Undo must not reach across a wholesale replacement of the mask.

    Without this, deleting one object and then clearing the whole image would
    leave an undo that puts that single object back into the emptied mask --
    resurrecting one member of a set the user deliberately cleared.
    """
    mask = world["mask"]
    doomed_id = _add(session, mask).id
    _add(session, mask, cx=500, cy=500)
    _delete_with_history(session, mask, doomed_id)
    assert history_db.get_status(mask.id, ANNOTATOR, session)["can_undo"] is True

    asyncio.run(masks_db.delete_all_contours_of_mask(mask.id, db=session))

    assert history_db.get_status(mask.id, ANNOTATOR, session)["can_undo"] is False
    assert _ids(session, mask) == set()
    with pytest.raises(history_db.HistoryError):
        history_db.undo(mask.id, ANNOTATOR, session)


def test_wiping_a_mask_discards_every_annotators_history(session, world):
    """The wipe is mask-wide, so it must not leave one user's stack behind."""
    mask = world["mask"]
    mine = _add(session, mask).id
    theirs = _add(session, mask, cx=500, cy=500).id
    _delete_with_history(session, mask, mine, username=ANNOTATOR)
    _delete_with_history(session, mask, theirs, username=OTHER)

    asyncio.run(masks_db.delete_all_contours_of_mask(mask.id, db=session))

    assert session.query(AnnotationActions).filter_by(mask_id=mask.id).count() == 0

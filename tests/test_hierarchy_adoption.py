"""Regression and unit tests for taxonomy-aware hierarchy saving and adoption.

Covers:
  * Child-first annotation: children created before their parent are not treated
    as colliding siblings and do not carve out bytes/notches from the parent boundary.
  * Edge-touching child: reproduces the boundary notch scenario and asserts the
    parent retains its complete boundary.
  * Atomic reparenting: contained children have parent_id updated to the parent's id
    while preserving IDs, geometries, labels, and provenance.
  * Partial overlap boundary: children extending outside the parent are neither
    reparented nor trimmed, preserving geometry for review.
  * Unrelated branch boundary: children already attached to an unrelated parent are
    never stolen.
  * Intermediate parent insertion: placing Surface between Fragment and contained Polyps.
  * True sibling collision: non-ancestor/descendant peer contours remain mutually exclusive.
"""
import asyncio
import numpy as np
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
import app.database.contours  # noqa: F401
from app.database.contours import Contours, save_contour_tree
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.services.database_access import masks as masks_db
from app.services.database_access import labels as labels_db
from app.services.database_access import annotation_history as history_db
from iquana_toolbox.schemas.database.contours import Contour


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
def env(session):
    """Seed dataset with Fragment -> Surface -> Polyp taxonomy, an image, and a mask."""
    user = Users(username="annotator", hashed_password="pw", is_admin=False)
    session.add(user)
    session.flush()

    ds = Datasets(name="coral_ds", description="", dataset_type="image",
                  folder_path="/tmp/coral_ds", created_by="annotator")
    session.add(ds)
    session.flush()

    # Taxonomy: Fragment (1) -> Surface (2) -> Polyp (3); Other (4)
    l_fragment = Labels(id=1, dataset_id=ds.id, name="Coral Fragment", parent_id=None, value=1)
    l_surface = Labels(id=2, dataset_id=ds.id, name="Coral Surface", parent_id=1, value=2)
    l_polyp = Labels(id=3, dataset_id=ds.id, name="Coral Polyp", parent_id=2, value=3)
    l_other = Labels(id=4, dataset_id=ds.id, name="Other Root", parent_id=None, value=4)
    session.add_all([l_fragment, l_surface, l_polyp, l_other])
    session.flush()

    img = Images(dataset_id=ds.id, file_name="coral.png", file_path="/tmp/coral.png",
                 thumbnail_file_path="/tmp/t.png", width=1000, height=1000,
                 color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    session.add(img)
    session.flush()

    mask = Masks(image_id=img.id, file_path="/tmp/mask.png")
    session.add(mask)
    session.commit()

    return {
        "dataset": ds,
        "image": img,
        "mask": mask,
        "fragment_label_id": 1,
        "surface_label_id": 2,
        "polyp_label_id": 3,
        "other_label_id": 4,
    }


def _box(x1, y1, x2, y2, label_id, parent_id=None):
    """Normalized rectangular contour in clockwise order."""
    return Contour(
        label_id=label_id,
        parent_id=parent_id,
        x=[x1, x2, x2, x1],
        y=[y1, y1, y2, y2],
        confidence=0.95,
    )


def test_child_first_edge_touching_preserves_parent_boundary_and_reparents(session, env):
    """Edge-touching child segmented first does not notch parent and is adopted cleanly."""
    mask_id = env["mask"].id
    polyp_lid = env["polyp_label_id"]
    surface_lid = env["surface_label_id"]

    # 1. Create child polyps first at root level (parent_id=None)
    # Child 1: fully internal [0.4, 0.4] to [0.5, 0.5]
    child_internal = _box(0.4, 0.4, 0.5, 0.5, polyp_lid)
    saved_c1 = asyncio.run(masks_db.add_contour_to_mask(mask_id, child_internal, session))
    assert saved_c1 is not None

    # Child 2: edge-touching child on the left margin [0.2, 0.2] to [0.3, 0.3]
    child_edge = _box(0.2, 0.2, 0.3, 0.3, polyp_lid)
    saved_c2 = asyncio.run(masks_db.add_contour_to_mask(mask_id, child_edge, session))
    assert saved_c2 is not None

    # Both polyps are currently root contours
    c1_row = session.query(Contours).filter_by(id=saved_c1.id).one()
    c2_row = session.query(Contours).filter_by(id=saved_c2.id).one()
    assert c1_row.parent_id is None
    assert c2_row.parent_id is None

    # 2. Add parent Coral Surface [0.2, 0.2] to [0.8, 0.8]
    reparented = []
    parent_surface = _box(0.2, 0.2, 0.8, 0.8, surface_lid)
    saved_parent = asyncio.run(masks_db.add_contour_to_mask(
        mask_id, parent_surface, session, reparented_contour_ids=reparented
    ))
    assert saved_parent is not None

    # 3. Assert parent boundary is NOT carved/notched by the edge-touching child
    # The parent bounding box should span [0.2, 0.8] without missing bytes
    assert min(saved_parent.x) == pytest.approx(0.2, abs=1e-3)
    assert max(saved_parent.x) == pytest.approx(0.8, abs=1e-3)
    assert min(saved_parent.y) == pytest.approx(0.2, abs=1e-3)
    assert max(saved_parent.y) == pytest.approx(0.8, abs=1e-3)

    # 4. Assert both children were adopted atomically
    assert set(reparented) == {saved_c1.id, saved_c2.id}

    # Reload from database and verify hierarchy and preserved attributes
    session.expire_all()
    c1_reloaded = session.query(Contours).filter_by(id=saved_c1.id).one()
    c2_reloaded = session.query(Contours).filter_by(id=saved_c2.id).one()
    assert c1_reloaded.parent_id == saved_parent.id
    assert c2_reloaded.parent_id == saved_parent.id
    assert c1_reloaded.label_id == polyp_lid
    assert c2_reloaded.label_id == polyp_lid
    assert list(c1_reloaded.x) == saved_c1.x
    assert list(c2_reloaded.x) == saved_c2.x

    # In-memory hierarchy reconstruction matches DB
    reloaded_hierarchy = asyncio.run(masks_db.get_contour_hierarchy_of_mask(mask_id, session))
    assert len(reloaded_hierarchy.root_contours) == 1
    root = reloaded_hierarchy.root_contours[0]
    assert root.id == saved_parent.id
    assert len(root.children) == 2
    assert {c.id for c in root.children} == {saved_c1.id, saved_c2.id}


def test_partial_overlap_preserves_geometry_and_does_not_adopt(session, env):
    """Partially overlapping child is not reparented and neither geometry is trimmed."""
    mask_id = env["mask"].id
    polyp_lid = env["polyp_label_id"]
    surface_lid = env["surface_label_id"]

    # Child partially outside: spans [0.15, 0.25] in x, while parent starts at 0.2
    child_partial = _box(0.15, 0.2, 0.25, 0.3, polyp_lid)
    saved_child = asyncio.run(masks_db.add_contour_to_mask(mask_id, child_partial, session))
    assert saved_child is not None

    # Add parent [0.2, 0.2] to [0.8, 0.8]
    reparented = []
    parent_surface = _box(0.2, 0.2, 0.8, 0.8, surface_lid)
    saved_parent = asyncio.run(masks_db.add_contour_to_mask(
        mask_id, parent_surface, session, reparented_contour_ids=reparented
    ))
    assert saved_parent is not None

    # Child is NOT adopted because it is only partially contained
    assert saved_child.id not in reparented
    session.expire_all()
    child_row = session.query(Contours).filter_by(id=saved_child.id).one()
    assert child_row.parent_id is None
    # Child geometry is preserved untouched
    assert list(child_row.x) == saved_child.x

    # Parent boundary is also preserved untouched (overlap is allowed for review)
    assert min(saved_parent.x) == pytest.approx(0.2, abs=1e-3)
    assert max(saved_parent.x) == pytest.approx(0.8, abs=1e-3)


def test_existing_valid_parents_not_stolen_between_unrelated_branches(session, env):
    """A child belonging to an unrelated branch is never stolen by a new parent."""
    mask_id = env["mask"].id
    polyp_lid = env["polyp_label_id"]
    surface_lid = env["surface_label_id"]

    # Surface A on the left: [0.1, 0.1] to [0.5, 0.5]
    surface_a = _box(0.1, 0.1, 0.5, 0.5, surface_lid)
    saved_surface_a = asyncio.run(masks_db.add_contour_to_mask(mask_id, surface_a, session))
    assert saved_surface_a is not None

    # Child polyp attached to Surface A: [0.2, 0.2] to [0.3, 0.3]
    child_on_a = _box(0.2, 0.2, 0.3, 0.3, polyp_lid, parent_id=saved_surface_a.id)
    saved_child = asyncio.run(masks_db.add_contour_to_mask(mask_id, child_on_a, session))
    assert saved_child is not None
    assert saved_child.parent_id == saved_surface_a.id

    # Surface B on an overlapping region: [0.2, 0.2] to [0.8, 0.8]
    reparented = []
    surface_b = _box(0.2, 0.2, 0.8, 0.8, surface_lid)
    saved_surface_b = asyncio.run(masks_db.add_contour_to_mask(
        mask_id, surface_b, session, reparented_contour_ids=reparented
    ))

    # Child already belongs to Surface A (an unrelated sibling branch), so it must NOT be stolen
    assert saved_child.id not in reparented
    session.expire_all()
    child_row = session.query(Contours).filter_by(id=saved_child.id).one()
    assert child_row.parent_id == saved_surface_a.id


def test_intermediate_parent_insertion(session, env):
    """Inserting Surface between Fragment and contained Polyps is valid."""
    mask_id = env["mask"].id
    frag_lid = env["fragment_label_id"]
    surface_lid = env["surface_label_id"]
    polyp_lid = env["polyp_label_id"]

    # 1. Fragment at root: [0.1, 0.1] to [0.9, 0.9]
    frag = _box(0.1, 0.1, 0.9, 0.9, frag_lid)
    saved_frag = asyncio.run(masks_db.add_contour_to_mask(mask_id, frag, session))
    assert saved_frag is not None

    # 2. Polyp temporarily attached to Fragment (before Surface existed): [0.3, 0.3] to [0.4, 0.4]
    polyp = _box(0.3, 0.3, 0.4, 0.4, polyp_lid, parent_id=saved_frag.id)
    saved_polyp = asyncio.run(masks_db.add_contour_to_mask(mask_id, polyp, session))
    assert saved_polyp is not None
    assert saved_polyp.parent_id == saved_frag.id

    # 3. Now insert Surface under Fragment: [0.2, 0.2] to [0.6, 0.6]
    reparented = []
    surface = _box(0.2, 0.2, 0.6, 0.6, surface_lid, parent_id=saved_frag.id)
    saved_surface = asyncio.run(masks_db.add_contour_to_mask(
        mask_id, surface, session, reparented_contour_ids=reparented
    ))
    assert saved_surface is not None
    assert saved_surface.parent_id == saved_frag.id

    # The polyp should be reparented under the intermediate Surface
    assert saved_polyp.id in reparented
    session.expire_all()
    polyp_row = session.query(Contours).filter_by(id=saved_polyp.id).one()
    assert polyp_row.parent_id == saved_surface.id

    # Verify 3-tier tree: Fragment -> Surface -> Polyp
    reloaded_hierarchy = asyncio.run(masks_db.get_contour_hierarchy_of_mask(mask_id, session))
    assert len(reloaded_hierarchy.root_contours) == 1
    root_frag = reloaded_hierarchy.root_contours[0]
    assert root_frag.id == saved_frag.id
    assert len(root_frag.children) == 1
    child_surf = root_frag.children[0]
    assert child_surf.id == saved_surface.id
    assert len(child_surf.children) == 1
    grandchild_polyp = child_surf.children[0]
    assert grandchild_polyp.id == saved_polyp.id


def test_true_siblings_remain_mutually_exclusive(session, env):
    """True siblings (same level, same or non-ancestor/descendant label) collide and clip."""
    mask_id = env["mask"].id
    surface_lid = env["surface_label_id"]

    # Surface 1: [0.2, 0.2] to [0.6, 0.6]
    s1 = _box(0.2, 0.2, 0.6, 0.6, surface_lid)
    saved_s1 = asyncio.run(masks_db.add_contour_to_mask(mask_id, s1, session))
    assert saved_s1 is not None

    # Surface 2 overlaps Surface 1 on [0.4, 0.2] to [0.6, 0.6], extends to 0.8
    s2 = _box(0.4, 0.2, 0.8, 0.6, surface_lid)
    saved_s2 = asyncio.run(masks_db.add_contour_to_mask(mask_id, s2, session))
    assert saved_s2 is not None

    # Surface 2 was clipped against Surface 1: its min x should be >= 0.6 (where Surface 1 ends)
    assert min(saved_s2.x) >= 0.59


def test_undo_parent_creation_restores_adopted_children_previous_parents(session, env):
    """Undoing parent creation restores adopted children's previous parent and preserves them."""
    mask_id = env["mask"].id
    polyp_lid = env["polyp_label_id"]
    surface_lid = env["surface_label_id"]
    username = "annotator"

    # 1. Create a child polyp (unparented root)
    child = _box(0.3, 0.3, 0.4, 0.4, polyp_lid)
    saved_child = asyncio.run(masks_db.add_contour_to_mask(mask_id, child, session))
    assert saved_child is not None
    child_id = saved_child.id

    # 2. Add parent Surface that adopts the child
    parent = _box(0.2, 0.2, 0.6, 0.6, surface_lid)
    reparented_info = []
    saved_parent = asyncio.run(masks_db.add_contour_to_mask(
        mask_id, parent, session, reparented_info=reparented_info
    ))
    assert saved_parent is not None
    assert len(reparented_info) == 1
    assert reparented_info[0]["contour_id"] == child_id
    assert reparented_info[0]["before_parent_id"] is None

    # Record create in history
    history_db.record_create(
        session, mask_id, username, saved_parent.id, adopted_children=reparented_info
    )

    # Verify child is now under parent in DB
    session.expire_all()
    child_row = session.query(Contours).filter_by(id=child_id).one()
    assert child_row.parent_id == saved_parent.id

    # 3. Undo parent creation
    undo_result = history_db.undo(mask_id, username, session)
    assert undo_result["action_type"] == "create"

    # 4. Verify parent was removed, but child survived with previous parent_id (None)
    session.expire_all()
    assert session.query(Contours).filter_by(id=saved_parent.id).first() is None
    child_reloaded = session.query(Contours).filter_by(id=child_id).first()
    assert child_reloaded is not None
    assert child_reloaded.parent_id is None
    assert list(child_reloaded.x) == saved_child.x

    # 5. Redo parent creation
    redo_result = history_db.redo(mask_id, username, session)
    assert redo_result["action_type"] == "create"

    # Verify parent is back and child is re-adopted under it
    session.expire_all()
    assert session.query(Contours).filter_by(id=saved_parent.id).first() is not None
    child_redone = session.query(Contours).filter_by(id=child_id).one()
    assert child_redone.parent_id == saved_parent.id

    # 6. Undo again
    history_db.undo(mask_id, username, session)
    session.expire_all()
    assert session.query(Contours).filter_by(id=saved_parent.id).first() is None
    child_final = session.query(Contours).filter_by(id=child_id).first()
    assert child_final is not None
    assert child_final.parent_id is None


def test_nested_model_output_plus_adoption_assigns_correct_ids(session, env):
    """When a model outputs nested schema children and adopts older children, IDs remain distinct."""
    mask_id = env["mask"].id
    polyp_lid = env["polyp_label_id"]
    surface_lid = env["surface_label_id"]

    # 1. Create an existing older child (will have lower DB id)
    older_child = _box(0.25, 0.25, 0.35, 0.35, polyp_lid)
    saved_older = asyncio.run(masks_db.add_contour_to_mask(mask_id, older_child, session))
    assert saved_older is not None
    older_id = saved_older.id

    # 2. Prepare parent Surface with a NEW nested schema child
    new_nested_child = _box(0.45, 0.45, 0.55, 0.55, polyp_lid)
    parent_surface = _box(0.2, 0.2, 0.8, 0.8, surface_lid)
    parent_surface.children = [new_nested_child]

    reparented_ids = []
    saved_parent = asyncio.run(masks_db.add_contour_to_mask(
        mask_id, parent_surface, session, reparented_contour_ids=reparented_ids
    ))
    assert saved_parent is not None

    # Verify older child was adopted
    assert older_id in reparented_ids

    # Verify the new nested schema child got its own distinct ID (greater than older_id)
    assert len(saved_parent.children) == 1
    new_child_schema_id = saved_parent.children[0].id
    assert new_child_schema_id is not None
    assert new_child_schema_id != older_id
    assert new_child_schema_id > older_id

    # In the database, both children exist under saved_parent
    session.expire_all()
    children_in_db = (
        session.query(Contours)
        .filter(Contours.parent_id == saved_parent.id)
        .order_by(Contours.id)
        .all()
    )
    db_ids = [c.id for c in children_in_db]
    assert older_id in db_ids
    assert new_child_schema_id in db_ids


def test_batch_adoption_multi_level_ancestors(session, env):
    """Batch adoption traverses DB ancestors when hierarchy is None."""
    ds_id = env["dataset"].id
    mask_id = env["mask"].id

    # Setup 4-tier taxonomy: Colony (10) -> Fragment (11) -> Surface (12) -> Polyp (13)
    l_colony = Labels(id=10, dataset_id=ds_id, name="Coral Colony", parent_id=None, value=10)
    l_frag = Labels(id=11, dataset_id=ds_id, name="Coral Fragment 4T", parent_id=10, value=11)
    l_surf = Labels(id=12, dataset_id=ds_id, name="Coral Surface 4T", parent_id=11, value=12)
    l_polyp = Labels(id=13, dataset_id=ds_id, name="Coral Polyp 4T", parent_id=12, value=13)
    session.add_all([l_colony, l_frag, l_surf, l_polyp])
    session.flush()

    label_hierarchy = labels_db.get_label_hierarchy_sync(ds_id, session)

    # 1. Create Colony at root
    colony = _box(0.05, 0.05, 0.95, 0.95, 10)
    saved_colony = save_contour_tree(session, colony, mask_id, parent_id=None)

    # 2. Create Fragment under Colony
    frag = _box(0.1, 0.1, 0.9, 0.9, 11)
    saved_frag = save_contour_tree(session, frag, mask_id, parent_id=saved_colony.id)

    # 3. Create Polyp attached directly to Colony (skipping Fragment & Surface)
    polyp = _box(0.3, 0.3, 0.4, 0.4, 13)
    saved_polyp = save_contour_tree(session, polyp, mask_id, parent_id=saved_colony.id)
    session.commit()

    # 4. Now run batch adoption for Surface under Fragment (hierarchy=None)
    surface = _box(0.2, 0.2, 0.6, 0.6, 12)
    saved_surface = save_contour_tree(session, surface, mask_id, parent_id=saved_frag.id)

    reparented_info = []
    adopted_ids = masks_db.adopt_descendants_for_parent(
        session, mask_id, saved_surface, surface, label_hierarchy,
        hierarchy=None, reparented_info=reparented_info
    )
    session.commit()

    # Polyp attached to Colony (2 levels up) was recognized and adopted under Surface
    assert saved_polyp.id in adopted_ids
    assert len(reparented_info) == 1
    assert reparented_info[0]["contour_id"] == saved_polyp.id
    assert reparented_info[0]["before_parent_id"] == saved_colony.id

    session.expire_all()
    polyp_row = session.query(Contours).filter_by(id=saved_polyp.id).one()
    assert polyp_row.parent_id == saved_surface.id


def test_empty_dataset_label_hierarchy(session):
    """Empty datasets return an empty LabelHierarchy without raising ValueError."""
    user = Users(username="empty_user", hashed_password="pw", is_admin=False)
    session.add(user)
    session.flush()

    ds_empty = Datasets(name="empty_ds", description="", dataset_type="image",
                        folder_path="/tmp/empty_ds", created_by="empty_user")
    session.add(ds_empty)
    session.commit()

    # Async method returns empty LabelHierarchy
    lh_async = asyncio.run(labels_db.get_label_hierarchy(ds_empty.id, session))
    assert len(lh_async) == 0
    assert lh_async.root_level_labels == []
    assert lh_async.id_to_label_object == {}
    assert lh_async.value_to_label_object == {}

    # Sync method returns empty LabelHierarchy
    lh_sync = labels_db.get_label_hierarchy_sync(ds_empty.id, session)
    assert len(lh_sync) == 0
    assert lh_sync.root_level_labels == []
    assert lh_sync.id_to_label_object == {}
    assert lh_sync.value_to_label_object == {}


def test_empty_dataset_add_contour(session):
    """Adding a contour to a mask in an empty dataset succeeds without failing on taxonomy."""
    user = Users(username="empty_user2", hashed_password="pw", is_admin=False)
    session.add(user)
    session.flush()

    ds_empty = Datasets(name="empty_ds2", description="", dataset_type="image",
                        folder_path="/tmp/empty_ds2", created_by="empty_user2")
    session.add(ds_empty)
    session.flush()

    img = Images(dataset_id=ds_empty.id, file_name="coral_empty.png", file_path="/tmp/ce.png",
                 thumbnail_file_path="/tmp/tce.png", width=1000, height=1000,
                 color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    session.add(img)
    session.flush()

    mask = Masks(image_id=img.id, file_path="/tmp/mask_empty.png")
    session.add(mask)
    session.commit()

    box = _box(0.1, 0.1, 0.5, 0.5, label_id=None)
    added = asyncio.run(masks_db.add_contour_to_mask(mask.id, box, session))
    assert added is not None
    assert added.id is not None


def test_zero_valued_label_conversion_and_no_clipping(session):
    """Labels with value=0 (e.g. Background) are preserved during conversion, preventing damaged saves."""
    user = Users(username="zero_user", hashed_password="pw", is_admin=False)
    session.add(user)
    session.flush()

    ds = Datasets(name="ds_zero", description="", dataset_type="image",
                  folder_path="/tmp/ds_zero", created_by="zero_user")
    session.add(ds)
    session.flush()

    # Taxonomy: Background (value=0), Fragment (value=1) -> Surface (value=2)
    bg = Labels(id=100, dataset_id=ds.id, name="Background", parent_id=None, value=0)
    frag = Labels(id=101, dataset_id=ds.id, name="Coral Fragment", parent_id=None, value=1)
    surf = Labels(id=102, dataset_id=ds.id, name="Coral Surface", parent_id=101, value=2)
    session.add_all([bg, frag, surf])
    session.flush()

    # 1. Verify Label.from_db preserves value=0
    bg_schema = labels_db.Label.from_db(bg)
    assert bg_schema.value == 0
    assert bg_schema.name == "Background"

    # 2. Verify get_label_hierarchy preserves value=0
    lh = asyncio.run(labels_db.get_label_hierarchy(ds.id, session))
    assert 0 in lh.value_to_label_object
    assert lh.value_to_label_object[0].name == "Background"
    assert lh.id_to_value_map[100] == 0

    # 3. Verify get_label preserves value=0
    single_label = asyncio.run(labels_db.get_label(100, session))
    assert single_label.value == 0

    # 4. Verify create_label preserves value=0
    new_zero_lbl = asyncio.run(labels_db.create_label("Extra BG", ds.id, session, label_value=0))
    assert new_zero_lbl.value == 0

    # 5. Verify child-first save with zero-valued label present does not clip parent
    img = Images(dataset_id=ds.id, file_name="zero.png", file_path="/tmp/zero.png",
                 thumbnail_file_path="/tmp/tzero.png", width=1000, height=1000,
                 color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    session.add(img)
    session.flush()

    mask = Masks(image_id=img.id, file_path="/tmp/mask_zero.png")
    session.add(mask)
    session.commit()

    # Create child Surface at root first (child-first annotation)
    child_surf = _box(0.2, 0.2, 0.6, 0.6, label_id=102)
    saved_child = asyncio.run(masks_db.add_contour_to_mask(mask.id, child_surf, session))
    assert saved_child is not None

    # Now create parent Fragment (covering child Surface)
    parent_frag = _box(0.1, 0.1, 0.9, 0.9, label_id=101)
    reparented_ids = []
    saved_parent = asyncio.run(masks_db.add_contour_to_mask(
        mask.id, parent_frag, session, reparented_contour_ids=reparented_ids
    ))

    # Parent must NOT be clipped or shrunk
    assert saved_parent is not None
    assert len(saved_parent.x) == 4
    # Child was adopted
    assert saved_child.id in reparented_ids


def test_taxonomy_load_error_propagated_not_swallowed(session, env, monkeypatch):
    """When a populated dataset's taxonomy fails to load, the error is propagated instead of clipping."""
    mask_id = env["mask"].id
    frag_lid = env["fragment_label_id"]

    # Mock get_label_hierarchy to simulate a taxonomy load/validation failure
    async def mock_fail(*args, **kwargs):
        raise ValueError("Corrupt taxonomy structure")

    monkeypatch.setattr(labels_db, "get_label_hierarchy", mock_fail)

    frag = _box(0.1, 0.1, 0.9, 0.9, frag_lid)
    with pytest.raises(RuntimeError, match="Failed to load taxonomy"):
        asyncio.run(masks_db.add_contour_to_mask(mask_id, frag, session))


def test_empty_dataset_http_get_labels(session):
    """GET /{dataset_id}/labels returns 200 with empty hierarchy when dataset has no labels."""
    from app.routes.general.datasets import get_labels
    from app.schemas.auth_user import AuthenticatedUser

    user = Users(username="admin_http", hashed_password="pw", is_admin=True)
    session.add(user)
    session.flush()

    ds_empty = Datasets(name="http_empty", description="", dataset_type="image",
                        folder_path="/tmp/http_empty", created_by="admin_http")
    session.add(ds_empty)
    session.commit()

    auth_user = AuthenticatedUser.from_query(user)
    data = asyncio.run(get_labels(dataset_id=ds_empty.id, db=session, user=auth_user))
    assert data["success"] is True
    assert data["message"] == f"Retrieved 0 labels for dataset {ds_empty.id}."
    assert data["labels"]["root_level_labels"] == []


"""Tests for Step 5: quantification profiles + the generic metric catalog.

Covers, against a temp-file SQLite database (same pattern as test_contour_metrics.py):
  * default-profile auto-creation (geometry-on-all-labels),
  * CRUD: create / update / delete + is_default exclusivity + default-promotion on delete,
  * metric-key validation rejects unknown keys,
  * ``list_metrics`` catalog exposes all 9 metrics with component names,
  * ``get_quantification_summary(metric_scoping=...)`` restricts metrics AND honors
    per-metric label scoping,
  * profile export (``get_dataset_as_df`` with scoping) emits per-component columns.
"""
import asyncio

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
import app.database.quantification_profiles  # noqa: F401
from app.database.contour_metrics import ContourMetrics
from app.database.contours import Contours, save_contour_tree
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.services.database_access import datasets as datasets_db
from app.services.database_access import quantification_profiles as profiles_db

from iquana_toolbox.quantification import list_metrics
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.quantification_profile import (
    ProfileEntry,
    QuantificationProfile,
)

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
    """Dataset -> image -> mask -> two labels -> two 'cell' contours + one 'nucleus' child."""
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    ds = Datasets(name="smoke", description="", dataset_type="image",
                  folder_path="/tmp/smoke", created_by="u")
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
    child_label.parent_id = parent_label.id
    session.flush()

    px1, py1 = _rect([100, 200, 200, 100], [100, 100, 150, 150])
    px2, py2 = _rect([300, 500, 500, 300], [200, 200, 300, 300])
    cpx, cpy = _rect([120, 160, 160, 120], [110, 110, 130, 130])

    tree1 = Contour(x=px1, y=py1, label_id=parent_label.id, added_by="u",
                    children=[Contour(x=cpx, y=cpy, label_id=child_label.id, added_by="u")])
    tree2 = Contour(x=px2, y=py2, label_id=parent_label.id, added_by="u")
    tree1.reviewed_by = ["u"]
    tree2.reviewed_by = ["u"]
    tree1.children[0].reviewed_by = ["u"]
    save_contour_tree(session, tree1, mask.id)
    save_contour_tree(session, tree2, mask.id)
    session.commit()

    return ds, parent_label, child_label


# --- Catalog -----------------------------------------------------------------

def test_metrics_catalog_lists_all_metrics():
    catalog = list_metrics()
    assert len(catalog) == 10
    keys = {m["key"] for m in catalog}
    assert {"area", "perimeter", "circularity", "max_diameter",
            "mean_color_rgb", "mean_color_lab", "mean_intensity",
            "nn_distance", "mean_knn_distance", "n_children"} <= keys
    rgb = next(m for m in catalog if m["key"] == "mean_color_rgb")
    assert rgb["value_dim"] == 3
    assert rgb["components"] == ["R", "G", "B"]
    area = next(m for m in catalog if m["key"] == "area")
    assert area["components"] is None  # single-component -> no component names


# --- Default profile ---------------------------------------------------------

def test_default_profile_auto_created(session):
    ds, _parent, _child = _seed(session)
    profiles = profiles_db.list_profiles(session, ds.id)
    assert len(profiles) == 1
    default = profiles[0]
    assert default.is_default is True
    assert default.metric_keys() == ["area", "perimeter", "circularity", "max_diameter"]
    # Every entry scopes to all labels.
    assert all(e.label_ids is None for e in default.entries)
    # Idempotent: listing again does not create a second default.
    assert len(profiles_db.list_profiles(session, ds.id)) == 1


# --- CRUD + is_default exclusivity -------------------------------------------

def test_crud_and_default_exclusivity(session):
    ds, parent, _child = _seed(session)
    profiles_db.list_profiles(session, ds.id)  # ensure default exists

    created = profiles_db.create_profile(session, QuantificationProfile(
        dataset_id=ds.id, name="Color", is_default=True,
        entries=[ProfileEntry(metric_key="mean_color_rgb")],
    ))
    assert created.id is not None
    assert created.is_default is True

    # The auto default should have been unset.
    all_profiles = profiles_db.list_profiles(session, ds.id)
    defaults = [p for p in all_profiles if p.is_default]
    assert len(defaults) == 1 and defaults[0].id == created.id

    # Update: rename + change entries + keep default.
    row = profiles_db.get_profile(session, ds.id, created.id)
    updated = profiles_db.update_profile(session, row, QuantificationProfile(
        id=created.id, dataset_id=ds.id, name="Color v2", is_default=True,
        entries=[ProfileEntry(metric_key="mean_color_lab", label_ids=[parent.id])],
    ))
    assert updated.name == "Color v2"
    assert updated.metric_keys() == ["mean_color_lab"]
    assert updated.entries[0].label_ids == [parent.id]

    # Delete the default -> the remaining (old geometry default) is promoted.
    row = profiles_db.get_profile(session, ds.id, created.id)
    profiles_db.delete_profile(session, row)
    remaining = profiles_db.list_profiles(session, ds.id)
    assert all(p.id != created.id for p in remaining)
    assert sum(1 for p in remaining if p.is_default) == 1


def test_unknown_metric_key_rejected():
    with pytest.raises(ValueError):
        ProfileEntry(metric_key="not_a_real_metric")
    with pytest.raises(ValueError):
        QuantificationProfile(dataset_id=1, name="bad",
                              entries=[{"metric_key": "nope"}])


# --- Summary scoping ---------------------------------------------------------

def test_summary_respects_metric_scoping(session):
    ds, parent, child = _seed(session)

    # No scoping: all geometry metrics present for both labels.
    full = asyncio.run(datasets_db.get_quantification_summary(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session,
    ))
    assert "area" in full["metrics"][str(parent.id)]
    assert "perimeter" in full["metrics"][str(parent.id)]

    # Scope to area only, and only for the parent ('cell') label.
    scoping = {"area": [parent.id]}
    scoped = asyncio.run(datasets_db.get_quantification_summary(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session,
        metric_scoping=scoping,
    ))
    parent_metrics = scoped["metrics"][str(parent.id)]
    assert set(parent_metrics) == {"area"}  # only the scoped metric
    # The child label ('nucleus') must NOT get an 'area' entry (out of scope).
    assert str(child.id) not in scoped["metrics"] or "area" not in scoped["metrics"][str(child.id)]

    # Empty profile -> no metrics but child counts still present.
    empty = asyncio.run(datasets_db.get_quantification_summary(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session,
        metric_scoping={},
    ))
    assert empty["metrics"] == {}
    assert empty["child_counts_per_label_id"]  # cell has a nucleus child


# --- Profile export ----------------------------------------------------------

def test_export_with_profile_emits_per_component_columns(session):
    ds, parent, child = _seed(session)

    # Fake a multi-component metric row so the export has components to split.
    cell_ids = [
        c.id for c in session.query(Contours)
        .filter(Contours.label_id == parent.id, Contours.parent_id.is_(None)).all()
    ]
    for cid in cell_ids:
        for comp, val in enumerate([10.0, 20.0, 30.0]):
            session.add(ContourMetrics(contour_id=cid, metric_key="mean_color_rgb",
                                       component=comp, value=val, unit="", stale=False))
    session.commit()

    scoping = {"area": None, "mean_color_rgb": [parent.id]}
    df = asyncio.run(datasets_db.get_dataset_as_df(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session,
        metric_scoping=scoping,
    ))
    # Per-component columns for the color metric, single column for area.
    for col in ["area", "mean_color_rgb_r", "mean_color_rgb_g", "mean_color_rgb_b"]:
        assert col in df.columns, f"missing column {col}"
    # Parent ('cell') rows carry the color values; nucleus rows are out of scope -> null.
    cell_rows = df[df["label_id"] == parent.id]
    assert (cell_rows["mean_color_rgb_r"] == 10.0).all()
    nucleus_rows = df[df["label_id"] == child.id]
    assert nucleus_rows["mean_color_rgb_r"].isna().all()


# --- Legacy export unchanged -------------------------------------------------

def test_export_without_profile_is_legacy_shape(session):
    ds, _parent, _child = _seed(session)
    df = asyncio.run(datasets_db.get_dataset_as_df(
        ds.id, exclude_not_fully_annotated=True, exclude_unreviewed=True, db=session,
    ))
    for col in ["area", "perimeter", "circularity", "diameter_avg", "coords_x", "coords_y"]:
        assert col in df.columns
    assert "mean_color_rgb_r" not in df.columns

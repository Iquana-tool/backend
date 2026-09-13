"""Tests for scoping the quantification reads to a single image (``image_id``).

The per-image inspection view is the same aggregation as the dataset one, one image
wide, so what is worth proving is that the two agree: an image's numbers must be exactly
what the dataset numbers would be if the other images did not exist, the object census
must narrow with them, and the export frame must carry the same rows the summary counted.

The scale case has its own test because scoping changes the answer rather than narrowing
it: a dataset whose images disagree on units reports pixels, but a single calibrated image
out of that same dataset is trivially consistent with itself and reports its own unit.
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
from app.database.contours import Contours, save_contour_tree
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users

from app.services.database_access.datasets import (
    get_dataset_as_df,
    get_quantification_summary,
)
from app.services.quantification import compute_geometry_metrics_for_dataset

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


def _square(cx_px, cy_px, half):
    """A normalized axis-aligned square of side ``2 * half`` pixels."""
    x_px = [cx_px - half, cx_px + half, cx_px + half, cx_px - half]
    y_px = [cy_px - half, cy_px - half, cy_px + half, cy_px + half]
    return ([x / WIDTH for x in x_px], [y / HEIGHT for y in y_px])


def _add(session, mask_id, label_id, cx, half, reviewed=True):
    px, py = _square(cx, 500, half)
    schema = Contour(x=px, y=py, label_id=label_id, added_by="u",
                     reviewed_by=(["u"] if reviewed else []))
    save_contour_tree(session, schema, mask_id)


def _image(session, dataset_id, name, unit="px", scale=1.0):
    img = Images(dataset_id=dataset_id, file_name=name, file_path=f"/tmp/{name}",
                 thumbnail_file_path=f"/tmp/t_{name}", width=WIDTH, height=HEIGHT,
                 color_mode="RGB", scale_x=scale, scale_y=scale, unit=unit)
    session.add(img)
    session.flush()
    mask = Masks(image_id=img.id, fully_annotated=True, file_path=f"/tmp/m_{name}")
    session.add(mask)
    session.flush()
    return img, mask


def _seed(session, units=("px", "px")):
    """A dataset with two annotated images: 10px squares on A, 20px squares on B."""
    ds = Datasets(name="per-image", description="", dataset_type="image",
                  folder_path="/tmp/per-image", created_by="u")
    session.add(Users(username="u", hashed_password="x", is_admin=False))
    session.add(ds)
    session.flush()

    label = Labels(dataset_id=ds.id, parent_id=None, name="cell", value=1)
    session.add(label)
    session.flush()

    img_a, mask_a = _image(session, ds.id, "a.png", unit=units[0])
    img_b, mask_b = _image(session, ds.id, "b.png", unit=units[1])

    # Two 20x20px objects on A, one 40x40px object on B, so every aggregate (count, mean,
    # min, max) differs per image and a leak from the other image cannot go unnoticed.
    _add(session, mask_a.id, label.id, 100, half=10)
    _add(session, mask_a.id, label.id, 300, half=10)
    _add(session, mask_b.id, label.id, 100, half=20)
    session.commit()

    compute_geometry_metrics_for_dataset(session, ds.id)
    session.commit()
    return ds, label, img_a, img_b


def _summary(ds_id, session, image_id=None):
    return asyncio.run(get_quantification_summary(
        ds_id, exclude_not_fully_annotated=False, exclude_unreviewed=False,
        db=session, image_id=image_id,
    ))


def test_summary_scoped_to_one_image(session):
    ds, label, img_a, img_b = _seed(session)
    key = str(label.id)

    whole = _summary(ds.id, session)
    only_a = _summary(ds.id, session, image_id=img_a.id)
    only_b = _summary(ds.id, session, image_id=img_b.id)

    # The dataset sees all three objects; each image sees only its own.
    assert whole["metrics"][key]["area"]["components"][0]["count"] == 3
    assert only_a["metrics"][key]["area"]["components"][0]["count"] == 2
    assert only_b["metrics"][key]["area"]["components"][0]["count"] == 1

    # A's objects are 20x20px, B's is 40x40px. Scoped means scoped: A's mean must be A's
    # own area, not something pulled toward B.
    assert only_a["metrics"][key]["area"]["components"][0]["mean"] == pytest.approx(400.0)
    assert only_b["metrics"][key]["area"]["components"][0]["mean"] == pytest.approx(1600.0)
    # Two identical objects on A -> no spread, which a leaked third object would break.
    assert only_a["metrics"][key]["area"]["components"][0]["std"] == pytest.approx(0.0)


def test_object_census_scoped_to_one_image(session):
    ds, label, img_a, img_b = _seed(session)
    key = str(label.id)

    # The census ignores the two exclude filters on purpose, but the image scope is not one
    # of them - it decides which objects are being counted at all.
    assert _summary(ds.id, session)["object_counts_per_label_id"][key]["total"] == 3
    assert _summary(ds.id, session, img_a.id)["object_counts_per_label_id"][key]["total"] == 2
    assert _summary(ds.id, session, img_b.id)["object_counts_per_label_id"][key]["total"] == 1


def test_export_carries_the_contour_hierarchy(session):
    """Children name their parent, so a pivot can group under it (issue #15 follow-up)."""
    ds, label, img_a, img_b = _seed(session)

    child_label = Labels(dataset_id=ds.id, parent_id=label.id, name="nucleus", value=2)
    session.add(child_label)
    session.flush()

    # A child contour under the first object on image A.
    parent = (
        session.query(Contours)
        .join(Masks, Masks.id == Contours.mask_id)
        .filter(Masks.image_id == img_a.id)
        .first()
    )
    px, py = _square(100, 500, 3)
    save_contour_tree(
        session,
        Contour(x=px, y=py, label_id=child_label.id, added_by="u", reviewed_by=["u"]),
        parent.mask_id,
        parent_id=parent.id,
    )
    session.commit()
    compute_geometry_metrics_for_dataset(session, ds.id)
    session.commit()

    df = asyncio.run(get_dataset_as_df(
        ds.id, False, False, session, metric_scoping={"area": None}, image_id=img_a.id))

    child_row = df[df["label"] == "nucleus"].iloc[0]
    assert child_row["parent_id"] == parent.id
    # The id alone is not groupable by a human; the label is what makes the pivot readable.
    assert child_row["parent_label"] == "cell"

    # Root-level objects have no parent, and must say so rather than borrow one.
    root_rows = df[df["label"] == "cell"]
    assert root_rows["parent_id"].isna().all()
    assert root_rows["parent_label"].isna().all()


def test_export_frame_scoped_to_one_image(session):
    ds, label, img_a, img_b = _seed(session)
    scoping = {"area": None}

    whole = asyncio.run(get_dataset_as_df(
        ds.id, False, False, session, metric_scoping=scoping))
    only_a = asyncio.run(get_dataset_as_df(
        ds.id, False, False, session, metric_scoping=scoping, image_id=img_a.id))

    assert len(whole) == 3
    assert len(only_a) == 2
    # The table on the per-image page and the count on its card come from these two reads,
    # so they have to agree about which image the rows belong to.
    assert set(only_a["file_name"]) == {"a.png"}
    assert list(only_a.columns) == list(whole.columns)


def test_scale_status_is_resolved_per_image(session):
    """A mixed-unit dataset reports pixels; one calibrated image out of it reports mm."""
    ds, label, img_a, img_b = _seed(session, units=("mm", "px"))
    # A is calibrated at 0.5 mm/px, B is not calibrated at all.
    session.query(Images).filter(Images.id == img_a.id).update({"scale_x": 0.5, "scale_y": 0.5})
    session.commit()

    whole = _summary(ds.id, session)
    only_a = _summary(ds.id, session, image_id=img_a.id)
    key = str(label.id)

    # Dataset-wide the units disagree, so everything falls back to pixels and the frontend
    # shows its warning banner.
    assert whole["scale_status"]["consistent"] is False
    assert whole["scale_status"]["display_unit"] == "px"

    # Scoped to A there is nothing to disagree with, so A's own scale applies: a 20x20px
    # square at 0.5 mm/px is 10x10 mm = 100 mm².
    assert only_a["scale_status"]["consistent"] is True
    assert only_a["scale_status"]["display_unit"] == "mm"
    assert only_a["metrics"][key]["area"]["components"][0]["mean"] == pytest.approx(100.0)

"""Grouping a quantification by an image-metadata key.

Metadata is effectively an image-wide label that every object on the image
inherits, so this is a GROUP BY on the existing aggregation rather than a new
kind of metric. What has to hold:

  * the dataset-wide numbers are **unchanged** by asking for a grouping, so a
    client that ignores it sees exactly what it saw before,
  * every contour lands in exactly one bucket — including those on images with no
    value for the key, which is what the outer join is for. If they vanished, the
    groups would silently sum to less than the dataset,
  * a grouped number is the same arithmetic as the ungrouped one,
  * keys that cannot sensibly group (a number, a date, free text) are refused
    rather than rendered as one band per image.
"""
import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.database import database, get_session
import app.database.dataset_members  # noqa: F401
import app.database.dataset_metadata_keys  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.image_metadata  # noqa: F401
import app.database.images  # noqa: F401
import app.database.rejections  # noqa: F401
import app.database.users  # noqa: F401
from app.database.contour_metrics import ContourMetrics
from app.database.contours import Contours
from app.database.dataset_members import DatasetMembers
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.exceptions import InvalidMetadataError
from app.routes.general.datasets import router as datasets_router
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import DatasetRole
from app.services.auth import get_current_user
from app.services.database_access import datasets as datasets_db
from app.services.database_access import image_metadata as meta
from app.services.database_access.datasets import UNTAGGED_GROUP
from app.services.metadata_types import MetadataValueType


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


#: Three images: two tagged site=reef_a with areas 10 and 20, one site=reef_b
#: with area 60. The means (15 vs 60) differ from the dataset mean (30), so a
#: grouped number can never accidentally pass by matching the ungrouped one.
_AREAS = {0: 10.0, 1: 20.0, 2: 60.0}
_SITES = {0: "reef_a", 1: "reef_a", 2: "reef_b"}


@pytest.fixture
def ctx(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'grouping.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    reviewer = Users(username="curator", hashed_password="x")
    db.add_all([reviewer, Users(username="ann", hashed_password="x")])
    ds = Datasets(name="ds", description="", dataset_type="image",
                  folder_path=str(tmp_path), created_by="curator")
    db.add(ds)
    db.flush()
    db.add(DatasetMembers(dataset_id=ds.id, username="curator",
                          role=DatasetRole.CURATOR.value,
                          extra_permissions=[], denied_permissions=[]))
    label = Labels(dataset_id=ds.id, name="coral", value=1)
    db.add(label)
    db.flush()

    image_ids = []
    for index in range(3):
        image = Images(dataset_id=ds.id, file_name=f"img_{index}.png",
                       file_path=str(tmp_path / f"img_{index}.png"),
                       thumbnail_file_path=str(tmp_path / f"t_{index}.png"),
                       width=100, height=80, color_mode="RGB")
        db.add(image)
        db.flush()
        mask = Masks(image_id=image.id, fully_annotated=True,
                     file_path=str(tmp_path / f"m_{index}.png"))
        db.add(mask)
        db.flush()
        contour = Contours(mask_id=mask.id, label_id=label.id, added_by="manual",
                           author_username="ann", confidence_score=1.0,
                           area=_AREAS[index], perimeter=1.0, circularity=0.5,
                           diameter=1.0,
                           x=[0.1, 0.5, 0.5, 0.1], y=[0.1, 0.1, 0.5, 0.5])
        contour.reviewed_by.append(reviewer)
        db.add(contour)
        db.flush()
        db.add(ContourMetrics(contour_id=contour.id, metric_key="area",
                              component=0, value=_AREAS[index], unit="", stale=False))
        image_ids.append(image.id)
    db.commit()

    yield {"db": db, "Session": Session, "dataset_id": ds.id,
           "image_ids": image_ids, "label_id": label.id}
    db.close()
    engine.dispose()


def _tag_all(ctx):
    """Give every image its site."""
    for index, image_id in enumerate(ctx["image_ids"]):
        meta.set_metadata_for_images(ctx["db"], [image_id], {"site": _SITES[index]})


def _summary(ctx, **kwargs):
    return asyncio.run(datasets_db.get_quantification_summary(
        ctx["dataset_id"], True, True, ctx["db"], **kwargs
    ))


def _area_mean(block, label_id):
    return block[str(label_id)]["area"]["components"][0]["mean"]


def _area_count(block, label_id):
    return block[str(label_id)]["area"]["components"][0]["count"]


# ---------------------------------------------------------------------------
# The aggregation
# ---------------------------------------------------------------------------

def test_without_grouping_the_response_is_unchanged(ctx):
    _tag_all(ctx)
    summary = _summary(ctx)
    assert "groups" not in summary
    assert _area_mean(summary["metrics"], ctx["label_id"]) == 30.0


def test_grouping_splits_the_metric_by_metadata_value(ctx):
    _tag_all(ctx)
    summary = _summary(ctx, group_by_key="site")

    assert summary["group_values"] == ["reef_a", "reef_b"]
    assert _area_mean(summary["groups"]["reef_a"], ctx["label_id"]) == 15.0
    assert _area_mean(summary["groups"]["reef_b"], ctx["label_id"]) == 60.0


def test_the_dataset_wide_numbers_survive_grouping(ctx):
    """A client that ignores `groups` must see what it saw before."""
    _tag_all(ctx)
    plain = _summary(ctx)
    grouped = _summary(ctx, group_by_key="site")
    assert grouped["metrics"] == plain["metrics"]
    assert grouped["object_counts_per_label_id"] == plain["object_counts_per_label_id"]


def test_every_contour_lands_in_exactly_one_bucket(ctx):
    _tag_all(ctx)
    summary = _summary(ctx, group_by_key="site")
    total = sum(_area_count(block, ctx["label_id"])
                for block in summary["groups"].values())
    assert total == _area_count(summary["metrics"], ctx["label_id"]) == 3


def test_untagged_images_get_their_own_bucket_rather_than_disappearing(ctx):
    """The outer join is the whole point: an inner one would make the groups sum
    to less than the dataset, with nothing on screen to say so."""
    meta.set_metadata_for_images(ctx["db"], ctx["image_ids"][:2], {"site": "reef_a"})

    summary = _summary(ctx, group_by_key="site")

    assert set(summary["groups"]) == {"reef_a", UNTAGGED_GROUP}
    assert _area_count(summary["groups"][UNTAGGED_GROUP], ctx["label_id"]) == 1
    assert _area_mean(summary["groups"][UNTAGGED_GROUP], ctx["label_id"]) == 60.0


def test_untagged_sorts_last(ctx):
    """It is the residue, not a category; leading with it reads wrong."""
    meta.set_metadata_for_images(ctx["db"], [ctx["image_ids"][0]], {"site": "zzz"})
    summary = _summary(ctx, group_by_key="site")
    assert summary["group_values"][-1] == UNTAGGED_GROUP


def test_numeric_group_values_sort_numerically(ctx):
    """A depth-like categorical must read 2, 12, 30 rather than 12, 2, 30."""
    for image_id, depth in zip(ctx["image_ids"], ["12", "2", "30"]):
        meta.set_metadata_for_images(ctx["db"], [image_id], {"band": depth})

    summary = _summary(ctx, group_by_key="band")
    assert summary["group_values"] == ["2", "12", "30"]


def test_grouping_respects_the_exclude_filters(ctx):
    """Grouping must not become a way around the review filter."""
    db = ctx["db"]
    _tag_all(ctx)
    mask = db.query(Masks).join(Images).filter(
        Images.id == ctx["image_ids"][2]).first()
    mask.fully_annotated = False
    db.commit()

    summary = _summary(ctx, group_by_key="site")
    assert set(summary["groups"]) == {"reef_a"}


def test_profile_scoping_still_applies_within_a_group(ctx):
    _tag_all(ctx)
    summary = _summary(ctx, group_by_key="site", metric_scoping={"area": []})
    # The metric is scoped to no labels at all, so every bucket empties out and
    # none is emitted rather than drawing blank bands.
    assert summary["groups"] == {}


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def test_distribution_gains_a_group_level(ctx):
    _tag_all(ctx)
    distribution = asyncio.run(datasets_db.get_quantification_distribution(
        ctx["dataset_id"], True, True, ctx["db"], group_by_key="site"
    ))

    assert set(distribution) == {"reef_a", "reef_b"}
    reef_a = distribution["reef_a"][str(ctx["label_id"])]["area"]["0"]
    assert reef_a["count"] == 2
    assert reef_a["median"] == 15.0


def test_ungrouped_distribution_keeps_its_shape(ctx):
    _tag_all(ctx)
    distribution = asyncio.run(datasets_db.get_quantification_distribution(
        ctx["dataset_id"], True, True, ctx["db"]
    ))
    assert set(distribution) == {str(ctx["label_id"])}


# ---------------------------------------------------------------------------
# Which keys may group
# ---------------------------------------------------------------------------

def test_a_numeric_key_is_refused_as_a_grouping(ctx):
    """One band per image is not a comparison; binning would be needed first."""
    db = ctx["db"]
    meta.ensure_key(db, ctx["dataset_id"], "depth", MetadataValueType.NUMBER)
    db.commit()

    with pytest.raises(InvalidMetadataError) as excinfo:
        meta.assert_groupable(db, ctx["dataset_id"], "depth")
    assert "number" in str(excinfo.value)


def test_a_text_key_is_refused_as_a_grouping(ctx):
    db = ctx["db"]
    meta.ensure_key(db, ctx["dataset_id"], "notes", MetadataValueType.TEXT)
    db.commit()
    with pytest.raises(InvalidMetadataError):
        meta.assert_groupable(db, ctx["dataset_id"], "notes")


def test_categorical_and_boolean_keys_may_group(ctx):
    db = ctx["db"]
    meta.ensure_key(db, ctx["dataset_id"], "site", MetadataValueType.CATEGORICAL)
    meta.ensure_key(db, ctx["dataset_id"], "bleached", MetadataValueType.BOOLEAN)
    db.commit()

    assert meta.assert_groupable(db, ctx["dataset_id"], "site") == "site"
    assert meta.assert_groupable(db, ctx["dataset_id"], "bleached") == "bleached"


def test_an_unknown_key_is_refused(ctx):
    with pytest.raises(InvalidMetadataError):
        meta.assert_groupable(ctx["db"], ctx["dataset_id"], "nope")


# ---------------------------------------------------------------------------
# Over HTTP
# ---------------------------------------------------------------------------

@pytest.fixture
def client(ctx):
    Session = ctx["Session"]
    app = FastAPI()
    app.include_router(datasets_router)

    def _session_override():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    def _user_override():
        session = Session()
        try:
            row = session.query(Users).filter_by(username="curator").one()
            return AuthenticatedUser.from_query(row)
        finally:
            session.close()

    app.dependency_overrides[get_session] = _session_override
    app.dependency_overrides[get_current_user] = _user_override
    return TestClient(app)


def test_summary_endpoint_returns_the_groups(ctx, client):
    _tag_all(ctx)
    response = client.get(
        f"/datasets/{ctx['dataset_id']}/quantification/summary",
        params={"group_by": "site", "include_appearance": False,
                "include_contextual": False, "include_relational": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["group_by"] == "site"
    assert body["group_values"] == ["reef_a", "reef_b"]
    assert _area_mean(body["groups"]["reef_a"], ctx["label_id"]) == 15.0


def test_summary_endpoint_omits_the_grouping_keys_when_not_asked(ctx, client):
    _tag_all(ctx)
    body = client.get(
        f"/datasets/{ctx['dataset_id']}/quantification/summary",
        params={"include_appearance": False, "include_contextual": False,
                "include_relational": False},
    ).json()
    assert "groups" not in body and "group_by" not in body


def test_grouping_by_an_ungroupable_key_is_a_422(ctx, client):
    db = ctx["db"]
    meta.ensure_key(db, ctx["dataset_id"], "depth", MetadataValueType.NUMBER)
    db.commit()

    response = client.get(
        f"/datasets/{ctx['dataset_id']}/quantification/summary",
        params={"group_by": "depth"},
    )
    assert response.status_code == 422
    assert "depth" in response.json()["detail"]

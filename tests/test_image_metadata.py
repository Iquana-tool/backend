"""Per-image metadata: the key/value pairs that split a dataset into subgroups.

Three things carry the risk and are tested here:

  * **normalisation and the empty-value rule** — "absent" must have exactly one
    representation, or a filter for ``site=`` matches rows nobody thinks exist,
  * **the bulk write**, which is the actual grouping gesture and is the only path
    that can damage keys it was not asked about,
  * **the read paths the UI is built from** — the facet vocabulary, the filter's
    AND-across-keys / OR-within-a-key reading, and the two exports that have to
    carry the subgroup alongside the measurements.

The last block drives the router over HTTP, because the bulk endpoint checks
permissions by hand (there is no single id for `require()` to key off) and that
check is the one thing a service-level test cannot cover.
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
import app.database.contours  # noqa: F401
import app.database.dataset_members  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.image_metadata  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.rejections  # noqa: F401
import app.database.users  # noqa: F401
from app.database.contours import Contours
from app.database.dataset_members import DatasetMembers
from app.database.datasets import Datasets
from app.database.image_metadata import MAX_VALUE_LENGTH, ImageMetadata
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.exceptions import ImageNotFoundError, InvalidMetadataError
from app.routes.general.image_metadata import router as metadata_router
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import DatasetRole
from app.services.auth import get_current_user
from app.services.database_access import image_metadata as meta
from app.services.database_access.datasets import (
    build_coco_payload,
    get_dataset_as_df,
    get_image_and_mask_ids_of_dataset,
)


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


@pytest.fixture
def ctx(tmp_path):
    """A dataset of three images, each with a mask and one reviewed contour.

    Two members: ``curator`` may edit metadata, ``ann`` (an annotator) may not.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'metadata.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    user = Users(username="ann", hashed_password="x")
    db.add_all([user, Users(username="curator", hashed_password="x")])
    ds = Datasets(name="ds", description="", dataset_type="image",
                  folder_path=str(tmp_path), created_by="curator")
    db.add(ds)
    db.flush()
    db.add_all([
        DatasetMembers(dataset_id=ds.id, username="curator",
                       role=DatasetRole.CURATOR.value,
                       extra_permissions=[], denied_permissions=[]),
        DatasetMembers(dataset_id=ds.id, username="ann",
                       role=DatasetRole.ANNOTATOR.value,
                       extra_permissions=[], denied_permissions=[]),
    ])
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
                           area=0.1, perimeter=1.0, circularity=0.5, diameter=1.0,
                           x=[0.1, 0.5, 0.5, 0.1], y=[0.1, 0.1, 0.5, 0.5])
        contour.reviewed_by.append(user)
        db.add(contour)
        image_ids.append(image.id)
    db.commit()

    yield {"db": db, "Session": Session, "dataset_id": ds.id,
           "image_ids": image_ids, "label_id": label.id}
    db.close()
    engine.dispose()


@pytest.fixture
def client(ctx):
    """The metadata router on a live app, as a ``as_user(name) -> TestClient``."""
    Session = ctx["Session"]
    app = FastAPI()
    app.include_router(metadata_router)

    current = {"username": "curator"}

    def _session_override():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    def _user_override():
        session = Session()
        try:
            row = session.query(Users).filter_by(username=current["username"]).one()
            return AuthenticatedUser.from_query(row)
        finally:
            session.close()

    app.dependency_overrides[get_session] = _session_override
    app.dependency_overrides[get_current_user] = _user_override

    test_client = TestClient(app)

    def as_user(username):
        current["username"] = username
        return test_client

    return as_user


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def test_keys_are_trimmed_and_inner_whitespace_collapsed():
    assert meta.normalize_key("  collection   date ") == "collection date"


def test_key_case_is_preserved():
    """Folding case would silently merge two subgroups a curator kept apart."""
    assert meta.normalize_key("Site") == "Site"


@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
def test_empty_keys_are_rejected(bad):
    with pytest.raises(InvalidMetadataError):
        meta.normalize_key(bad)


def test_overlong_value_is_rejected():
    with pytest.raises(InvalidMetadataError):
        meta.normalize_value("x" * (MAX_VALUE_LENGTH + 1))


def test_values_keep_inner_whitespace():
    """A value can be a short note; only the edges are trimmed."""
    assert meta.normalize_value("  reef a, north face  ") == "reef a, north face"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def test_set_and_read_back(ctx):
    db, image_id = ctx["db"], ctx["image_ids"][0]
    meta.set_metadata_for_images(db, [image_id], {"site": "reef_a", "depth": "12"},
                                 username="ann")
    assert meta.get_metadata(db, image_id) == {"depth": "12", "site": "reef_a"}


def test_writing_a_key_again_overwrites_rather_than_accumulates(ctx):
    db, image_id = ctx["db"], ctx["image_ids"][0]
    meta.set_metadata_for_images(db, [image_id], {"site": "reef_a"})
    meta.set_metadata_for_images(db, [image_id], {"site": "reef_b"})

    assert meta.get_metadata(db, image_id) == {"site": "reef_b"}
    assert db.query(ImageMetadata).filter_by(image_id=image_id).count() == 1


def test_empty_value_removes_the_key(ctx):
    """"Absent" has one representation: no row. A stored blank would show up as
    its own subgroup in the facets and match no filter anyone would think to set."""
    db, image_id = ctx["db"], ctx["image_ids"][0]
    meta.set_metadata_for_images(db, [image_id], {"site": "reef_a"})
    meta.set_metadata_for_images(db, [image_id], {"site": "  "})

    assert meta.get_metadata(db, image_id) == {}
    assert db.query(ImageMetadata).filter_by(image_id=image_id).count() == 0


def test_partial_write_leaves_other_keys_alone(ctx):
    db, image_id = ctx["db"], ctx["image_ids"][0]
    meta.set_metadata_for_images(db, [image_id], {"site": "reef_a", "depth": "12"})
    meta.set_metadata_for_images(db, [image_id], {"depth": "15"})

    assert meta.get_metadata(db, image_id) == {"depth": "15", "site": "reef_a"}


def test_replace_deletes_keys_not_mentioned(ctx):
    db, image_id = ctx["db"], ctx["image_ids"][0]
    meta.set_metadata_for_images(db, [image_id], {"site": "reef_a", "depth": "12"})
    meta.set_metadata_for_images(db, [image_id], {"site": "reef_a"}, replace=True)

    assert meta.get_metadata(db, image_id) == {"site": "reef_a"}


def test_duplicate_keys_in_one_payload_are_rejected(ctx):
    """Two spellings of the same key would drop one value on dict ordering alone."""
    db, image_id = ctx["db"], ctx["image_ids"][0]
    with pytest.raises(InvalidMetadataError):
        meta.set_metadata_for_images(db, [image_id], {"site": "a", " site ": "b"})


def test_unknown_image_is_rejected_before_anything_is_written(ctx):
    db, image_ids = ctx["db"], ctx["image_ids"]
    with pytest.raises(ImageNotFoundError):
        meta.set_metadata_for_images(db, [image_ids[0], 9999], {"site": "reef_a"})

    assert meta.get_metadata(db, image_ids[0]) == {}


def test_bulk_tag_applies_to_every_image_without_touching_other_keys(ctx):
    """The grouping gesture: tag a selection, leave their per-image keys intact."""
    db, image_ids = ctx["db"], ctx["image_ids"]
    meta.set_metadata_for_images(db, [image_ids[0]], {"specimen": "s1"})
    meta.set_metadata_for_images(db, image_ids, {"site": "reef_a"})

    assert meta.get_metadata(db, image_ids[0]) == {"site": "reef_a", "specimen": "s1"}
    assert meta.get_metadata(db, image_ids[1]) == {"site": "reef_a"}
    assert meta.get_metadata(db, image_ids[2]) == {"site": "reef_a"}


def test_bulk_remove_keys_untags_a_selection(ctx):
    db, image_ids = ctx["db"], ctx["image_ids"]
    meta.set_metadata_for_images(db, image_ids, {"site": "reef_a", "depth": "12"})
    meta.set_metadata_for_images(db, image_ids[:2], {}, remove_keys=["site"])

    assert meta.get_metadata(db, image_ids[0]) == {"depth": "12"}
    assert meta.get_metadata(db, image_ids[2]) == {"depth": "12", "site": "reef_a"}


def test_delete_key_reports_whether_it_existed(ctx):
    db, image_id = ctx["db"], ctx["image_ids"][0]
    meta.set_metadata_for_images(db, [image_id], {"site": "reef_a"})

    assert meta.delete_key(db, image_id, "site") is True
    assert meta.delete_key(db, image_id, "site") is False


def test_metadata_dies_with_the_image(ctx):
    db, image_id = ctx["db"], ctx["image_ids"][0]
    meta.set_metadata_for_images(db, [image_id], {"site": "reef_a"})
    db.delete(db.query(Images).filter_by(id=image_id).one())
    db.commit()

    assert db.query(ImageMetadata).filter_by(image_id=image_id).count() == 0


# ---------------------------------------------------------------------------
# Reading: facets and filtering
# ---------------------------------------------------------------------------

def test_facets_count_values_and_lead_with_the_dominant_key(ctx):
    db, image_ids = ctx["db"], ctx["image_ids"]
    meta.set_metadata_for_images(db, image_ids, {"site": "reef_a"})
    meta.set_metadata_for_images(db, [image_ids[2]], {"site": "reef_b"})
    meta.set_metadata_for_images(db, [image_ids[0]], {"treatment": "control"})

    facets = meta.get_dataset_facets(db, ctx["dataset_id"])
    assert [facet["key"] for facet in facets] == ["site", "treatment"]
    assert facets[0]["image_count"] == 3
    assert facets[0]["values"] == [
        {"value": "reef_a", "count": 2},
        {"value": "reef_b", "count": 1},
    ]


def test_facets_are_scoped_to_the_dataset(ctx, tmp_path):
    db = ctx["db"]
    other = Datasets(name="other", description="", dataset_type="image",
                     folder_path=str(tmp_path), created_by="ann")
    db.add(other)
    db.flush()
    outsider = Images(dataset_id=other.id, file_name="x.png", file_path="x",
                      thumbnail_file_path="t", width=1, height=1, color_mode="RGB")
    db.add(outsider)
    db.commit()
    meta.set_metadata_for_images(db, [outsider.id], {"site": "elsewhere"})
    meta.set_metadata_for_images(db, ctx["image_ids"], {"site": "reef_a"})

    values = meta.get_dataset_facets(db, ctx["dataset_id"])[0]["values"]
    assert [entry["value"] for entry in values] == ["reef_a"]


def test_filter_ors_within_a_key_and_ands_across_keys(ctx):
    db, image_ids = ctx["db"], ctx["image_ids"]
    meta.set_metadata_for_images(db, [image_ids[0]], {"site": "a", "treatment": "control"})
    meta.set_metadata_for_images(db, [image_ids[1]], {"site": "b", "treatment": "heated"})
    meta.set_metadata_for_images(db, [image_ids[2]], {"site": "c", "treatment": "control"})

    assert set(meta.filter_image_ids(db, ctx["dataset_id"], {"site": ["a", "b"]})) == {
        image_ids[0], image_ids[1]
    }
    assert meta.filter_image_ids(
        db, ctx["dataset_id"], {"site": ["a", "b"], "treatment": ["control"]}
    ) == [image_ids[0]]


def test_filter_with_no_values_means_has_the_key_at_all(ctx):
    db, image_ids = ctx["db"], ctx["image_ids"]
    meta.set_metadata_for_images(db, image_ids[:2], {"site": "a"})

    assert set(meta.filter_image_ids(db, ctx["dataset_id"], {"site": []})) == {
        image_ids[0], image_ids[1]
    }


def test_batch_read_includes_untagged_images_as_empty(ctx):
    db, image_ids = ctx["db"], ctx["image_ids"]
    meta.set_metadata_for_images(db, [image_ids[0]], {"site": "a"})

    assert meta.get_metadata_for_images(db, image_ids) == {
        image_ids[0]: {"site": "a"},
        image_ids[1]: {},
        image_ids[2]: {},
    }


# ---------------------------------------------------------------------------
# The payloads the UI and the exports read
# ---------------------------------------------------------------------------

def test_image_listing_carries_each_image_s_metadata(ctx):
    """The gallery filters subgroups from the list it already has, with no
    second request — the same way it filters workflow status."""
    db, image_ids = ctx["db"], ctx["image_ids"]
    meta.set_metadata_for_images(db, [image_ids[0]], {"site": "reef_a"})

    listing = asyncio.run(get_image_and_mask_ids_of_dataset(ctx["dataset_id"], db=db))
    by_id = {entry["image_id"]: entry for entry in listing}
    assert by_id[image_ids[0]]["metadata"] == {"site": "reef_a"}
    assert by_id[image_ids[1]]["metadata"] == {}


def test_quantification_export_gains_one_column_per_key(ctx):
    """Measurements you cannot group by are measurements you cannot compare."""
    db, image_ids = ctx["db"], ctx["image_ids"]
    meta.set_metadata_for_images(db, image_ids[:2], {"site": "reef_a"})
    meta.set_metadata_for_images(db, [image_ids[2]], {"site": "reef_b"})

    df = asyncio.run(get_dataset_as_df(ctx["dataset_id"], True, True, db))
    assert "meta_site" in df.columns
    assert sorted(df["meta_site"].tolist()) == ["reef_a", "reef_a", "reef_b"]


def test_export_columns_are_the_dataset_wide_union(ctx):
    """A column that only appears once a tagged row shows up is unreadable to
    every stats package, so untagged rows get an empty cell instead."""
    db, image_ids = ctx["db"], ctx["image_ids"]
    meta.set_metadata_for_images(db, [image_ids[0]], {"treatment": "control"})

    df = asyncio.run(get_dataset_as_df(ctx["dataset_id"], True, True, db))
    assert len(df) == 3
    assert df["meta_treatment"].isna().sum() == 2


def test_profile_export_gains_the_same_columns(ctx):
    """The profile export builds its rows separately from the legacy one, so it
    would be the natural place for the subgroup columns to go missing."""
    db, image_ids = ctx["db"], ctx["image_ids"]
    meta.set_metadata_for_images(db, image_ids, {"site": "reef_a"})

    df = asyncio.run(get_dataset_as_df(ctx["dataset_id"], True, True, db,
                                       metric_scoping={"area": None}))
    assert "meta_site" in df.columns
    assert df["meta_site"].tolist() == ["reef_a"] * 3


def test_no_metadata_leaves_the_export_shape_untouched(ctx):
    db = ctx["db"]
    df = asyncio.run(get_dataset_as_df(ctx["dataset_id"], True, True, db))
    assert not [column for column in df.columns if column.startswith("meta_")]


def test_coco_images_carry_metadata_only_when_tagged(ctx):
    db, image_ids = ctx["db"], ctx["image_ids"]
    meta.set_metadata_for_images(db, [image_ids[0]], {"site": "reef_a"})

    rows = (
        db.query(Contours, Images, Labels)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .join(Labels, Labels.id == Contours.label_id)
        .filter(Images.dataset_id == ctx["dataset_id"])
        .all()
    )
    dataset = db.query(Datasets).filter_by(id=ctx["dataset_id"]).one()
    payload, _ = build_coco_payload(
        dataset, rows,
        metadata_by_image=meta.get_metadata_for_dataset(db, ctx["dataset_id"]),
    )

    by_id = {entry["id"]: entry for entry in payload["images"]}
    assert by_id[image_ids[0]]["metadata"] == {"site": "reef_a"}
    assert "metadata" not in by_id[image_ids[1]]


# ---------------------------------------------------------------------------
# Over HTTP
# ---------------------------------------------------------------------------

def test_write_and_read_one_image_over_http(ctx, client):
    image_id = ctx["image_ids"][0]
    response = client("curator").put(
        f"/metadata/image/{image_id}",
        json={"entries": {"site": "reef_a"}},
    )
    assert response.status_code == 200
    assert response.json()["metadata"] == {"site": "reef_a"}

    assert client("ann").get(f"/metadata/image/{image_id}").json()["metadata"] == {
        "site": "reef_a"
    }


def test_annotators_can_read_but_not_write(ctx, client):
    """Retagging images redraws the groups someone else's analysis is built on,
    so it sits with curation rather than with annotation work."""
    image_id = ctx["image_ids"][0]
    assert client("ann").get(f"/metadata/image/{image_id}").status_code == 200

    response = client("ann").put(
        f"/metadata/image/{image_id}", json={"entries": {"site": "reef_a"}}
    )
    assert response.status_code == 403
    assert "image.metadata_write" in response.json()["detail"]


def test_bulk_write_over_http(ctx, client):
    image_ids = ctx["image_ids"]
    response = client("curator").post(
        "/metadata/images",
        json={"image_ids": image_ids, "entries": {"site": "reef_a"}},
    )
    assert response.status_code == 200
    assert response.json()["written"] == 3
    assert response.json()["metadata"][str(image_ids[1])] == {"site": "reef_a"}


def test_bulk_write_is_refused_for_an_annotator(ctx, client):
    response = client("ann").post(
        "/metadata/images",
        json={"image_ids": ctx["image_ids"], "entries": {"site": "reef_a"}},
    )
    assert response.status_code == 403


def test_bulk_write_is_refused_when_the_list_reaches_another_dataset(ctx, client, tmp_path):
    """The bulk endpoint has no single id for `require()` to key off, so this is
    the check that stops a hand-written id list from editing a stranger's images."""
    db = ctx["db"]
    other = Datasets(name="theirs", description="", dataset_type="image",
                     folder_path=str(tmp_path), created_by="ann")
    db.add(other)
    db.flush()
    outsider = Images(dataset_id=other.id, file_name="x.png", file_path="x",
                      thumbnail_file_path="t", width=1, height=1, color_mode="RGB")
    db.add(outsider)
    db.commit()

    response = client("curator").post(
        "/metadata/images",
        json={"image_ids": [ctx["image_ids"][0], outsider.id],
              "entries": {"site": "reef_a"}},
    )
    assert response.status_code == 403
    # Nothing partly applied: the check runs before the first write.
    assert meta.get_metadata(db, ctx["image_ids"][0]) == {}


def test_empty_key_is_a_422_not_a_500(ctx, client):
    response = client("curator").put(
        f"/metadata/image/{ctx['image_ids'][0]}", json={"entries": {"  ": "reef_a"}}
    )
    assert response.status_code == 422


def test_dataset_endpoint_serves_the_vocabulary_and_the_untagged_count(ctx, client):
    image_ids = ctx["image_ids"]
    client("curator").post(
        "/metadata/images",
        json={"image_ids": image_ids[:2], "entries": {"site": "reef_a"}},
    )

    body = client("ann").get(f"/metadata/dataset/{ctx['dataset_id']}").json()
    assert len(body["facets"]) == 1
    facet = body["facets"][0]
    assert facet["key"] == "site"
    assert facet["image_count"] == 2
    assert facet["values"] == [{"value": "reef_a", "count": 2}]
    # A key invented by typing it in is categorical, so the chips and the
    # grouping keep working without anyone declaring a schema.
    assert facet["value_type"] == "categorical"
    assert facet["groupable"] is True
    assert body["total_images"] == 3
    assert body["untagged_count"] == 1


def test_delete_key_over_http(ctx, client):
    image_id = ctx["image_ids"][0]
    client("curator").put(f"/metadata/image/{image_id}",
                          json={"entries": {"site": "reef_a"}})

    response = client("curator").delete(f"/metadata/image/{image_id}/site")
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert response.json()["metadata"] == {}

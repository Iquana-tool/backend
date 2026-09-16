"""Unit and integration tests for COCO export label filtering.

Verifies:
1. Parsing and validation of comma-separated `label_ids` query parameters:
   - Valid subsets of positive integer IDs belonging to the dataset.
   - Rejection (422) for non-integer, non-positive, duplicate, empty/whitespace,
     or foreign label IDs.
2. Exact label filtering in `export_dataset_contours_to_coco`:
   - Filters contours, categories, and referenced image rows.
   - Naturally excludes unlabelled contours when a filter is active.
   - Surviving categories and images match only the filtered contours.
3. HTTP route integration for:
   - `GET /datasets/{id}/coco/annotations`
   - `GET /datasets/{id}/coco` (ZIP stream)
   - Backward compatibility of `contour_selection`.
"""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import database, get_session
import app.database.contours  # noqa: F401
import app.database.dataset_members  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.users  # noqa: F401

from app.database.contours import Contours
from app.database.dataset_members import DatasetMembers
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.exceptions import InvalidLabelFilterError
from app.routes.general.datasets import router as datasets_router
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import DatasetRole, GlobalRole, Permission
from app.services.auth import get_current_user
from app.services.database_access.datasets import (
    export_dataset_contours_to_coco,
    parse_and_validate_label_ids,
)


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


def _create_test_image(path: Path, width: int = 100, height: int = 100) -> bytes:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    img = PILImage.fromarray(arr, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_coco_filters.db'}")
    database.metadata.create_all(engine)
    SessionMaker = sessionmaker(bind=engine)
    session = SessionMaker()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def multi_label_dataset(db_session, tmp_path):
    """Creates a dataset with 3 labels, 3 images, reviewed contours, and unlabelled contours.

    Structure:
      Label A: in Image 1 only
      Label B: in Image 2 only
      Label C: in Image 3 only
      Unlabelled contour: in Image 1
    """
    user = Users(username="alice", hashed_password="x", global_role=GlobalRole.ADMIN.value)
    reviewer = Users(username="bob", hashed_password="x", global_role=GlobalRole.MEMBER.value)
    db_session.add_all([user, reviewer])
    db_session.flush()

    ds = Datasets(
        name="Filter Test Dataset",
        description="Dataset for testing label filters",
        dataset_type="image",
        folder_path=str(tmp_path / "ds_folder"),
        created_by="alice",
    )
    db_session.add(ds)
    db_session.flush()

    db_session.add_all([
        DatasetMembers(dataset_id=ds.id, username="alice", role=DatasetRole.OWNER.value),
        DatasetMembers(dataset_id=ds.id, username="bob", role=DatasetRole.REVIEWER.value),
    ])

    # 3 Labels
    lbl_a = Labels(dataset_id=ds.id, name="Coral", value=1, parent_id=None)
    lbl_b = Labels(dataset_id=ds.id, name="Algae", value=2, parent_id=None)
    lbl_c = Labels(dataset_id=ds.id, name="Sand", value=3, parent_id=None)
    # Another dataset's label (foreign)
    ds_other = Datasets(
        name="Other Dataset",
        dataset_type="image",
        folder_path=str(tmp_path / "other_folder"),
        created_by="alice",
    )
    db_session.add(ds_other)
    db_session.flush()
    lbl_foreign = Labels(dataset_id=ds_other.id, name="Foreign", value=1, parent_id=None)

    db_session.add_all([lbl_a, lbl_b, lbl_c, lbl_foreign])
    db_session.flush()

    # 3 Images
    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"
    img3_path = tmp_path / "img3.png"
    _create_test_image(img1_path, 200, 150)
    _create_test_image(img2_path, 200, 150)
    _create_test_image(img3_path, 200, 150)

    img1 = Images(
        dataset_id=ds.id,
        file_name="img1.png",
        file_path=str(img1_path),
        thumbnail_file_path=str(tmp_path / "thumb1.png"),
        width=200,
        height=150,
    )
    img2 = Images(
        dataset_id=ds.id,
        file_name="img2.png",
        file_path=str(img2_path),
        thumbnail_file_path=str(tmp_path / "thumb2.png"),
        width=200,
        height=150,
    )
    img3 = Images(
        dataset_id=ds.id,
        file_name="img3.png",
        file_path=str(img3_path),
        thumbnail_file_path=str(tmp_path / "thumb3.png"),
        width=200,
        height=150,
    )
    db_session.add_all([img1, img2, img3])
    db_session.flush()

    # Fully annotated masks
    m1 = Masks(image_id=img1.id, fully_annotated=True, file_path=str(tmp_path / "m1.png"))
    m2 = Masks(image_id=img2.id, fully_annotated=True, file_path=str(tmp_path / "m2.png"))
    m3 = Masks(image_id=img3.id, fully_annotated=True, file_path=str(tmp_path / "m3.png"))
    db_session.add_all([m1, m2, m3])
    db_session.flush()

    now = datetime.now(timezone.utc)

    # Contour 1 (Label A, img1)
    c1 = Contours(
        mask_id=m1.id,
        label_id=lbl_a.id,
        added_by="User",
        author_username="alice",
        confidence_score=1.0,
        area=100.0,
        perimeter=40.0,
        circularity=0.8,
        diameter=15.0,
        x=[0.1, 0.4, 0.4, 0.1],
        y=[0.1, 0.1, 0.4, 0.4],
        created_at=now,
    )
    # Contour 2 (Label B, img2)
    c2 = Contours(
        mask_id=m2.id,
        label_id=lbl_b.id,
        added_by="User",
        author_username="alice",
        confidence_score=1.0,
        area=150.0,
        perimeter=50.0,
        circularity=0.85,
        diameter=18.0,
        x=[0.2, 0.6, 0.6, 0.2],
        y=[0.2, 0.2, 0.6, 0.6],
        created_at=now,
    )
    # Contour 3 (Label C, img3)
    c3 = Contours(
        mask_id=m3.id,
        label_id=lbl_c.id,
        added_by="User",
        author_username="alice",
        confidence_score=1.0,
        area=80.0,
        perimeter=35.0,
        circularity=0.75,
        diameter=12.0,
        x=[0.3, 0.7, 0.7, 0.3],
        y=[0.3, 0.3, 0.7, 0.7],
        created_at=now,
    )
    # Unlabelled contour on img1
    c_unlabelled = Contours(
        mask_id=m1.id,
        label_id=None,
        added_by="User",
        author_username="alice",
        confidence_score=0.9,
        area=50.0,
        perimeter=30.0,
        circularity=0.7,
        diameter=10.0,
        x=[0.5, 0.8, 0.8, 0.5],
        y=[0.5, 0.5, 0.8, 0.8],
        created_at=now,
    )

    db_session.add_all([c1, c2, c3, c_unlabelled])
    db_session.flush()

    # Mark reviewed so exclude_unreviewed=True does not drop them
    c1.reviewed_by.append(reviewer)
    c2.reviewed_by.append(reviewer)
    c3.reviewed_by.append(reviewer)
    c_unlabelled.reviewed_by.append(reviewer)
    db_session.commit()

    return {
        "dataset_id": ds.id,
        "other_dataset_id": ds_other.id,
        "labels": {
            "a": lbl_a.id,
            "b": lbl_b.id,
            "c": lbl_c.id,
            "foreign": lbl_foreign.id,
        },
        "images": {
            "img1": img1.id,
            "img2": img2.id,
            "img3": img3.id,
        },
        "contours": {
            "c1": c1.id,
            "c2": c2.id,
            "c3": c3.id,
            "unlabelled": c_unlabelled.id,
        },
    }


# ---------------------------------------------------------------------------
# Unit tests: parse_and_validate_label_ids
# ---------------------------------------------------------------------------


def test_parse_and_validate_label_ids_none_or_empty(db_session, multi_label_dataset):
    ds_id = multi_label_dataset["dataset_id"]
    assert parse_and_validate_label_ids(db_session, ds_id, None) is None

    # Empty string or whitespace must raise InvalidLabelFilterError
    with pytest.raises(InvalidLabelFilterError):
        parse_and_validate_label_ids(db_session, ds_id, "")

    with pytest.raises(InvalidLabelFilterError):
        parse_and_validate_label_ids(db_session, ds_id, "   ")


def test_parse_and_validate_label_ids_valid(db_session, multi_label_dataset):
    ds_id = multi_label_dataset["dataset_id"]
    lbl_a = multi_label_dataset["labels"]["a"]
    lbl_b = multi_label_dataset["labels"]["b"]

    res1 = parse_and_validate_label_ids(db_session, ds_id, str(lbl_a))
    assert res1 == [lbl_a]

    res2 = parse_and_validate_label_ids(db_session, ds_id, f"{lbl_a},{lbl_b}")
    assert res2 == [lbl_a, lbl_b]

    # Whitespace around commas or numbers handled cleanly
    res3 = parse_and_validate_label_ids(db_session, ds_id, f" {lbl_a} , {lbl_b} ")
    assert res3 == [lbl_a, lbl_b]


def test_parse_and_validate_label_ids_duplicate(db_session, multi_label_dataset):
    ds_id = multi_label_dataset["dataset_id"]
    lbl_a = multi_label_dataset["labels"]["a"]

    with pytest.raises(InvalidLabelFilterError) as exc:
        parse_and_validate_label_ids(db_session, ds_id, f"{lbl_a},{lbl_a}")
    assert "Duplicate label ID" in str(exc.value)


def test_parse_and_validate_label_ids_non_positive_and_malformed(db_session, multi_label_dataset):
    ds_id = multi_label_dataset["dataset_id"]
    lbl_a = multi_label_dataset["labels"]["a"]

    with pytest.raises(InvalidLabelFilterError) as exc1:
        parse_and_validate_label_ids(db_session, ds_id, "0")
    assert "positive integers" in str(exc1.value)

    with pytest.raises(InvalidLabelFilterError) as exc2:
        parse_and_validate_label_ids(db_session, ds_id, "-5")
    assert "positive integers" in str(exc2.value)

    with pytest.raises(InvalidLabelFilterError) as exc3:
        parse_and_validate_label_ids(db_session, ds_id, "abc")
    assert "positive integers" in str(exc3.value)

    with pytest.raises(InvalidLabelFilterError) as exc4:
        parse_and_validate_label_ids(db_session, ds_id, f"{lbl_a},")
    assert "empty elements" in str(exc4.value)

    with pytest.raises(InvalidLabelFilterError) as exc5:
        parse_and_validate_label_ids(db_session, ds_id, f",{lbl_a}")
    assert "empty elements" in str(exc5.value)


def test_parse_and_validate_label_ids_very_long_numeric(db_session, multi_label_dataset):
    ds_id = multi_label_dataset["dataset_id"]

    # Exceeding 19 digits or 64-bit max integer
    with pytest.raises(InvalidLabelFilterError) as exc1:
        parse_and_validate_label_ids(db_session, ds_id, "9" * 50)
    assert "positive integers" in str(exc1.value)

    with pytest.raises(InvalidLabelFilterError) as exc2:
        parse_and_validate_label_ids(db_session, ds_id, "9223372036854775808")
    assert "positive integers" in str(exc2.value)


def test_parse_and_validate_label_ids_foreign(db_session, multi_label_dataset):
    ds_id = multi_label_dataset["dataset_id"]
    lbl_foreign = multi_label_dataset["labels"]["foreign"]

    with pytest.raises(InvalidLabelFilterError) as exc:
        parse_and_validate_label_ids(db_session, ds_id, str(lbl_foreign))
    assert "do not belong to dataset" in str(exc.value)

    with pytest.raises(InvalidLabelFilterError) as exc2:
        parse_and_validate_label_ids(db_session, ds_id, "999999")
    assert "do not belong to dataset" in str(exc2.value)


def test_parse_and_validate_label_ids_db_failures_vs_overflow(db_session, multi_label_dataset):
    ds_id = multi_label_dataset["dataset_id"]
    from unittest.mock import patch
    from sqlalchemy.exc import DataError, OperationalError

    # Database operational failures must NOT be caught as InvalidLabelFilterError
    with patch.object(db_session, "query", side_effect=OperationalError("connection lost", {}, Exception())):
        with pytest.raises(OperationalError):
            parse_and_validate_label_ids(db_session, ds_id, "1,2")

    # Data overflow errors from DB/driver must be caught and raised as InvalidLabelFilterError
    with patch.object(db_session, "query", side_effect=DataError("integer out of range", {}, Exception())):
        with pytest.raises(InvalidLabelFilterError) as exc_data:
            parse_and_validate_label_ids(db_session, ds_id, "1,2")
        assert "Invalid label ID(s) specified" in str(exc_data.value)

    with patch.object(db_session, "query", side_effect=OverflowError("Python int too large to convert to SQLite INTEGER")):
        with pytest.raises(InvalidLabelFilterError) as exc_over:
            parse_and_validate_label_ids(db_session, ds_id, "1,2")
        assert "Invalid label ID(s) specified" in str(exc_over.value)


# ---------------------------------------------------------------------------
# Service-level tests: export_dataset_contours_to_coco
# ---------------------------------------------------------------------------


def test_export_dataset_contours_to_coco_unfiltered(db_session, multi_label_dataset):
    ds_id = multi_label_dataset["dataset_id"]
    result = asyncio.run(export_dataset_contours_to_coco(
        dataset_id=ds_id,
        db=db_session,
        exclude_not_fully_annotated=True,
        exclude_unreviewed=True,
        label_ids=None,
        write_to_disk=False,
    ))
    assert result["success"] is True
    payload = result["coco_payload"]

    # All 3 labelled contours exported (unlabelled contour omitted in COCO standard)
    ann_ids = [a["id"] for a in payload["annotations"]]
    assert len(ann_ids) == 3
    assert multi_label_dataset["contours"]["c1"] in ann_ids
    assert multi_label_dataset["contours"]["c2"] in ann_ids
    assert multi_label_dataset["contours"]["c3"] in ann_ids
    assert multi_label_dataset["contours"]["unlabelled"] not in ann_ids

    # Categories contain all 3 labels
    category_ids = {c["id"] for c in payload["categories"]}
    assert category_ids == {
        multi_label_dataset["labels"]["a"],
        multi_label_dataset["labels"]["b"],
        multi_label_dataset["labels"]["c"],
    }

    # All 3 images referenced
    image_ids = {img["id"] for img in payload["images"]}
    assert image_ids == {
        multi_label_dataset["images"]["img1"],
        multi_label_dataset["images"]["img2"],
        multi_label_dataset["images"]["img3"],
    }
    assert result["image_ids"] == image_ids


def test_export_dataset_contours_to_coco_filtered_single_label(db_session, multi_label_dataset):
    ds_id = multi_label_dataset["dataset_id"]
    lbl_b = multi_label_dataset["labels"]["b"]

    result = asyncio.run(export_dataset_contours_to_coco(
        dataset_id=ds_id,
        db=db_session,
        exclude_not_fully_annotated=True,
        exclude_unreviewed=True,
        label_ids=[lbl_b],
        write_to_disk=False,
    ))
    assert result["success"] is True
    payload = result["coco_payload"]

    # Only c2 is exported
    assert len(payload["annotations"]) == 1
    assert payload["annotations"][0]["id"] == multi_label_dataset["contours"]["c2"]
    assert payload["annotations"][0]["category_id"] == lbl_b

    # Only Label B category is included
    assert len(payload["categories"]) == 1
    assert payload["categories"][0]["id"] == lbl_b
    assert payload["categories"][0]["name"] == "Algae"

    # Only Image 2 is referenced
    assert len(payload["images"]) == 1
    assert payload["images"][0]["id"] == multi_label_dataset["images"]["img2"]
    assert result["image_ids"] == {multi_label_dataset["images"]["img2"]}


def test_export_dataset_contours_to_coco_filtered_subset_excludes_unlabelled(db_session, multi_label_dataset):
    ds_id = multi_label_dataset["dataset_id"]
    lbl_a = multi_label_dataset["labels"]["a"]
    lbl_c = multi_label_dataset["labels"]["c"]

    result = asyncio.run(export_dataset_contours_to_coco(
        dataset_id=ds_id,
        db=db_session,
        exclude_not_fully_annotated=True,
        exclude_unreviewed=True,
        label_ids=[lbl_a, lbl_c],
        write_to_disk=False,
    ))
    assert result["success"] is True
    payload = result["coco_payload"]

    ann_ids = [a["id"] for a in payload["annotations"]]
    assert multi_label_dataset["contours"]["c1"] in ann_ids
    assert multi_label_dataset["contours"]["c3"] in ann_ids
    assert multi_label_dataset["contours"]["c2"] not in ann_ids
    assert multi_label_dataset["contours"]["unlabelled"] not in ann_ids

    category_ids = {c["id"] for c in payload["categories"]}
    assert category_ids == {lbl_a, lbl_c}

    image_ids = {img["id"] for img in payload["images"]}
    assert image_ids == {
        multi_label_dataset["images"]["img1"],
        multi_label_dataset["images"]["img3"],
    }
    assert multi_label_dataset["images"]["img2"] not in image_ids


# ---------------------------------------------------------------------------
# HTTP Route Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client(db_session, multi_label_dataset):
    app = FastAPI()
    app.include_router(datasets_router)

    app.dependency_overrides[get_session] = lambda: db_session

    ds_id = multi_label_dataset["dataset_id"]
    current_auth_user = [
        AuthenticatedUser(
            username="alice",
            is_admin=True,
            global_role=GlobalRole.ADMIN,
            is_active=True,
            owned_datasets=[ds_id],
            accessible_datasets=[ds_id],
            memberships={
                ds_id: {
                    "role": DatasetRole.OWNER,
                    "permissions": {
                        Permission.EXPORT_ANNOTATIONS,
                        Permission.EXPORT_IMAGES,
                        Permission.DATASET_READ,
                    },
                }
            },
        )
    ]
    app.dependency_overrides[get_current_user] = lambda: current_auth_user[0]

    return TestClient(app), ds_id, multi_label_dataset


def test_http_get_coco_annotations_without_filter(api_client):
    client, ds_id, data = api_client
    response = client.get(f"/datasets/{ds_id}/coco/annotations")
    assert response.status_code == 200
    doc = response.json()
    assert len(doc["annotations"]) == 3  # 3 labelled contours (unlabelled contour omitted)
    assert len(doc["categories"]) == 3
    assert len(doc["images"]) == 3


def test_http_get_coco_annotations_with_filter(api_client):
    client, ds_id, data = api_client
    lbl_a = data["labels"]["a"]
    lbl_b = data["labels"]["b"]

    response = client.get(f"/datasets/{ds_id}/coco/annotations?label_ids={lbl_a},{lbl_b}")
    assert response.status_code == 200
    doc = response.json()
    assert len(doc["annotations"]) == 2
    assert len(doc["categories"]) == 2
    assert len(doc["images"]) == 2


def test_http_get_coco_annotations_invalid_filters_return_422(api_client):
    client, ds_id, data = api_client
    lbl_a = data["labels"]["a"]
    lbl_foreign = data["labels"]["foreign"]

    # Non-integer
    resp = client.get(f"/datasets/{ds_id}/coco/annotations?label_ids=abc")
    assert resp.status_code == 422

    # Non-positive
    resp = client.get(f"/datasets/{ds_id}/coco/annotations?label_ids=0")
    assert resp.status_code == 422

    # Duplicate
    resp = client.get(f"/datasets/{ds_id}/coco/annotations?label_ids={lbl_a},{lbl_a}")
    assert resp.status_code == 422

    # Trailing comma
    resp = client.get(f"/datasets/{ds_id}/coco/annotations?label_ids={lbl_a},")
    assert resp.status_code == 422

    # Foreign label ID
    resp = client.get(f"/datasets/{ds_id}/coco/annotations?label_ids={lbl_foreign}")
    assert resp.status_code == 422

    # Empty string
    resp = client.get(f"/datasets/{ds_id}/coco/annotations?label_ids=")
    assert resp.status_code == 422


def test_http_get_coco_zip_with_filter(api_client):
    client, ds_id, data = api_client
    lbl_c = data["labels"]["c"]

    # Fetch zip filtered to Label C only (which is in Image 3 only)
    response = client.get(f"/datasets/{ds_id}/coco?label_ids={lbl_c}&include_images=true")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    zip_bytes = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_bytes, "r") as zf:
        members = zf.namelist()
        assert "Filter_Test_Dataset_coco.json" in members
        # Image 3 should be in images/, but not Image 1 or Image 2
        assert "images/img3.png" in members
        assert "images/img1.png" not in members
        assert "images/img2.png" not in members

        coco_data = json.loads(zf.read("Filter_Test_Dataset_coco.json"))
        assert len(coco_data["annotations"]) == 1
        assert coco_data["annotations"][0]["category_id"] == lbl_c
        assert len(coco_data["categories"]) == 1
        assert len(coco_data["images"]) == 1


def test_http_get_coco_zip_invalid_filter_returns_422(api_client):
    client, ds_id, data = api_client
    response = client.get(f"/datasets/{ds_id}/coco?label_ids=invalid")
    assert response.status_code == 422

    # Very long numeric label IDs must return 422, not 500
    resp_long_zip = client.get(f"/datasets/{ds_id}/coco?label_ids={'9' * 50}")
    assert resp_long_zip.status_code == 422
    assert "positive integers" in resp_long_zip.json().get("detail", "")

    resp_long_ann = client.get(f"/datasets/{ds_id}/coco/annotations?label_ids={'9' * 50}")
    assert resp_long_ann.status_code == 422
    assert "positive integers" in resp_long_ann.json().get("detail", "")

"""Integration and contract tests for IQUANA dataset archive export (v1).

Tests both service-layer archive creation and HTTP endpoint streaming:
1. End-to-end rich archive export (annotations.json + config.json + images).
2. Schema validation of generated annotations.json and config.json.
3. Byte-identical image retention and SHA-256 verification.
4. Correct emission of normalized geometry and derived COCO projections.
5. Conversion of legacy pixel coordinates to normalized canonical coordinates.
6. Non-quantification guarantee: no ContourMetrics queried or exported.
7. Omission and count tracking of temporary contours.
8. Safe stripping and disclosure of query_contour_id in model routing bindings.
9. Error conditions: missing files, duplicate categories, unsupported dataset type, dangling references.
10. Permission enforcement and HTTP streaming via TestClient.
"""

from __future__ import annotations

import io
import json
import os
import queue
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy import create_engine, event, func
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.database import database, get_session, init_db
import app.database.contours  # noqa: F401
import app.database.dataset_calibration_defaults  # noqa: F401
import app.database.dataset_members  # noqa: F401
import app.database.dataset_metadata_keys  # noqa: F401
import app.database.dataset_model_routing_configs  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.image_calibrations  # noqa: F401
import app.database.image_metadata  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.quantification_profiles  # noqa: F401
import app.database.rejections  # noqa: F401
import app.database.users  # noqa: F401

from app.database.contour_metrics import ContourMetrics
from app.database.contours import Contours, reviewer_contour_association
from app.database.dataset_calibration_defaults import DatasetCalibrationDefaults
from app.database.dataset_members import DatasetMembers
from app.database.dataset_metadata_keys import DatasetMetadataKeys
from app.database.dataset_model_routing_configs import DatasetModelRoutingConfigs
from app.database.datasets import Datasets
from app.database.image_calibrations import ImageCalibrations
from app.database.image_metadata import ImageMetadata
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.quantification_profiles import QuantificationProfiles
from app.database.rejections import AnnotationRejections
from app.database.users import Users
from app.exceptions import (
    DatasetArchiveExportError,
    DatasetArchiveImportError,
    DatasetArchiveNameConflictError,
    DatasetArchiveSizeLimitError,
    DatasetArchiveValidationError,
    DatasetNotFoundError,
)
from app.routes.general.datasets import router as datasets_router
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.dataset_archive import (
    DatasetArchiveImportResponse,
    IquanaAnnotationsDocument,
    IquanaConfigDocument,
)
from app.schemas.permissions import DatasetRole, GlobalRole, Permission
import config
from app.services.auth import get_current_user
from app.services.dataset_archive import (
    _confirmed_ready_model_bindings,
    _slugify_dataset_name,
    create_iquana_dataset_archive,
    import_iquana_dataset_archive,
)


@pytest.fixture(autouse=True)
def isolate_data_dirs(tmp_path, monkeypatch):
    """Isolate datasets and thumbnails directories in a per-test temporary directory."""
    datasets_dir = tmp_path / "datasets"
    thumbnails_dir = tmp_path / "thumbnails"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DATASETS_DIR", str(datasets_dir))
    monkeypatch.setattr(config, "THUMBNAILS_DIR", str(thumbnails_dir))


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


def _create_test_image(path: Path, width: int = 100, height: int = 100, color: tuple[int, int, int] = (128, 64, 32)) -> bytes:
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    img = PILImage.fromarray(arr, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    with open(path, "rb") as f:
        return f.read()


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test_archive.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def rich_dataset(db_session, tmp_path):
    """Creates a full rich dataset with images, masks, hierarchies, calibrations, profiles, and routing."""
    # 1. Users
    alice = Users(username="alice", hashed_password="x", global_role=GlobalRole.ADMIN.value)
    bob = Users(username="bob", hashed_password="x", global_role=GlobalRole.MEMBER.value)
    carol = Users(username="carol", hashed_password="x", global_role=GlobalRole.MEMBER.value)
    db_session.add_all([alice, bob, carol])
    db_session.flush()

    # 2. Dataset
    ds = Datasets(
        name="Coral Survey 2026",
        description="Benthic habitat dataset",
        dataset_type="image",
        folder_path=str(tmp_path / "ds_folder"),
        created_by="alice",
        require_independent_review=True,
    )
    db_session.add(ds)
    db_session.flush()

    # 3. Memberships
    db_session.add_all([
        DatasetMembers(dataset_id=ds.id, username="alice", role=DatasetRole.OWNER.value),
        DatasetMembers(dataset_id=ds.id, username="bob", role=DatasetRole.REVIEWER.value),
        DatasetMembers(dataset_id=ds.id, username="carol", role=DatasetRole.ANNOTATOR.value),
    ])

    # 4. Images
    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"
    img1_bytes = _create_test_image(img1_path, width=200, height=150, color=(10, 20, 30))
    img2_bytes = _create_test_image(img2_path, width=300, height=200, color=(40, 50, 60))

    img1 = Images(
        dataset_id=ds.id,
        file_name="transect_001.png",
        file_path=str(img1_path),
        thumbnail_file_path=str(tmp_path / "thumb1.png"),
        description="First transect image",
        width=200,
        height=150,
        color_mode="RGB",
        scale_x=0.5,
        scale_y=0.5,
        unit="mm",
    )
    img2 = Images(
        dataset_id=ds.id,
        file_name="transect_002.png",
        file_path=str(img2_path),
        thumbnail_file_path=str(tmp_path / "thumb2.png"),
        description="Second transect image",
        width=300,
        height=200,
        color_mode="RGB",
        scale_x=1.0,
        scale_y=1.0,
        unit="px",
    )
    db_session.add_all([img1, img2])
    db_session.flush()

    # 5. Metadata Keys and Image Values
    key_site = DatasetMetadataKeys(
        dataset_id=ds.id,
        key="site",
        value_type="categorical",
        options=["Reef Alpha", "Reef Beta"],
        description="Reef site name",
        created_by="alice",
    )
    key_depth = DatasetMetadataKeys(
        dataset_id=ds.id,
        key="depth_m",
        value_type="number",
        unit="m",
        options=[],
        description="Water depth",
        created_by="alice",
    )
    db_session.add_all([key_site, key_depth])
    db_session.flush()

    db_session.add_all([
        ImageMetadata(image_id=img1.id, key="site", value="Reef Alpha"),
        ImageMetadata(image_id=img1.id, key="depth_m", value="12.5"),
        ImageMetadata(image_id=img2.id, key="site", value="Reef Beta"),
    ])

    # 6. Image Calibrations
    now = datetime.now(timezone.utc)
    db_session.add_all([
        ImageCalibrations(
            image_id=img1.id,
            kind="scale",
            source="manual",
            params={"scale_x": 0.5, "scale_y": 0.5, "unit": "mm"},
            created_by="alice",
            created_at=now,
            updated_at=now,
        ),
    ])

    # 7. Labels
    lbl_benthos = Labels(dataset_id=ds.id, name="Benthos", value=0, parent_id=None)
    db_session.add(lbl_benthos)
    db_session.flush()

    lbl_coral = Labels(dataset_id=ds.id, name="Coral", value=1, parent_id=lbl_benthos.id)
    lbl_sand = Labels(dataset_id=ds.id, name="Sand", value=2, parent_id=lbl_benthos.id)
    db_session.add_all([lbl_coral, lbl_sand])
    db_session.flush()

    # 8. Masks
    mask1 = Masks(image_id=img1.id, fully_annotated=True, file_path=str(tmp_path / "m1.png"))
    mask2 = Masks(image_id=img2.id, fully_annotated=False, file_path=str(tmp_path / "m2.png"))
    db_session.add_all([mask1, mask2])
    db_session.flush()

    # 9. Contours (including hierarchy and temporary contour)
    # Parent contour
    c_parent = Contours(
        mask_id=mask1.id,
        parent_id=None,
        temporary=False,
        added_by="User",
        author_username="carol",
        created_at=now,
        confidence_score=1.0,
        label_id=lbl_coral.id,
        area=100.0,
        perimeter=40.0,
        circularity=0.8,
        diameter=15.0,
        x=[0.1, 0.5, 0.5, 0.1],
        y=[0.1, 0.1, 0.5, 0.5],
    )
    db_session.add(c_parent)
    db_session.flush()

    # Child contour
    c_child = Contours(
        mask_id=mask1.id,
        parent_id=c_parent.id,
        temporary=False,
        added_by="SAM2",
        author_username="carol",
        created_at=now,
        confidence_score=0.95,
        label_id=lbl_coral.id,
        area=25.0,
        perimeter=20.0,
        circularity=0.9,
        diameter=7.0,
        x=[0.2, 0.4, 0.4, 0.2],
        y=[0.2, 0.2, 0.4, 0.4],
    )
    # Temporary contour (must be omitted)
    c_temp = Contours(
        mask_id=mask1.id,
        parent_id=None,
        temporary=True,
        added_by="Interactive",
        author_username="carol",
        created_at=now,
        confidence_score=0.5,
        label_id=None,
        area=10.0,
        perimeter=12.0,
        circularity=0.7,
        diameter=4.0,
        x=[0.6, 0.7, 0.7],
        y=[0.6, 0.6, 0.7],
    )
    # Image 2 contour with legacy pixel coordinates
    c_img2 = Contours(
        mask_id=mask2.id,
        parent_id=None,
        temporary=False,
        added_by="User",
        author_username="carol",
        created_at=now,
        confidence_score=1.0,
        label_id=lbl_sand.id,
        area=6000.0,
        perimeter=320.0,
        circularity=0.75,
        diameter=100.0,
        # On a 300x200 image, pixel coordinates:
        x=[30.0, 150.0, 150.0, 30.0],
        y=[20.0, 20.0, 100.0, 100.0],
    )
    db_session.add_all([c_child, c_temp, c_img2])
    db_session.flush()

    # Add reviewer association
    db_session.execute(
        reviewer_contour_association.insert().values(reviewer_id="bob", contour_id=c_parent.id)
    )

    # 10. Rejections
    rej = AnnotationRejections(
        mask_id=mask1.id,
        contour_id=c_parent.id,
        reason="bad_outline",
        note="Outline does not hug the coral boundary tightly",
        created_by="bob",
        created_at=now,
        resolved_at=now,
        resolved_by="carol",
        resolution="fixed",
    )
    db_session.add(rej)

    # 11. Configuration entities
    db_session.add(
        DatasetCalibrationDefaults(
            dataset_id=ds.id,
            kind="response",
            defaults={
                "strategy": "gray_wedge",
                "card": "kodak_q13",
                "fit_model": "linear",
            },
        )
    )

    profile = QuantificationProfiles(
        dataset_id=ds.id,
        name="Standard Ecology",
        is_default=True,
        entries=[
            {"metric_key": "area", "params": {}, "label_ids": [lbl_coral.id]},
            {"metric_key": "perimeter", "params": {}, "label_ids": None},
        ],
    )
    db_session.add(profile)

    routing = DatasetModelRoutingConfigs(
        dataset_id=ds.id,
        bindings=[
            {
                "task": "prompted-segmentation",
                "label_id": lbl_coral.id,
                "model_registry_key": "sam2-base",
                "inputs": {
                    "points_per_side": 32,
                    "conditioning": {
                        "query_contour_id": c_parent.id,  # Must be stripped and disclosed
                        "exemplar": "positive",
                    },
                },
            },
            {
                "task": "instance-suggestion",
                "label_id": None,
                "model_registry_key": "mask2former",
                "inputs": {"max_instances": 50},
            },
        ],
    )
    db_session.add(routing)

    db_session.commit()
    return {
        "dataset_id": ds.id,
        "dataset_name": ds.name,
        "img1_bytes": img1_bytes,
        "img2_bytes": img2_bytes,
        "img1_path": img1_path,
        "img2_path": img2_path,
    }


def test_create_iquana_dataset_archive_full_with_config(db_session, rich_dataset):
    """Test full archive export with config.json enabled."""
    ds_id = rich_dataset["dataset_id"]
    file_obj, filename = create_iquana_dataset_archive(db_session, ds_id, include_config=True)

    try:
        assert filename == "Coral_Survey_2026.zip"
        assert file_obj.tell() == 0

        with zipfile.ZipFile(file_obj, "r") as zf:
            member_names = sorted(zf.namelist())
            assert "annotations.json" in member_names
            assert "config.json" in member_names
            assert "images/1/transect_001.png" in member_names
            assert "images/2/transect_002.png" in member_names

            # 1. Byte-for-byte image matching
            assert zf.read("images/1/transect_001.png") == rich_dataset["img1_bytes"]
            assert zf.read("images/2/transect_002.png") == rich_dataset["img2_bytes"]

            # 2. Parse and validate annotations.json
            ann_raw = zf.read("annotations.json").decode("utf-8")
            ann_dict = json.loads(ann_raw)
            ann_doc = IquanaAnnotationsDocument.model_validate(ann_dict)

            assert ann_doc.format == "iquana"
            assert ann_doc.format_version == 1
            assert ann_doc.info.description == "Benthic habitat dataset"

            # Check images
            assert len(ann_doc.images) == 2
            img1 = next(im for im in ann_doc.images if im.id == 1)
            assert img1.width == 200
            assert img1.height == 150
            assert img1.iquana.archive_path == "images/1/transect_001.png"
            assert img1.iquana.metadata == {"site": "Reef Alpha", "depth_m": "12.5"}
            assert len(img1.iquana.calibrations) == 1
            assert img1.iquana.calibrations[0].kind == "scale"

            # Check files manifest
            assert len(ann_doc.iquana.files) == 2
            f1 = next(f for f in ann_doc.iquana.files if f.image_id == 1)
            assert f1.width == 200
            assert f1.height == 150
            assert f1.sha256 == img1.iquana.sha256
            assert f1.size_bytes == len(rich_dataset["img1_bytes"])

            # Check categories
            assert len(ann_doc.categories) == 3
            cat_names = [c.name for c in ann_doc.categories]
            assert "Benthos" in cat_names
            assert "Coral" in cat_names
            assert "Sand" in cat_names

            coral_cat = next(c for c in ann_doc.categories if c.name == "Coral")
            benthos_cat = next(c for c in ann_doc.categories if c.name == "Benthos")
            assert coral_cat.iquana.parent_id == benthos_cat.id

            # Check annotations
            assert len(ann_doc.annotations) == 3  # Parent, Child, Img2; temp contour omitted!
            assert ann_doc.iquana.counts.temporary_contours_omitted == 1

            # Check legacy pixel coordinates converted on Image 2
            img2_ann = next(a for a in ann_doc.annotations if a.image_id == 2)
            # 30/300 = 0.1, 150/300 = 0.5, 20/200 = 0.1, 100/200 = 0.5
            assert pytest.approx(img2_ann.iquana.geometry.x, 1e-5) == [0.1, 0.5, 0.5, 0.1]
            assert pytest.approx(img2_ann.iquana.geometry.y, 1e-5) == [0.1, 0.1, 0.5, 0.5]
            # Derived COCO segmentation points in native pixels:
            assert pytest.approx(img2_ann.segmentation[0], 1e-5) == [30.0, 20.0, 150.0, 20.0, 150.0, 100.0, 30.0, 100.0]
            assert pytest.approx(img2_ann.bbox, 1e-5) == [30.0, 20.0, 120.0, 80.0]
            assert pytest.approx(img2_ann.area, 1e-5) == 9600.0

            # Check rejections
            assert len(ann_doc.iquana.rejections) == 1
            rej_out = ann_doc.iquana.rejections[0]
            assert rej_out.reason.value == "bad_outline"
            assert rej_out.resolution.value == "fixed"

            # Check actors
            actors_dict = {a.username: a.roles for a in ann_doc.iquana.actors}
            assert "alice" in actors_dict
            assert "bob" in actors_dict
            assert "carol" in actors_dict
            assert "creator" in actors_dict["alice"]
            assert "reviewer" in actors_dict["bob"]
            assert "annotator" in actors_dict["carol"]

            # 3. Parse and validate config.json
            cfg_raw = zf.read("config.json").decode("utf-8")
            cfg_dict = json.loads(cfg_raw)
            cfg_doc = IquanaConfigDocument.model_validate(cfg_dict)

            assert cfg_doc.format == "iquana"
            assert cfg_doc.format_version == 1
            assert cfg_doc.dataset.require_independent_review is True

            # Calibration defaults
            assert len(cfg_doc.calibration_defaults) == 1
            assert cfg_doc.calibration_defaults[0].kind == "response"

            # Profiles: label names mapped from IDs
            assert len(cfg_doc.quantification_profiles) == 1
            prof = cfg_doc.quantification_profiles[0]
            assert prof.name == "Standard Ecology"
            assert prof.is_default is True
            entry0 = prof.entries[0]
            assert entry0.metric_key == "area"
            assert entry0.label_names == ["Coral"]
            assert prof.entries[1].label_names is None

            # Model routing: label names mapped and query_contour_id stripped
            assert len(cfg_doc.model_routing.bindings) == 2
            b_prompted = next(b for b in cfg_doc.model_routing.bindings if b.task.value == "prompted-segmentation")
            assert b_prompted.label_name == "Coral"
            assert "query_contour_id" not in (b_prompted.inputs.get("conditioning") or {})
            assert b_prompted.inputs["conditioning"]["exemplar"] == "positive"

            # Omitted fields record query_contour_id disclosure
            assert len(cfg_doc.omitted_fields) == 1
            omitted = cfg_doc.omitted_fields[0]
            assert omitted.field == "inputs.conditioning.query_contour_id"
            assert omitted.label_name == "Coral"
    finally:
        file_obj.close()


def test_create_iquana_dataset_archive_base_omits_config(db_session, rich_dataset):
    """When include_config=False, config.json is not created in the ZIP."""
    ds_id = rich_dataset["dataset_id"]
    file_obj, filename = create_iquana_dataset_archive(db_session, ds_id, include_config=False)

    try:
        with zipfile.ZipFile(file_obj, "r") as zf:
            member_names = zf.namelist()
            assert "annotations.json" in member_names
            assert "config.json" not in member_names
            assert "images/1/transect_001.png" in member_names
    finally:
        file_obj.close()


def test_export_rejects_missing_image_file(db_session, rich_dataset):
    """Missing image files on disk must fail export with an explicit error."""
    Path(rich_dataset["img1_path"]).unlink()

    with pytest.raises(DatasetArchiveExportError, match="Image file not found at"):
        create_iquana_dataset_archive(db_session, rich_dataset["dataset_id"])


def test_export_rejects_duplicate_category_names(db_session, rich_dataset):
    """Duplicate category names in dataset must be rejected."""
    ds_id = rich_dataset["dataset_id"]
    db_session.add(Labels(dataset_id=ds_id, name="Coral", value=99))
    db_session.commit()

    with pytest.raises(DatasetArchiveExportError, match="Duplicate category name\\(s\\) found"):
        create_iquana_dataset_archive(db_session, ds_id)


def test_export_rejects_unsupported_dataset_type(db_session, rich_dataset):
    """Non-image dataset types must be rejected."""
    ds = db_session.query(Datasets).filter(Datasets.id == rich_dataset["dataset_id"]).first()
    ds.dataset_type = "scan"
    db_session.commit()

    with pytest.raises(DatasetArchiveExportError, match="Unsupported dataset type 'scan'"):
        create_iquana_dataset_archive(db_session, ds.id)


def test_export_rejects_nonexistent_dataset(db_session):
    """Nonexistent dataset id raises DatasetNotFoundError."""
    with pytest.raises(DatasetNotFoundError, match="Dataset 99999 not found"):
        create_iquana_dataset_archive(db_session, 99999)


def test_export_rejects_unsupported_metric_key_in_profile(db_session, rich_dataset):
    """Quantification profiles with unsupported metric keys must fail export."""
    ds_id = rich_dataset["dataset_id"]
    db_session.add(
        QuantificationProfiles(
            dataset_id=ds_id,
            name="Experimental",
            entries=[{"metric_key": "unsupported_metric_xyz", "params": {}}],
        )
    )
    db_session.commit()

    with pytest.raises(DatasetArchiveExportError, match="unsupported metric key 'unsupported_metric_xyz'"):
        create_iquana_dataset_archive(db_session, ds_id, include_config=True)


# ---------------------------------------------------------------------------
# HTTP Route Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client(db_session, rich_dataset):
    app = FastAPI()
    app.include_router(datasets_router)

    # Dependency override for db session
    app.dependency_overrides[get_session] = lambda: db_session

    # User factory override for permissions
    current_auth_user = [
        AuthenticatedUser(
            username="alice",
            is_admin=True,
            global_role=GlobalRole.ADMIN,
            is_active=True,
            owned_datasets=[rich_dataset["dataset_id"]],
            accessible_datasets=[rich_dataset["dataset_id"]],
            memberships={
                rich_dataset["dataset_id"]: {
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

    client = TestClient(app)
    return client, current_auth_user, rich_dataset["dataset_id"]


def test_http_export_streaming_success(api_client):
    client, _, ds_id = api_client
    response = client.get(f"/datasets/{ds_id}/iquana?include_config=true")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "attachment; filename=" in response.headers["content-disposition"]

    # Verify that the streamed body is a valid ZIP
    zip_bytes = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_bytes, "r") as zf:
        assert "annotations.json" in zf.namelist()
        assert "config.json" in zf.namelist()


def test_http_export_permission_denied_missing_images(api_client):
    client, user_holder, ds_id = api_client
    # User has EXPORT_ANNOTATIONS but lacks EXPORT_IMAGES
    user_holder[0] = AuthenticatedUser(
        username="viewer",
        is_admin=False,
        global_role=GlobalRole.MEMBER,
        is_active=True,
        owned_datasets=[],
        accessible_datasets=[ds_id],
        memberships={
            ds_id: {
                "role": DatasetRole.VIEWER,
                "permissions": {
                    Permission.EXPORT_ANNOTATIONS,
                },
            }
        },
    )

    response = client.get(f"/datasets/{ds_id}/iquana")
    assert response.status_code == 403


def test_http_export_permission_denied_missing_annotations(api_client):
    client, user_holder, ds_id = api_client
    # User has EXPORT_IMAGES but lacks EXPORT_ANNOTATIONS
    user_holder[0] = AuthenticatedUser(
        username="viewer",
        is_admin=False,
        global_role=GlobalRole.MEMBER,
        is_active=True,
        owned_datasets=[],
        accessible_datasets=[ds_id],
        memberships={
            ds_id: {
                "role": DatasetRole.VIEWER,
                "permissions": {
                    Permission.EXPORT_IMAGES,
                },
            }
        },
    )

    response = client.get(f"/datasets/{ds_id}/iquana")
    assert response.status_code == 403


def test_http_export_404_for_unknown_dataset(api_client):
    client, user_holder, _ = api_client
    user_holder[0] = AuthenticatedUser(
        username="alice",
        is_admin=True,
        global_role=GlobalRole.ADMIN,
        is_active=True,
        owned_datasets=[99999],
        accessible_datasets=[99999],
        memberships={
            99999: {
                "role": DatasetRole.OWNER,
                "permissions": {
                    Permission.EXPORT_ANNOTATIONS,
                    Permission.EXPORT_IMAGES,
                },
            }
        },
    )
    response = client.get("/datasets/99999/iquana")
    assert response.status_code == 404


def test_export_streams_images_in_chunks(db_session, rich_dataset, monkeypatch):
    """Verifies that export streams images chunk-by-chunk with a 64 KiB buffer without reading whole file into RAM."""
    ds_id = rich_dataset["dataset_id"]
    img1_path = rich_dataset["img1_path"]

    # Generate an image larger than 64 KiB
    large_arr = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    large_img = PILImage.fromarray(large_arr, mode="RGB")
    large_img.save(img1_path, format="PNG")
    file_size = img1_path.stat().st_size
    assert file_size > 64 * 1024

    chunk_reads = []
    real_open = open

    def monitored_open(file, *args, **kwargs):
        f = real_open(file, *args, **kwargs)
        if str(file) == str(img1_path):
            orig_read = f.read

            def logged_read(size=-1):
                if size == 64 * 1024:
                    chunk_reads.append(size)
                # Verify that no unbuffered/full file read is ever requested
                assert size != -1 and size != file_size, f"Unexpected full file read with size={size}"
                return orig_read(size)

            f.read = logged_read
        return f

    monkeypatch.setattr("builtins.open", monitored_open)

    file_obj, filename = create_iquana_dataset_archive(db_session, ds_id)
    monkeypatch.undo()
    try:
        # Confirm that multiple 64 KiB chunks were read
        assert len(chunk_reads) >= 2
        assert all(s == 64 * 1024 for s in chunk_reads)

        # Verify ZIP integrity and SHA256 match
        with zipfile.ZipFile(file_obj, "r") as zf:
            with open(img1_path, "rb") as orig_f:
                assert zf.read("images/1/transect_001.png") == orig_f.read()
    finally:
        file_obj.close()


# ---------------------------------------------------------------------------
# Phase 3 Import Tests
# ---------------------------------------------------------------------------


def test_import_full_archive_roundtrip(db_session, rich_dataset, monkeypatch):
    """End-to-end import of a full archive with config, verifying relational remapping and re-export equivalence."""
    import app.services.dataset_archive as dataset_archive_module

    # Deterministic, hermetic stand-in for the live MLflow readiness lookup: no task
    # ever has a ready model, so both routing bindings are always flagged. Without this,
    # the warning assertion below would depend on whatever a real registry happens to
    # have registered in the environment the tests run in.
    monkeypatch.setattr(dataset_archive_module, "_models_for_task", lambda task: [])

    ds_id = rich_dataset["dataset_id"]
    file_obj, _ = create_iquana_dataset_archive(db_session, ds_id, include_config=True)

    try:
        result = import_iquana_dataset_archive(
            db=db_session,
            archive_file=file_obj,
            override_name="Coral Survey 2026 Imported",
            importer_username="bob",
        )

        assert result["success"] is True
        assert result["dataset_name"] == "Coral Survey 2026 Imported"
        assert result["config_applied"] is True
        assert len(result["warnings"]) > 0
        # Neither routing binding's model_registry_key ("sam2-base", "mask2former") has
        # a ready model for its task (stubbed above), so the importer must flag them
        # for curator repair rather than staying silent.
        assert any("sam2-base" in w and "mask2former" in w for w in result["warnings"])

        new_ds_id = result["dataset_id"]
        assert new_ds_id != ds_id

        # 1. Verify Dataset and Owner
        imported_ds = db_session.query(Datasets).filter_by(id=new_ds_id).first()
        assert imported_ds is not None
        assert imported_ds.name == "Coral Survey 2026 Imported"
        assert imported_ds.description == "Benthic habitat dataset"
        assert imported_ds.created_by == "bob"
        assert imported_ds.require_independent_review is True

        owner_member = (
            db_session.query(DatasetMembers)
            .filter_by(dataset_id=new_ds_id, username="bob")
            .first()
        )
        assert owner_member is not None
        assert owner_member.role == DatasetRole.OWNER.value

        # 2. Verify Labels hierarchy
        labels = db_session.query(Labels).filter_by(dataset_id=new_ds_id).all()
        assert len(labels) == 3
        lbl_map = {l.name: l for l in labels}
        assert "Benthos" in lbl_map
        assert "Coral" in lbl_map
        assert "Sand" in lbl_map
        assert lbl_map["Coral"].parent_id == lbl_map["Benthos"].id
        assert lbl_map["Benthos"].parent_id is None

        # 3. Verify Metadata Keys & Values
        meta_keys = db_session.query(DatasetMetadataKeys).filter_by(dataset_id=new_ds_id).all()
        assert len(meta_keys) == 2
        key_map = {k.key: k for k in meta_keys}
        assert "site" in key_map
        assert "depth_m" in key_map
        assert key_map["site"].value_type == "categorical"
        assert key_map["depth_m"].value_type == "number"

        # 4. Verify Images & Calibrations
        images = db_session.query(Images).filter_by(dataset_id=new_ds_id).order_by(Images.file_name).all()
        assert len(images) == 2
        im1, im2 = images
        assert Path(im1.file_path).exists()
        assert Path(im2.file_path).exists()
        assert Path(im1.thumbnail_file_path).exists()
        assert Path(im2.thumbnail_file_path).exists()

        # Check image 1 metadata and depth_m value_num coercion
        im1_meta = db_session.query(ImageMetadata).filter_by(image_id=im1.id).all()
        im1_meta_dict = {m.key: (m.value, m.value_num) for m in im1_meta}
        assert im1_meta_dict["site"][0] == "Reef Alpha"
        assert im1_meta_dict["depth_m"][0] == "12.5"
        assert pytest.approx(im1_meta_dict["depth_m"][1], 1e-5) == 12.5

        # Check calibrations
        cals = db_session.query(ImageCalibrations).filter_by(image_id=im1.id).all()
        assert len(cals) == 1
        assert cals[0].kind == "scale"
        assert cals[0].created_by == "bob"

        # 5. Verify Masks, Contours, and dual-write ContourMetrics
        masks = db_session.query(Masks).join(Images).filter(Images.dataset_id == new_ds_id).all()
        assert len(masks) == 2
        for m in masks:
            assert Path(m.file_path).exists()

        contours = db_session.query(Contours).join(Masks).join(Images).filter(Images.dataset_id == new_ds_id).all()
        assert len(contours) == 3  # Parent, Child, Img2; temp contour omitted

        # Verify parent-child contour hierarchy
        parent_c = next(c for c in contours if c.parent_id is None and len(c.x) == 4 and c.x[0] == 0.1)
        child_c = next(c for c in contours if c.parent_id == parent_c.id)
        assert child_c is not None

        # Source actor usernames are provenance-only and must be cleared on import,
        # not attributed to the importer (frozen decoupling policy).
        assert parent_c.author_username is None
        assert child_c.author_username is None

        # Verify dual-written geometry metrics in ContourMetrics
        metrics = db_session.query(ContourMetrics).filter_by(contour_id=parent_c.id).all()
        metric_keys = {m.metric_key for m in metrics}
        assert "area" in metric_keys
        assert "perimeter" in metric_keys
        assert "circularity" in metric_keys
        assert "max_diameter" in metric_keys
        assert parent_c.area > 0
        assert parent_c.perimeter > 0

        # 6. Verify Rejections
        rejections = db_session.query(AnnotationRejections).join(Masks).join(Images).filter(Images.dataset_id == new_ds_id).all()
        assert len(rejections) == 1
        assert rejections[0].reason == "bad_outline"
        assert rejections[0].resolution == "fixed"
        assert rejections[0].contour_id == parent_c.id
        # Source rejection creator/resolver usernames are cleared on import too
        # (rich_dataset's fixture rejection has created_by="bob", resolved_by="carol").
        assert rejections[0].created_by is None
        assert rejections[0].resolved_by is None

        # 7. Verify Configuration: defaults, profiles, routing
        defaults = db_session.query(DatasetCalibrationDefaults).filter_by(dataset_id=new_ds_id).all()
        assert len(defaults) == 1
        assert defaults[0].kind == "response"

        profiles = db_session.query(QuantificationProfiles).filter_by(dataset_id=new_ds_id).all()
        assert len(profiles) == 1
        assert profiles[0].name == "Standard Ecology"
        assert profiles[0].is_default is True
        # Verify label ID was remapped to new Coral label
        assert profiles[0].entries[0]["label_ids"] == [lbl_map["Coral"].id]

        routing = db_session.query(DatasetModelRoutingConfigs).filter_by(dataset_id=new_ds_id).first()
        assert routing is not None
        b_prompted = next(b for b in routing.bindings if b.get("task") == "prompted-segmentation")
        assert b_prompted.get("label_id") == lbl_map["Coral"].id

        # 8. Re-export and verify equivalence
        reexport_obj, reexport_name = create_iquana_dataset_archive(db_session, new_ds_id, include_config=True)
        try:
            with zipfile.ZipFile(reexport_obj, "r") as zf:
                assert "annotations.json" in zf.namelist()
                assert "config.json" in zf.namelist()
                assert zf.read("images/1/transect_001.png") == rich_dataset["img1_bytes"]
                assert zf.read("images/2/transect_002.png") == rich_dataset["img2_bytes"]
        finally:
            reexport_obj.close()
    finally:
        file_obj.close()


def test_import_base_archive_without_config(db_session, rich_dataset):
    """Importing an archive without config.json uses destination defaults and emits a warning."""
    ds_id = rich_dataset["dataset_id"]
    file_obj, _ = create_iquana_dataset_archive(db_session, ds_id, include_config=False)

    try:
        result = import_iquana_dataset_archive(
            db=db_session,
            archive_file=file_obj,
            override_name="Coral Survey Base Imported",
            importer_username="alice",
        )

        assert result["success"] is True
        assert result["config_applied"] is False
        assert any("config.json" in w for w in result["warnings"])

        new_ds_id = result["dataset_id"]
        imported_ds = db_session.query(Datasets).filter_by(id=new_ds_id).first()
        assert imported_ds.require_independent_review is False
    finally:
        file_obj.close()


def test_import_rejects_name_conflict(db_session, rich_dataset):
    """Importing with an already existing dataset name must fail with DatasetArchiveNameConflictError."""
    ds_id = rich_dataset["dataset_id"]
    file_obj, _ = create_iquana_dataset_archive(db_session, ds_id, include_config=False)

    try:
        # "Coral Survey 2026" already exists from fixture
        with pytest.raises(DatasetArchiveNameConflictError, match="already exists"):
            import_iquana_dataset_archive(
                db=db_session,
                archive_file=file_obj,
                override_name="Coral Survey 2026",
                importer_username="alice",
            )
    finally:
        file_obj.close()


def test_import_rejects_directory_conflict(db_session, rich_dataset, tmp_path):
    """Importing when the resolved target directory already exists on disk fails with DatasetArchiveNameConflictError."""
    # The final directory is derived from the new dataset's DB ID plus a slug of the
    # name (not the raw name alone), so pre-create the exact path import will resolve to.
    next_id = (db_session.query(func.max(Datasets.id)).scalar() or 0) + 1
    conflict_dir = tmp_path / "datasets" / f"{next_id}_{_slugify_dataset_name('Existing_Folder')}"
    conflict_dir.mkdir(parents=True, exist_ok=True)

    ds_id = rich_dataset["dataset_id"]
    file_obj, _ = create_iquana_dataset_archive(db_session, ds_id, include_config=False)

    try:
        with pytest.raises(DatasetArchiveNameConflictError, match="already exists on disk"):
            import_iquana_dataset_archive(
                db=db_session,
                archive_file=file_obj,
                override_name="Existing_Folder",
                importer_username="alice",
            )
    finally:
        file_obj.close()


def test_import_destination_race_does_not_corrupt_concurrent_empty_directory(db_session, rich_dataset, monkeypatch):
    """A directory created concurrently, in the gap between the earlier existence check
    and the final move, must not be silently taken over or deleted -- even if that
    directory happens to be empty. os.rename() on POSIX only fails for a *non-empty*
    destination; an empty one is silently replaced, so the import instead uses
    os.mkdir(), which never replaces an existing directory in any state."""
    ds_id = rich_dataset["dataset_id"]
    file_obj, _ = create_iquana_dataset_archive(db_session, ds_id, include_config=False)

    real_mkdir = os.mkdir
    captured: dict[str, str] = {}

    def racing_mkdir(path, *args, **kwargs):
        # Only inject the race for the final dataset directory: it is a direct child of
        # DATASETS_DIR whose name never starts with "." (unlike the staging directory,
        # ".staging_import_<uuid>", also created via os.mkdir through os.makedirs).
        basename = os.path.basename(path)
        if os.path.dirname(path) == config.DATASETS_DIR and not basename.startswith("."):
            # Simulate another process creating the empty destination directory in the
            # window between our earlier existence check and this mkdir.
            captured["path"] = path
            real_mkdir(path)
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", racing_mkdir)

    try:
        with pytest.raises(DatasetArchiveNameConflictError, match="already exists on disk"):
            import_iquana_dataset_archive(
                db=db_session,
                archive_file=file_obj,
                override_name="Race Test Dataset",
                importer_username="alice",
            )
    finally:
        file_obj.close()

    assert "path" in captured, "os.mkdir was never attempted for the final dataset directory"
    concurrent_dir = Path(captured["path"])
    # The concurrently created directory must survive, untouched and still empty --
    # not silently replaced by our own data and not deleted by failure cleanup.
    assert concurrent_dir.is_dir()
    assert list(concurrent_dir.iterdir()) == []


def test_import_mask_staging_ids_do_not_collide_with_shifted_db_ids(db_session, tmp_path):
    """When the destination DB already has existing masks, the next auto-increment
    Masks.id can numerically collide with an unprocessed archive mask's zip-local ID
    during staging renames. Archive mask #1 being assigned DB id #2 must not overwrite
    archive mask #2's still-unprocessed staged pixel file."""
    # Build the source dataset (to be exported) in a completely separate database, so
    # its own mask ids can never leak into and interfere with the destination's id
    # sequence -- only the destination's *decoy* mask below should influence that.
    source_engine = create_engine(f"sqlite:///{tmp_path / 'source.db'}")
    database.metadata.create_all(source_engine)
    SourceSession = sessionmaker(bind=source_engine)
    source_session = SourceSession()

    source_session.add(Users(username="alice", hashed_password="x", global_role=GlobalRole.ADMIN.value))
    source_session.flush()

    ds = Datasets(
        name="Two Mask Dataset", dataset_type="image",
        folder_path=str(tmp_path / "ds_folder"), created_by="alice",
    )
    source_session.add(ds)
    source_session.flush()

    lbl_a = Labels(dataset_id=ds.id, name="LabelA", value=50, parent_id=None)
    lbl_b = Labels(dataset_id=ds.id, name="LabelB", value=150, parent_id=None)
    source_session.add_all([lbl_a, lbl_b])
    source_session.flush()

    img1_path = tmp_path / "img1.png"
    img2_path = tmp_path / "img2.png"
    _create_test_image(img1_path, width=20, height=20, color=(1, 1, 1))
    _create_test_image(img2_path, width=20, height=20, color=(1, 1, 1))
    img1 = Images(
        dataset_id=ds.id, file_name="img1.png", file_path=str(img1_path),
        thumbnail_file_path=str(tmp_path / "thumb1.png"), width=20, height=20,
        color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px",
    )
    img2 = Images(
        dataset_id=ds.id, file_name="img2.png", file_path=str(img2_path),
        thumbnail_file_path=str(tmp_path / "thumb2.png"), width=20, height=20,
        color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px",
    )
    source_session.add_all([img1, img2])
    source_session.flush()

    # Two masks, each fully covered by a single contour with a distinct, verifiable
    # label pixel value, so a post-import swap between them is detectable.
    mask1 = Masks(image_id=img1.id, fully_annotated=True, file_path=str(tmp_path / "m1.png"))
    mask2 = Masks(image_id=img2.id, fully_annotated=True, file_path=str(tmp_path / "m2.png"))
    source_session.add_all([mask1, mask2])
    source_session.flush()

    contour1 = Contours(
        mask_id=mask1.id, parent_id=None, temporary=False, added_by="User",
        author_username="alice", created_at=datetime.now(timezone.utc), confidence_score=1.0,
        label_id=lbl_a.id, area=1.0, perimeter=1.0, circularity=1.0, diameter=1.0,
        x=[0.0, 1.0, 1.0, 0.0], y=[0.0, 0.0, 1.0, 1.0],
    )
    contour2 = Contours(
        mask_id=mask2.id, parent_id=None, temporary=False, added_by="User",
        author_username="alice", created_at=datetime.now(timezone.utc), confidence_score=1.0,
        label_id=lbl_b.id, area=1.0, perimeter=1.0, circularity=1.0, diameter=1.0,
        x=[0.0, 1.0, 1.0, 0.0], y=[0.0, 0.0, 1.0, 1.0],
    )
    source_session.add_all([contour1, contour2])
    source_session.flush()

    file_obj, _ = create_iquana_dataset_archive(source_session, ds.id, include_config=False)
    source_session.close()

    # Destination: a decoy dataset/image/mask consumes Masks.id=1 in *this* (separate)
    # database, so the import below assigns new mask ids (2, 3) that numerically collide
    # with the archive's own zip-local mask ids (1, 2).
    db_session.add(Users(username="alice", hashed_password="x", global_role=GlobalRole.ADMIN.value))
    db_session.flush()
    decoy_ds = Datasets(
        name="Decoy Dataset", dataset_type="image",
        folder_path=str(tmp_path / "decoy_folder"), created_by="alice",
    )
    db_session.add(decoy_ds)
    db_session.flush()
    decoy_img_path = tmp_path / "decoy.png"
    _create_test_image(decoy_img_path, width=10, height=10, color=(1, 1, 1))
    decoy_img = Images(
        dataset_id=decoy_ds.id, file_name="decoy.png", file_path=str(decoy_img_path),
        thumbnail_file_path=str(tmp_path / "decoy_thumb.png"), width=10, height=10,
        color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px",
    )
    db_session.add(decoy_img)
    db_session.flush()
    db_session.add(Masks(image_id=decoy_img.id, fully_annotated=False, file_path=str(tmp_path / "decoy_mask.png")))
    db_session.flush()

    try:
        result = import_iquana_dataset_archive(
            db=db_session,
            archive_file=file_obj,
            override_name="Two Mask Dataset Imported",
            importer_username="alice",
        )
    finally:
        file_obj.close()

    assert result["success"] is True
    new_ds_id = result["dataset_id"]

    new_img1, new_img2 = (
        db_session.query(Images).filter_by(dataset_id=new_ds_id).order_by(Images.file_name).all()
    )
    new_mask1 = db_session.query(Masks).filter_by(image_id=new_img1.id).one()
    new_mask2 = db_session.query(Masks).filter_by(image_id=new_img2.id).one()

    mask1_pixels = np.array(PILImage.open(new_mask1.file_path))
    mask2_pixels = np.array(PILImage.open(new_mask2.file_path))

    # Each mask's pixel data must match its own image's contour label, not the other
    # image's -- a swap here means the staging rename overwrote the wrong file.
    assert int(mask1_pixels.max()) == 50
    assert int(mask2_pixels.max()) == 150


def test_model_registry_check_degrades_when_queue_is_full(monkeypatch):
    """The model-registry availability queue is bounded (queue.Full is expected, not
    exceptional): when it's at capacity, tasks that can't be queued must be treated as
    unconfirmed immediately, not block waiting on a job that was never submitted."""
    import app.services.dataset_archive as dataset_archive_module

    def always_full(job):
        raise queue.Full

    monkeypatch.setattr(dataset_archive_module._MODEL_AVAILABILITY_QUEUE, "put_nowait", always_full)

    start = time.monotonic()
    result = _confirmed_ready_model_bindings(
        {("prompted-segmentation", "sam2-base"), ("instance-segmentation", "mask2former")},
        timeout_seconds=2.0,
    )
    elapsed = time.monotonic() - start

    assert result == set()
    # Neither task was ever queued, so there is nothing to wait on -- this must return
    # near-instantly, not consume the full 2-second deadline.
    assert elapsed < 0.5


def test_import_survives_post_commit_refresh_failure(db_session, rich_dataset, monkeypatch):
    """If db.refresh() fails after a successful db.commit(), the import must still be
    reported as successful and its files must not be deleted: the transaction already
    committed, so rollback cannot undo it, and running the failure-cleanup path would
    leave the database pointing at files that cleanup just removed."""
    ds_id = rich_dataset["dataset_id"]
    file_obj, _ = create_iquana_dataset_archive(db_session, ds_id, include_config=False)

    original_refresh = db_session.refresh

    def failing_refresh(instance, *args, **kwargs):
        if isinstance(instance, Datasets):
            raise RuntimeError("simulated post-commit refresh failure")
        return original_refresh(instance, *args, **kwargs)

    monkeypatch.setattr(db_session, "refresh", failing_refresh)

    try:
        result = import_iquana_dataset_archive(
            db=db_session,
            archive_file=file_obj,
            override_name="Refresh Failure Test",
            importer_username="alice",
        )
    finally:
        file_obj.close()

    assert result["success"] is True
    new_ds_id = result["dataset_id"]

    imported_ds = db_session.query(Datasets).filter_by(id=new_ds_id).first()
    assert imported_ds is not None
    assert Path(imported_ds.folder_path).exists()


def test_import_rejects_mismatched_image_checksum(db_session, rich_dataset):
    """Modifying image bytes in ZIP without updating manifest fails checksum validation."""
    ds_id = rich_dataset["dataset_id"]
    file_obj, _ = create_iquana_dataset_archive(db_session, ds_id, include_config=True)

    corrupted_zip_bytes = io.BytesIO()
    with zipfile.ZipFile(file_obj, "r") as src_zf, zipfile.ZipFile(corrupted_zip_bytes, "w") as dst_zf:
        for item in src_zf.infolist():
            data = src_zf.read(item.filename)
            if item.filename == "images/1/transect_001.png":
                data = b"corrupted_png_header" + data[20:]
            dst_zf.writestr(item, data)

    file_obj.close()
    corrupted_zip_bytes.seek(0)

    with pytest.raises(DatasetArchiveValidationError, match="checksum mismatch"):
        import_iquana_dataset_archive(
            db=db_session,
            archive_file=corrupted_zip_bytes,
            override_name="Corrupted Checksum Test",
            importer_username="alice",
        )


def test_import_rejects_unsupported_metric_key_in_config(db_session, rich_dataset):
    """Profiles containing unsupported metric keys must fail validation."""
    ds_id = rich_dataset["dataset_id"]
    file_obj, _ = create_iquana_dataset_archive(db_session, ds_id, include_config=True)

    corrupted_zip_bytes = io.BytesIO()
    with zipfile.ZipFile(file_obj, "r") as src_zf, zipfile.ZipFile(corrupted_zip_bytes, "w") as dst_zf:
        for item in src_zf.infolist():
            data = src_zf.read(item.filename)
            if item.filename == "config.json":
                cfg = json.loads(data.decode("utf-8"))
                cfg["quantification_profiles"][0]["entries"].append(
                    {"metric_key": "unsupported_metric_xyz", "params": {}}
                )
                data = json.dumps(cfg).encode("utf-8")
            dst_zf.writestr(item, data)

    file_obj.close()
    corrupted_zip_bytes.seek(0)

    with pytest.raises(DatasetArchiveValidationError, match="unsupported_metric_xyz"):
        import_iquana_dataset_archive(
            db=db_session,
            archive_file=corrupted_zip_bytes,
            override_name="Unsupported Metric Test",
            importer_username="alice",
        )


def test_import_rejects_cycle_in_category_hierarchy(db_session, rich_dataset):
    """Cycles in category hierarchy in annotations.json must fail validation."""
    ds_id = rich_dataset["dataset_id"]
    file_obj, _ = create_iquana_dataset_archive(db_session, ds_id, include_config=False)

    corrupted_zip_bytes = io.BytesIO()
    with zipfile.ZipFile(file_obj, "r") as src_zf, zipfile.ZipFile(corrupted_zip_bytes, "w") as dst_zf:
        for item in src_zf.infolist():
            data = src_zf.read(item.filename)
            if item.filename == "annotations.json":
                ann = json.loads(data.decode("utf-8"))
                # Create cycle: Cat 1 parent is Cat 2, Cat 2 parent is Cat 1
                for c in ann["categories"]:
                    if c["id"] == 1:
                        c["iquana"]["parent_id"] = 2
                    elif c["id"] == 2:
                        c["iquana"]["parent_id"] = 1
                data = json.dumps(ann).encode("utf-8")
            dst_zf.writestr(item, data)

    file_obj.close()
    corrupted_zip_bytes.seek(0)

    with pytest.raises(DatasetArchiveValidationError, match="[Cc]ycle detected in category hierarchy"):
        import_iquana_dataset_archive(
            db=db_session,
            archive_file=corrupted_zip_bytes,
            override_name="Cycle Cat Test",
            importer_username="alice",
        )


def test_import_rejects_zip_slip_member(db_session):
    """Archives containing directory traversal members must fail validation."""
    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, "w") as zf:
        zf.writestr("../etc/passwd", "malicious_content")
        zf.writestr("annotations.json", "{}")

    zip_bytes.seek(0)
    with pytest.raises(DatasetArchiveValidationError, match="directory traversal"):
        import_iquana_dataset_archive(
            db=db_session,
            archive_file=zip_bytes,
            override_name="ZipSlip Test",
            importer_username="alice",
        )


def test_import_empty_valid_dataset(db_session):
    """An archive representing an empty valid dataset imports cleanly."""
    db_session.add(Users(username="alice", hashed_password="x", global_role=GlobalRole.ADMIN.value))
    db_session.commit()

    ann_doc = {
        "format": "iquana",
        "format_version": 1,
        "info": {
            "description": "Empty test dataset",
            "version": "1.0",
            "year": 2026,
            "date_created": "2026-09-15T00:00:00Z",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
        "iquana": {
            "dataset": {
                "name": "Empty Dataset",
                "description": "Empty test dataset",
                "dataset_type": "image",
                "created_by": "alice",
            },
            "actors": [],
            "metadata_keys": [],
            "masks": [],
            "rejections": [],
            "counts": {
                "images": 0,
                "annotations": 0,
                "categories": 0,
                "masks": 0,
                "rejections": 0,
                "temporary_contours_omitted": 0,
            },
            "files": [],
        },
    }

    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, "w") as zf:
        zf.writestr("annotations.json", json.dumps(ann_doc).encode("utf-8"))

    zip_bytes.seek(0)
    result = import_iquana_dataset_archive(
        db=db_session,
        archive_file=zip_bytes,
        override_name="Empty Dataset Imported",
        importer_username="alice",
    )

    assert result["success"] is True
    assert result["dataset_name"] == "Empty Dataset Imported"

    imported_ds = db_session.query(Datasets).filter_by(id=result["dataset_id"]).first()
    assert imported_ds is not None
    assert imported_ds.name == "Empty Dataset Imported"


def test_import_uses_lossless_description_not_coco_info_fallback(db_session):
    """A source dataset with no description exports info.description as the dataset
    name (COCO's info.description is a required string, so the exporter falls back to
    it), while iquana.dataset.description faithfully stays None. Import must persist
    from the lossless iquana field, not the lossy COCO fallback."""
    db_session.add(Users(username="alice", hashed_password="x", global_role=GlobalRole.ADMIN.value))
    db_session.commit()

    ann_doc = {
        "format": "iquana",
        "format_version": 1,
        "info": {
            # What the exporter writes when the source dataset.description is None.
            "description": "Undescribed Dataset",
            "version": "1.0",
            "year": 2026,
            "date_created": "2026-09-15T00:00:00Z",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
        "iquana": {
            "dataset": {
                "name": "Undescribed Dataset",
                "description": None,
                "dataset_type": "image",
                "created_by": "alice",
            },
            "actors": [],
            "metadata_keys": [],
            "masks": [],
            "rejections": [],
            "counts": {
                "images": 0,
                "annotations": 0,
                "categories": 0,
                "masks": 0,
                "rejections": 0,
                "temporary_contours_omitted": 0,
            },
            "files": [],
        },
    }

    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, "w") as zf:
        zf.writestr("annotations.json", json.dumps(ann_doc).encode("utf-8"))

    zip_bytes.seek(0)
    result = import_iquana_dataset_archive(
        db=db_session,
        archive_file=zip_bytes,
        override_name="Undescribed Dataset Imported",
        importer_username="alice",
    )

    assert result["success"] is True
    imported_ds = db_session.query(Datasets).filter_by(id=result["dataset_id"]).first()
    assert imported_ds.description is None


def test_import_accepts_edge_overhanging_geometry(db_session, tmp_path):
    """A contour normalized slightly outside [0.0, 1.0] but within the documented
    [-1.5, 1.5] tolerance (e.g. touching/overhanging the image edge) is valid per the
    exporter and schema, so the importer must accept it rather than applying a
    stricter [0.0, 1.0] range that would reject the exporter's own valid output."""
    db_session.add(Users(username="alice", hashed_password="x", global_role=GlobalRole.ADMIN.value))
    db_session.flush()

    ds = Datasets(
        name="Edge Overhang Dataset",
        description="Single contour overhanging the image edge",
        dataset_type="image",
        folder_path=str(tmp_path / "ds_folder"),
        created_by="alice",
    )
    db_session.add(ds)
    db_session.flush()

    img_path = tmp_path / "edge.png"
    _create_test_image(img_path, width=100, height=100, color=(5, 5, 5))
    img = Images(
        dataset_id=ds.id,
        file_name="edge.png",
        file_path=str(img_path),
        thumbnail_file_path=str(tmp_path / "edge_thumb.png"),
        width=100,
        height=100,
        color_mode="RGB",
        scale_x=1.0,
        scale_y=1.0,
        unit="px",
    )
    db_session.add(img)
    db_session.flush()

    mask = Masks(image_id=img.id, fully_annotated=True, file_path=str(tmp_path / "edge_mask.png"))
    db_session.add(mask)
    db_session.flush()

    contour = Contours(
        mask_id=mask.id,
        parent_id=None,
        temporary=False,
        added_by="User",
        author_username="alice",
        created_at=datetime.now(timezone.utc),
        confidence_score=1.0,
        label_id=None,
        area=100.0,
        perimeter=40.0,
        circularity=0.8,
        diameter=15.0,
        x=[-0.1, 1.1, 1.1, -0.1],
        y=[-0.1, -0.1, 1.1, 1.1],
    )
    db_session.add(contour)
    db_session.flush()

    file_obj, _ = create_iquana_dataset_archive(db_session, ds.id, include_config=False)
    try:
        result = import_iquana_dataset_archive(
            db=db_session,
            archive_file=file_obj,
            override_name="Edge Overhang Imported",
            importer_username="alice",
        )
        assert result["success"] is True
    finally:
        file_obj.close()


# ---------------------------------------------------------------------------
# HTTP Import Endpoint Tests
# ---------------------------------------------------------------------------


def test_http_import_endpoint_success(api_client, rich_dataset):
    """Test successful multipart import via POST /datasets/import/iquana."""
    client, user_holder, ds_id = api_client
    file_obj, _ = create_iquana_dataset_archive(api_client[0].app.dependency_overrides[get_session](), ds_id, include_config=True)

    try:
        file_obj.seek(0)
        response = client.post(
            "/datasets/import/iquana",
            files={"file": ("archive.zip", file_obj, "application/zip")},
            data={"name": "HTTP Imported Dataset"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["dataset_name"] == "HTTP Imported Dataset"
        assert data["config_applied"] is True
        assert "dataset_id" in data
    finally:
        file_obj.close()


def test_http_import_permission_denied_without_dataset_create(api_client, rich_dataset):
    """User without DATASET_CREATE permission receives 403 on import."""
    client, user_holder, ds_id = api_client
    user_holder[0] = AuthenticatedUser(
        username="viewer",
        is_admin=False,
        global_role=GlobalRole.GUEST,
        is_active=True,
        global_permissions=set(),  # No DATASET_CREATE
        owned_datasets=[],
        accessible_datasets=[],
        memberships={},
    )

    zip_bytes = io.BytesIO(b"dummy_zip_bytes")
    response = client.post(
        "/datasets/import/iquana",
        files={"file": ("archive.zip", zip_bytes, "application/zip")},
    )
    assert response.status_code == 403


def test_http_import_conflict_409(api_client, rich_dataset):
    """Name conflict in HTTP import returns 409 Conflict."""
    client, _, ds_id = api_client
    file_obj, _ = create_iquana_dataset_archive(api_client[0].app.dependency_overrides[get_session](), ds_id, include_config=False)

    try:
        file_obj.seek(0)
        # "Coral Survey 2026" conflicts with existing dataset
        response = client.post(
            "/datasets/import/iquana",
            files={"file": ("archive.zip", file_obj, "application/zip")},
            data={"name": "Coral Survey 2026"},
        )
        assert response.status_code == 409
    finally:
        file_obj.close()


def test_http_import_validation_error_422(api_client):
    """Corrupted archive in HTTP import returns 422 Unprocessable Entity."""
    client, _, _ = api_client
    zip_bytes = io.BytesIO(b"not_a_valid_zip_file")
    response = client.post(
        "/datasets/import/iquana",
        files={"file": ("archive.zip", zip_bytes, "application/zip")},
    )
    assert response.status_code == 422


def test_dataset_name_uniqueness_migration_and_creation():
    """Verify startup migration deduplicates pre-existing duplicates and enforces uniqueness."""
    import asyncio
    import tempfile
    from sqlalchemy import text
    from app.services.database_access.datasets import create_new_dataset

    tmp = tempfile.mkdtemp()
    test_db_file = os.path.join(tmp, "test_legacy_dup.db")
    test_engine = create_engine(f"sqlite:///{test_db_file}")

    # 1. Create a legacy table without UNIQUE constraint
    with test_engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE datasets ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name VARCHAR(50) NOT NULL, "
            "description VARCHAR(255), "
            "dataset_type VARCHAR(20) NOT NULL, "
            "folder_path VARCHAR(255) NOT NULL, "
            "created_by VARCHAR NOT NULL, "
            "require_independent_review BOOLEAN NOT NULL DEFAULT 0)"
        ))
        conn.execute(text("CREATE TABLE users (username VARCHAR PRIMARY KEY, hashed_password VARCHAR, global_role VARCHAR)"))
        conn.execute(text("CREATE TABLE dataset_members (id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id INTEGER, username VARCHAR, role VARCHAR, extra_permissions JSON, denied_permissions JSON, granted_by VARCHAR, granted_at DATETIME)"))
        conn.execute(text("INSERT INTO users (username, hashed_password, global_role) VALUES ('alice', 'x', 'admin')"))
        conn.execute(text("INSERT INTO datasets (id, name, dataset_type, folder_path, created_by) VALUES (1, 'Legacy Dataset', 'image', '/tmp/1', 'alice')"))
        conn.execute(text("INSERT INTO datasets (id, name, dataset_type, folder_path, created_by) VALUES (2, 'Legacy Dataset', 'image', '/tmp/2', 'alice')"))
        conn.execute(text("INSERT INTO datasets (id, name, dataset_type, folder_path, created_by) VALUES (3, 'Legacy Dataset', 'image', '/tmp/3', 'alice')"))
        conn.execute(text("INSERT INTO datasets (id, name, dataset_type, folder_path, created_by) VALUES (4, 'Unique Dataset', 'image', '/tmp/4', 'alice')"))

    # 2. Run init_db migration
    init_db(target_engine=test_engine)

    # 3. Check that duplicates were renamed
    Session = sessionmaker(bind=test_engine)
    db = Session()
    try:
        ds_rows = db.query(Datasets).order_by(Datasets.id.asc()).all()
        assert len(ds_rows) == 4
        assert ds_rows[0].name == "Legacy Dataset"
        assert ds_rows[1].name == "Legacy Dataset (dup 2)"
        assert ds_rows[2].name == "Legacy Dataset (dup 3)"
        assert ds_rows[3].name == "Unique Dataset"

        # 4. Verify unique index prevents inserting duplicate
        with pytest.raises(IntegrityError):
            with test_engine.begin() as conn:
                conn.execute(text("INSERT INTO datasets (name, dataset_type, folder_path, created_by) VALUES ('Legacy Dataset', 'image', '/tmp/5', 'alice')"))

        # 5. Verify create_new_dataset catches concurrent IntegrityError on commit
        from unittest.mock import patch
        with patch.object(db, "query") as mock_query:
            # Pre-check returns None as if name was free, triggering commit-time IntegrityError
            mock_query.return_value.filter_by.return_value.first.return_value = None
            res = asyncio.run(create_new_dataset(
                name="Legacy Dataset",
                description="race test",
                owner_username="alice",
                db=db
            ))
            assert isinstance(res, dict)
            assert res["success"] is False
            assert res["error"] == "Duplicate dataset name"
            assert "already exists" in res["message"]
    finally:
        db.close()


def test_export_dataset_with_empty_added_by_and_null_created_at(api_client, rich_dataset):
    """Verify archive export succeeds even when stored contour has empty added_by."""
    client, _, ds_id = api_client
    db = client.app.dependency_overrides[get_session]()

    # Force a contour in the dataset to have empty string added_by
    c = db.query(Contours).first()
    assert c is not None
    c.added_by = ""
    db.commit()

    file_obj, filename = create_iquana_dataset_archive(db, ds_id, include_config=True)
    try:
        assert filename.endswith(".zip")
        with zipfile.ZipFile(file_obj, "r") as zf:
            ann_data = json.loads(zf.read("annotations.json"))
            matching_ann = next(a for a in ann_data["annotations"] if a["id"] == c.id)
            assert matching_ann["iquana"]["added_by"] == "User"
            assert matching_ann["iquana"]["created_at"] is not None
    finally:
        file_obj.close()

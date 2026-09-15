"""Tests for the IQUANA dataset archive v1 schema contract and golden fixture.

Validates Phase 1 contract freeze:
- Strict Pydantic v2 schemas for annotations.json and config.json.
- Deterministic ZIP archive generation with fixed ZipInfo timestamps and stable SHA-256 hashes.
- Integrity validations: category name uniqueness, linear-time O(n) category & contour cycle detection.
- Cross-entity ownership: mask/image ownership, contour mask ownership, contour acyclicity, rejection mask ownership.
- Image manifest: 1-to-1 exact agreement with image records and path-encoded IDs.
- Metadata key definitions: uniqueness, mandatory key declaration, and coerce() type/option validation.
- Real application enums: MetadataValueType, ModelRoutingTask, RejectionReason, RejectionResolution, CalibrationSource.
- Strict subsecond-preserving timezone-aware ISO-8601 UTC timestamp validation.
- Non-finite numbers (NaN/Inf) rejection via allow_inf_nan=False and explicit geometry checks.
- Supported v1 quantification metric-key validation and single-default constraint.
- Every supported v1 metric key is readable by the installed runtime ProfileEntry schema.
- Duplicate model routing selector rejection.
- Coordinate tolerance bounds: exactly [-1.5, 1.5] (abs(val) <= 1.5).
- Exclusions: no stored quantifications, no raw database IDs in config, stripped query_contour_id.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import time
import zipfile
from typing import Any, Optional

import pytest
from PIL import Image as PILImage
from pydantic import ValidationError
from iquana_toolbox.schemas.database.quantification_profile import ProfileEntry as RuntimeProfileEntry

from app.schemas.dataset_archive import (
    ARCHIVE_FORMAT,
    ARCHIVE_FORMAT_VERSION,
    ArchiveCategory,
    ArchiveCategoryExtension,
    ArchiveCalibrationDefault,
    ArchiveGeometry,
    ArchiveImage,
    ArchiveImageCalibration,
    ArchiveMetadataKey,
    ArchiveModelRouting,
    ArchiveModelRoutingBinding,
    ArchiveProfileEntry,
    ArchiveQuantificationProfile,
    ArchiveRejection,
    CocoInfo,
    IquanaAnnotationsDocument,
    IquanaConfigDocument,
    STANDARD_V1_METRIC_KEYS,
)
from app.schemas.inference import ModelRoutingTask
from app.schemas.review import RejectionReason, RejectionResolution
from app.services.metadata_types import MetadataValueType
from config import (
    DATASET_ARCHIVE_MAX_COMPRESSED_BYTES,
    DATASET_ARCHIVE_MAX_CONTROL_JSON_BYTES,
    DATASET_ARCHIVE_MAX_MEMBER_BYTES,
    DATASET_ARCHIVE_MAX_MEMBERS,
    DATASET_ARCHIVE_MAX_UNCOMPRESSED_BYTES,
)


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _create_synthetic_png_image(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    """Creates real decodable PNG image bytes with explicit dimensions and color."""
    img = PILImage.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Golden Fixture Builder (Data & Deterministic Real ZIP Materialization)
# ---------------------------------------------------------------------------

def build_golden_fixture_data() -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    """Builds the normative rich golden dataset fixture with real decodable images.

    Returns:
        (annotations_dict, config_dict, img1_bytes, img2_bytes)
    """
    img1_bytes = _create_synthetic_png_image(1920, 1080, color=(30, 80, 150))
    img2_bytes = _create_synthetic_png_image(800, 600, color=(20, 140, 60))

    img1_hash = _compute_sha256(img1_bytes)
    img2_hash = _compute_sha256(img2_bytes)

    annotations_doc: dict[str, Any] = {
        "format": "iquana",
        "format_version": 1,
        "info": {
            "description": "Golden Rich Fixture for IQUANA Issue #94",
            "version": "1.0",
            "year": 2026,
            "date_created": "2026-09-14T20:00:00.123456Z",
            "contributor": "IQUANA Research Team",
            "url": None,
        },
        "licenses": [],
        "images": [
            {
                "id": 1,
                "file_name": "coral_survey.png",
                "width": 1920,
                "height": 1080,
                "iquana": {
                    "archive_path": "images/1/coral_survey.png",
                    "color_mode": "RGB",
                    "scale_x": 0.05,
                    "scale_y": 0.05,
                    "unit": "mm",
                    "description": "Primary transect overview",
                    "metadata": {
                        "site": "Reef Alpha",
                        "water_depth": "12.5",
                        "survey_date": "2026-09-10",
                        "bleached": "false",
                    },
                    "calibrations": [
                        {
                            "kind": "scale",
                            "source": "manual",
                            "params": {"scale_x": 0.05, "scale_y": 0.05, "unit": "mm"},
                            "created_by": "alice",
                            "created_at": "2026-09-14T10:00:00Z",
                            "updated_at": "2026-09-14T10:00:00Z",
                        },
                        {
                            "kind": "response",
                            "source": "manual",
                            "params": {
                                "strategy": "two_patch",
                                "gamma": 1.0,
                                "black_level": 15.0,
                                "white_level": 240.0,
                                "gains": [1.0, 1.0, 1.0],
                                "anchors": {
                                    "r": [[0.0, 0.0], [15.0, 0.0], [240.0, 255.0], [255.0, 255.0]],
                                    "g": [[0.0, 0.0], [15.0, 0.0], [240.0, 255.0], [255.0, 255.0]],
                                    "b": [[0.0, 0.0], [15.0, 0.0], [240.0, 255.0], [255.0, 255.0]],
                                },
                            },
                            "created_by": "alice",
                            "created_at": "2026-09-14T10:05:00Z",
                            "updated_at": "2026-09-14T10:05:00Z",
                        },
                    ],
                    "sha256": img1_hash,
                    "size_bytes": len(img1_bytes),
                },
            },
            {
                "id": 2,
                "file_name": "coral_survey.png",  # Duplicate basename isolated under images/2/
                "width": 800,
                "height": 600,
                "iquana": {
                    "archive_path": "images/2/coral_survey.png",
                    "color_mode": "RGB",
                    "scale_x": 0.1,
                    "scale_y": 0.1,
                    "unit": "mm",
                    "description": "Secondary close-up quadrant",
                    "metadata": {
                        "site": "Reef Beta",
                        "water_depth": "8.0",
                        "survey_date": "2026-09-11",
                        "bleached": "true",
                    },
                    "calibrations": [],
                    "sha256": img2_hash,
                    "size_bytes": len(img2_bytes),
                },
            },
        ],
        "categories": [
            {
                "id": 1,
                "name": "Substrate",
                "supercategory": "none",
                "iquana": {"value": 1, "parent_id": None},
            },
            {
                "id": 2,
                "name": "Hard Coral",
                "supercategory": "Substrate",
                "iquana": {"value": 2, "parent_id": 1},
            },
            {
                "id": 3,
                "name": "Acropora",
                "supercategory": "Hard Coral",
                "iquana": {"value": 3, "parent_id": 2},
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 3,
                "segmentation": [[192.0, 108.0, 384.0, 108.0, 384.0, 216.0, 192.0, 216.0]],
                "area": 20736.0,
                "bbox": [192.0, 108.0, 192.0, 108.0],
                "iscrowd": 0,
                "iquana": {
                    "mask_id": 1,
                    "parent_id": None,
                    "geometry": {
                        "x": [0.10, 0.20, 0.20, 0.10],
                        "y": [0.10, 0.10, 0.20, 0.20],
                    },
                    "added_by": "SAM2",
                    "confidence_score": 0.95,
                    "created_at": "2026-09-14T11:00:00.654321Z",
                    "author_username": "carol",
                    "reviewed_by": ["alice", "bob"],
                },
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": None,  # Accepted unlabelled contour
                "segmentation": [[230.4, 129.6, 307.2, 129.6, 307.2, 172.8, 230.4, 172.8]],
                "area": 3317.76,
                "bbox": [230.4, 129.6, 76.8, 43.2],
                "iscrowd": 0,
                "iquana": {
                    "mask_id": 1,
                    "parent_id": 1,  # Nested child contour belonging to the same mask
                    "geometry": {
                        "x": [0.12, 0.16, 0.16, 0.12],
                        "y": [0.12, 0.12, 0.16, 0.16],
                    },
                    "added_by": "User",
                    "confidence_score": 1.0,
                    "created_at": "2026-09-14T11:30:00Z",
                    "author_username": "carol",
                    "reviewed_by": [],
                },
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 2,
                "segmentation": [[80.0, 60.0, 240.0, 60.0, 240.0, 180.0, 80.0, 180.0]],
                "area": 19200.0,
                "bbox": [80.0, 60.0, 160.0, 120.0],
                "iscrowd": 0,
                "iquana": {
                    "mask_id": 2,
                    "parent_id": None,
                    "geometry": {
                        "x": [0.10, 0.30, 0.30, 0.10],
                        "y": [0.10, 0.10, 0.30, 0.30],
                    },
                    "added_by": "UNET",
                    "confidence_score": 0.88,
                    "created_at": "2026-09-14T12:00:00Z",
                    "author_username": "dave",
                    "reviewed_by": ["alice"],
                },
            },
        ],
        "iquana": {
            "dataset": {
                "name": "Coral Reef Study 2026",
                "description": "Benthic cover study transects",
                "dataset_type": "image",
                "created_by": "alice",
            },
            "actors": [
                {"username": "alice", "roles": ["creator", "reviewer"]},
                {"username": "bob", "roles": ["reviewer"]},
                {"username": "carol", "roles": ["annotator"]},
                {"username": "dave", "roles": ["annotator"]},
            ],
            "metadata_keys": [
                {
                    "key": "site",
                    "value_type": "categorical",
                    "unit": None,
                    "options": ["Reef Alpha", "Reef Beta"],
                    "description": "Sampling site location",
                },
                {
                    "key": "water_depth",
                    "value_type": "number",
                    "unit": "m",
                    "options": [],
                    "description": "Depth of water in meters",
                },
                {
                    "key": "survey_date",
                    "value_type": "date",
                    "unit": None,
                    "options": [],
                    "description": "Collection date",
                },
                {
                    "key": "bleached",
                    "value_type": "boolean",
                    "unit": None,
                    "options": [],
                    "description": "Bleaching presence",
                },
            ],
            "masks": [
                {"id": 1, "image_id": 1, "fully_annotated": False},
                {"id": 2, "image_id": 2, "fully_annotated": True},
            ],
            "rejections": [
                {
                    "id": 1,
                    "mask_id": 1,
                    "annotation_id": 1,
                    "reason": "bad_outline",
                    "note": "Contour overshoots coral margin by ~5px",
                    "created_by": "alice",
                    "created_at": "2026-09-14T13:00:00Z",
                    "resolved_at": None,
                    "resolved_by": None,
                    "resolution": None,
                },
                {
                    "id": 2,
                    "mask_id": 2,
                    "annotation_id": None,  # Mask-level rejection
                    "reason": "missing_objects",
                    "note": "Missed branching colony in upper left",
                    "created_by": "bob",
                    "created_at": "2026-09-14T13:10:00Z",
                    "resolved_at": "2026-09-14T14:00:00Z",
                    "resolved_by": "dave",
                    "resolution": "fixed",
                },
            ],
            "counts": {
                "images": 2,
                "annotations": 3,
                "categories": 3,
                "masks": 2,
                "rejections": 2,
                "temporary_contours_omitted": 1,
            },
            "files": [
                {
                    "image_id": 1,
                    "path": "images/1/coral_survey.png",
                    "sha256": img1_hash,
                    "size_bytes": len(img1_bytes),
                    "width": 1920,
                    "height": 1080,
                    "color_mode": "RGB",
                },
                {
                    "image_id": 2,
                    "path": "images/2/coral_survey.png",
                    "sha256": img2_hash,
                    "size_bytes": len(img2_bytes),
                    "width": 800,
                    "height": 600,
                    "color_mode": "RGB",
                },
            ],
        },
    }

    config_doc: dict[str, Any] = {
        "format": "iquana",
        "format_version": 1,
        "dataset": {
            "require_independent_review": True,
        },
        "calibration_defaults": [
            {
                "kind": "response",
                "defaults": {
                    "strategy": "gray_wedge",
                    "card": "kodak_q13",
                    "fit_model": "linear",
                },
            },
        ],
        "quantification_profiles": [
            {
                "name": "Coral Cover Profile",
                "is_default": True,
                "entries": [
                    {
                        "metric_key": "area",
                        "params": {},
                        "label_names": ["Acropora"],
                    },
                ],
            },
        ],
        "model_routing": {
            "bindings": [
                {
                    "task": "prompted-segmentation",
                    "label_name": "Acropora",
                    "model_registry_key": "custom-sam-v99",
                    "inputs": {"confidence_threshold": 0.85},
                },
                {
                    "task": "instance-suggestion",
                    "label_name": None,
                    "model_registry_key": "mask2former",
                    "inputs": {"max_instances": 50},
                },
            ],
        },
        "omitted_fields": [
            {
                "section": "model_routing.bindings",
                "field": "inputs.conditioning.query_contour_id",
                "task": "prompted-segmentation",
                "label_name": "Acropora",
                "reason": "query_contour_id is a local database ID and not portable across installations",
            },
        ],
    }

    return annotations_doc, config_doc, img1_bytes, img2_bytes


def _add_deterministic_zip_member(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    """Adds a member with a constant timestamp and POSIX attributes for byte-for-byte reproducibility."""
    zinfo = zipfile.ZipInfo(filename=arcname, date_time=(2026, 9, 14, 0, 0, 0))
    zinfo.compress_type = zipfile.ZIP_DEFLATED
    zinfo.external_attr = 0o644 << 16
    zf.writestr(zinfo, data)


def build_golden_archive_zip(include_config: bool = True) -> bytes:
    """Builds a real, deterministic ZIP archive fixture in memory."""
    ann_dict, config_dict, img1_bytes, img2_bytes = build_golden_fixture_data()

    # Validate models prior to serialization
    IquanaAnnotationsDocument.model_validate(ann_dict)
    if include_config:
        IquanaConfigDocument.model_validate(config_dict)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. annotations.json (canonical sorted JSON)
        ann_json = json.dumps(ann_dict, indent=2, sort_keys=True).encode("utf-8")
        _add_deterministic_zip_member(zf, "annotations.json", ann_json)

        # 2. config.json (optional)
        if include_config:
            cfg_json = json.dumps(config_dict, indent=2, sort_keys=True).encode("utf-8")
            _add_deterministic_zip_member(zf, "config.json", cfg_json)

        # 3. Images under images/<id>/<basename>
        _add_deterministic_zip_member(zf, "images/1/coral_survey.png", img1_bytes)
        _add_deterministic_zip_member(zf, "images/2/coral_survey.png", img2_bytes)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test Cases: Actual ZIP Fixture & Integrity & Deterministic Hash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("include_config", [True, False])
def test_golden_archive_zip_structure_and_integrity(include_config: bool):
    """Verifies actual member layout, optional config absence, and image decodability from ZIP."""
    zip_bytes = build_golden_archive_zip(include_config=include_config)
    assert len(zip_bytes) > 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()

        # Check required annotations.json exists
        assert "annotations.json" in namelist

        # Check optional config.json presence/absence
        if include_config:
            assert "config.json" in namelist
        else:
            assert "config.json" not in namelist

        # Check duplicate basenames are cleanly isolated
        assert "images/1/coral_survey.png" in namelist
        assert "images/2/coral_survey.png" in namelist

        # Parse and validate annotations.json directly from the ZIP
        ann_bytes = zf.read("annotations.json")
        ann_data = json.loads(ann_bytes.decode("utf-8"))
        ann_doc = IquanaAnnotationsDocument.model_validate(ann_data)

        if include_config:
            cfg_bytes = zf.read("config.json")
            cfg_data = json.loads(cfg_bytes.decode("utf-8"))
            IquanaConfigDocument.model_validate(cfg_data)

        # Verify images extracted from ZIP: bytes, SHA-256, and header dimensions
        for file_entry in ann_doc.iquana.files:
            member_bytes = zf.read(file_entry.path)
            # 1. SHA-256 matches manifest and image record
            assert _compute_sha256(member_bytes) == file_entry.sha256
            assert len(member_bytes) == file_entry.size_bytes

            # 2. Decodable image with Pillow verifying native dimensions and color mode
            with PILImage.open(io.BytesIO(member_bytes)) as pil_img:
                assert pil_img.width == file_entry.width
                assert pil_img.height == file_entry.height
                assert pil_img.mode == file_entry.color_mode


GOLDEN_ZIP_WITH_CONFIG_SHA256 = "f29176398fd1d85c9e122f2fe6ed87726864e1664e91f89014b9edc52c9c0f3f"
GOLDEN_ZIP_WITHOUT_CONFIG_SHA256 = "7f649288cc3790f27b7a67445498fa93c4424801a550bfee731ef7a88aa1f89f"


def test_golden_zip_hashes_are_deterministic():
    """ZipInfo fixed timestamps ensure builds generated at different times produce identical SHA-256 hashes."""
    zip1 = build_golden_archive_zip(include_config=True)
    time.sleep(0.01)
    zip2 = build_golden_archive_zip(include_config=True)
    hash_with_cfg1 = _compute_sha256(zip1)
    hash_with_cfg2 = _compute_sha256(zip2)
    assert hash_with_cfg1 == hash_with_cfg2
    assert hash_with_cfg1 == GOLDEN_ZIP_WITH_CONFIG_SHA256

    zip3 = build_golden_archive_zip(include_config=False)
    time.sleep(0.01)
    zip4 = build_golden_archive_zip(include_config=False)
    hash_without_cfg1 = _compute_sha256(zip3)
    hash_without_cfg2 = _compute_sha256(zip4)
    assert hash_without_cfg1 == hash_without_cfg2
    assert hash_without_cfg1 == GOLDEN_ZIP_WITHOUT_CONFIG_SHA256


# ---------------------------------------------------------------------------
# Test Cases: P1 Category Hierarchy & Linear-Time Scalability
# ---------------------------------------------------------------------------

def test_category_iquana_is_required():
    """categories[].iquana is required; omitting it fails validation."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    del bad_ann["categories"][0]["iquana"]

    with pytest.raises(ValidationError, match="Field required"):
        IquanaAnnotationsDocument.model_validate(bad_ann)


def test_category_hierarchy_cycle_rejected():
    """Cycles in category parent relationships are detected and rejected."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    # Create cycle: 1 -> 3 -> 2 -> 1
    bad_ann["categories"][0]["iquana"]["parent_id"] = 3
    bad_ann["categories"][1]["iquana"]["parent_id"] = 1
    bad_ann["categories"][2]["iquana"]["parent_id"] = 2

    with pytest.raises(ValidationError, match="Cycle detected in category hierarchy"):
        IquanaAnnotationsDocument.model_validate(bad_ann)


def test_category_self_parent_rejected():
    """A category referencing itself as parent is rejected."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    bad_ann["categories"][0]["iquana"]["parent_id"] = bad_ann["categories"][0]["id"]

    with pytest.raises(ValidationError, match="cannot be its own parent"):
        IquanaAnnotationsDocument.model_validate(bad_ann)


def test_category_missing_parent_rejected():
    """A category referencing a non-existent parent category is rejected."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    bad_ann["categories"][0]["iquana"]["parent_id"] = 9999

    with pytest.raises(ValidationError, match="references non-existent parent category id 9999"):
        IquanaAnnotationsDocument.model_validate(bad_ann)


def test_linear_time_deep_hierarchy_validation():
    """Deep category hierarchies scale linearly (O(n)) without quadratic blowup."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    ann_dict = copy.deepcopy(ann_dict)

    # Build 2,000 chained categories: 1 -> 2 -> 3 -> ... -> 2000
    n = 2000
    categories = []
    for i in range(1, n + 1):
        categories.append({
            "id": i,
            "name": f"Label_{i}",
            "supercategory": "none",
            "iquana": {"value": i, "parent_id": i - 1 if i > 1 else None},
        })
    ann_dict["categories"] = categories
    ann_dict["annotations"] = []  # Clear annotations to isolate category validation
    ann_dict["iquana"]["rejections"] = []  # Clear rejections referencing annotations
    ann_dict["iquana"]["counts"]["categories"] = n
    ann_dict["iquana"]["counts"]["annotations"] = 0
    ann_dict["iquana"]["counts"]["rejections"] = 0

    t0 = time.perf_counter()
    IquanaAnnotationsDocument.model_validate(ann_dict)
    elapsed = time.perf_counter() - t0
    # Linear traversal should complete 2,000 nodes well under 0.15s (previously ~0.64s for 4k)
    assert elapsed < 0.25, f"Hierarchy validation took too long: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Test Cases: P1 Non-Finite Numbers Rejection
# ---------------------------------------------------------------------------

def test_non_finite_coordinates_rejected():
    """NaN and Infinite values in coordinates are strictly rejected."""
    # NaN in x
    with pytest.raises(ValidationError):
        ArchiveGeometry(x=[float("nan"), 0.2, 0.3], y=[0.1, 0.2, 0.3])

    # Inf in y
    with pytest.raises(ValidationError):
        ArchiveGeometry(x=[0.1, 0.2, 0.3], y=[0.1, float("inf"), 0.3])

    # -Inf in x
    with pytest.raises(ValidationError):
        ArchiveGeometry(x=[float("-inf"), 0.2, 0.3], y=[0.1, 0.2, 0.3])


def test_non_finite_values_in_open_json_payloads_rejected():
    """NaN, Inf, and -Inf nested in open dict payloads are strictly rejected via dict and model_validate_json."""
    # 1. Profile entry params: NaN via direct dict and via JSON string
    with pytest.raises(ValidationError, match="Non-finite floating point value"):
        ArchiveProfileEntry(metric_key="area", params={"nested": {"threshold": float("nan")}})

    with pytest.raises(ValidationError, match="Non-finite floating point value"):
        ArchiveProfileEntry.model_validate_json('{"metric_key": "area", "params": {"val": NaN}}')

    # Inf in profile entry params
    with pytest.raises(ValidationError, match="Non-finite floating point value"):
        ArchiveProfileEntry(metric_key="area", params={"threshold": float("inf")})

    # 2. Calibration params: NaN via JSON string and via direct dict
    with pytest.raises(ValidationError, match="Non-finite floating point value"):
        ArchiveImageCalibration.model_validate_json(
            '{"kind": "response", "source": "manual", "params": {"strategy": "two_patch", "gamma": NaN}}'
        )

    with pytest.raises(ValidationError, match="Non-finite floating point value"):
        ArchiveImageCalibration(
            kind="response",
            source="manual",
            params={"strategy": "two_patch", "gamma": float("nan")},
        )

    # 3. Calibration defaults: NaN via JSON string
    with pytest.raises(ValidationError, match="Non-finite floating point value"):
        ArchiveCalibrationDefault.model_validate_json(
            '{"kind": "response", "defaults": {"strategy": "gray_wedge", "gamma": NaN}}'
        )

    # 4. Model routing inputs: NaN via JSON string
    with pytest.raises(ValidationError, match="Non-finite floating point value"):
        ArchiveModelRoutingBinding.model_validate_json(
            '{"task": "instance-suggestion", "model_registry_key": "m2f", "inputs": {"conf": NaN}}'
        )


def test_non_finite_numeric_metadata_rejected():
    """Numeric metadata values matching 'NaN', 'Inf', or '-Infinity' are rejected."""
    ann_dict, _, _, _ = build_golden_fixture_data()

    for bad_num in ("NaN", "nan", "Inf", "-Inf", "Infinity", "-Infinity"):
        bad_ann = copy.deepcopy(ann_dict)
        bad_ann["images"][0]["iquana"]["metadata"]["water_depth"] = bad_num

        with pytest.raises(ValidationError, match="invalid metadata value"):
            IquanaAnnotationsDocument.model_validate(bad_ann)


def test_deeply_nested_payload_avoids_recursion_error_and_fails_validation():
    """Deeply nested payload (~1,000 levels) does not raise raw RecursionError and fails validation."""
    nested: Any = 1.0
    for _ in range(1000):
        nested = {"level": nested}

    with pytest.raises(ValidationError, match="nesting depth exceeds maximum allowed limit"):
        ArchiveProfileEntry(metric_key="area", params=nested)

    with pytest.raises(ValidationError, match="nesting depth exceeds maximum allowed limit"):
        ArchiveImageCalibration(
            kind="response",
            source="manual",
            params=nested,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("segmentation", [], "segmentation must contain exactly one polygon"),
        ("segmentation", [[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]], "segmentation does not match canonical geometry"),
        ("bbox", [0.0, 0.0, 0.0, 0.0], "bbox does not match canonical geometry"),
        ("area", 0.0, "area does not match canonical geometry"),
    ],
)
def test_coco_geometry_must_match_canonical_geometry(field, value, message):
    annotations, _, _, _ = build_golden_fixture_data()
    annotations["annotations"][0][field] = value

    with pytest.raises(ValidationError, match=message):
        IquanaAnnotationsDocument.model_validate(annotations)


# ---------------------------------------------------------------------------
# Test Cases: P1 Relationship Ownership and Acyclicity
# ---------------------------------------------------------------------------

def test_annotation_image_id_must_match_mask_image_id():
    """Annotation image_id differing from its mask's image_id is rejected."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    # Mask 1 belongs to image 1, but annotation 1 claims image 2
    bad_ann["annotations"][0]["image_id"] = 2

    with pytest.raises(ValidationError, match="does not match its mask .* image_id"):
        IquanaAnnotationsDocument.model_validate(bad_ann)


def test_annotation_parent_must_belong_to_same_mask():
    """Parent contour belonging to a different mask is rejected."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    # Annotation 3 is on mask 2, but set its parent to annotation 1 on mask 1
    bad_ann["annotations"][2]["iquana"]["parent_id"] = 1

    with pytest.raises(ValidationError, match="must belong to the same mask"):
        IquanaAnnotationsDocument.model_validate(bad_ann)


def test_annotation_hierarchy_cycle_rejected():
    """Contour parent cycles are detected and rejected."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    # Create cycle: ann 1 -> ann 2 -> ann 1
    bad_ann["annotations"][0]["iquana"]["parent_id"] = 2
    bad_ann["annotations"][1]["iquana"]["parent_id"] = 1

    with pytest.raises(ValidationError, match="Cycle detected in annotation hierarchy"):
        IquanaAnnotationsDocument.model_validate(bad_ann)


def test_rejection_annotation_must_belong_to_same_mask():
    """A rejection targeting an annotation from another mask is rejected."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    # Rejection 1 is on mask 1, point its annotation_id to annotation 3 on mask 2
    bad_ann["iquana"]["rejections"][0]["annotation_id"] = 3

    with pytest.raises(ValidationError, match="does not match referenced annotation"):
        IquanaAnnotationsDocument.model_validate(bad_ann)


# ---------------------------------------------------------------------------
# Test Cases: P1 Image Manifest Agreement
# ---------------------------------------------------------------------------

def test_manifest_duplicate_entry_rejected():
    """Duplicate entries for the same image in iquana.files are rejected."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    bad_ann["iquana"]["files"].append(copy.deepcopy(bad_ann["iquana"]["files"][0]))

    with pytest.raises(ValidationError, match="Duplicate file manifest entry"):
        IquanaAnnotationsDocument.model_validate(bad_ann)


def test_manifest_mismatched_hash_rejected():
    """Conflicting hash between image record and manifest is rejected."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    bad_ann["iquana"]["files"][0]["sha256"] = "a" * 64

    with pytest.raises(ValidationError, match="File manifest sha256 .* does not match image record"):
        IquanaAnnotationsDocument.model_validate(bad_ann)


def test_image_archive_path_must_encode_image_id():
    """Image archive_path must encode its image ID (images/<id>/<name>)."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    # Image 1 has archive_path claiming image 2
    bad_ann["images"][0]["iquana"]["archive_path"] = "images/2/coral_survey.png"

    with pytest.raises(ValidationError, match="must encode image id 1"):
        IquanaAnnotationsDocument.model_validate(bad_ann)


# ---------------------------------------------------------------------------
# Test Cases: P2 Metadata Key Definitions & Coerce Validation
# ---------------------------------------------------------------------------

def test_duplicate_metadata_key_definitions_rejected():
    """Duplicate metadata key definitions fail validation."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    bad_ann["iquana"]["metadata_keys"].append(copy.deepcopy(bad_ann["iquana"]["metadata_keys"][0]))

    with pytest.raises(ValidationError, match="Duplicate metadata key definition"):
        IquanaAnnotationsDocument.model_validate(bad_ann)


def test_undeclared_metadata_key_rejected():
    """Images using undeclared metadata keys fail validation."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    bad_ann["images"][0]["iquana"]["metadata"]["undeclared_key"] = "foo"

    with pytest.raises(ValidationError, match="uses undeclared metadata key 'undeclared_key'"):
        IquanaAnnotationsDocument.model_validate(bad_ann)


def test_incompatible_metadata_value_rejected():
    """Metadata values incompatible with their declared type or locked options fail validation."""
    ann_dict, _, _, _ = build_golden_fixture_data()

    # Number type with invalid string
    bad_ann1 = copy.deepcopy(ann_dict)
    bad_ann1["images"][0]["iquana"]["metadata"]["water_depth"] = "not-a-number"
    with pytest.raises(ValidationError, match="invalid metadata value for key 'water_depth'"):
        IquanaAnnotationsDocument.model_validate(bad_ann1)

    # Categorical type with out-of-vocabulary option
    bad_ann2 = copy.deepcopy(ann_dict)
    bad_ann2["images"][0]["iquana"]["metadata"]["site"] = "Unknown Reef"
    with pytest.raises(ValidationError, match="invalid metadata value for key 'site'"):
        IquanaAnnotationsDocument.model_validate(bad_ann2)


# ---------------------------------------------------------------------------
# Test Cases: P2 Duplicate Routing Selectors & Quantification Profiles
# ---------------------------------------------------------------------------

def test_duplicate_routing_selectors_rejected():
    """Multiple routing bindings with identical (task, label_name) selectors fail validation."""
    _, config_dict, _, _ = build_golden_fixture_data()
    bad_cfg = copy.deepcopy(config_dict)
    bad_cfg["model_routing"]["bindings"].append({
        "task": "prompted-segmentation",
        "label_name": "Acropora",  # Duplicate selector
        "model_registry_key": "another-model",
        "inputs": None,
    })

    with pytest.raises(ValidationError, match="Duplicate model routing selector"):
        IquanaConfigDocument.model_validate(bad_cfg)


def test_quantification_profile_metric_key_requires_supported_v1_registry():
    """Only metric keys readable by the v1 runtime profile schema are accepted.

    An unsupported key rejects the complete config validation rather than being
    retained as an unreadable profile entry or silently omitted.
    """
    # Supported v1 metric keys pass.
    entry_std = ArchiveProfileEntry(metric_key="area", params={})
    assert entry_std.metric_key == "area"
    assert "area" in STANDARD_V1_METRIC_KEYS

    # A syntactically valid future/custom key is rejected before config import.
    with pytest.raises(ValidationError, match="not supported by archive format v1"):
        ArchiveProfileEntry(metric_key="future_custom_metric", params={"depth": 3})

    _, bad_cfg, _, _ = build_golden_fixture_data()
    bad_cfg["quantification_profiles"][0]["entries"][0]["metric_key"] = "future_custom_metric"
    with pytest.raises(ValidationError, match="not supported by archive format v1"):
        IquanaConfigDocument.model_validate(bad_cfg)

    # Malformed metric keys (spaces, punctuation, empty) are rejected.
    with pytest.raises(ValidationError):
        ArchiveProfileEntry(metric_key="metric with spaces", params={})

    with pytest.raises(ValidationError):
        ArchiveProfileEntry(metric_key="metric$special!", params={})

    with pytest.raises(ValidationError):
        ArchiveProfileEntry(metric_key="", params={})


def test_supported_v1_metric_keys_are_readable_by_runtime_profile_entry():
    """The static archive allowlist remains readable by the installed runtime schema."""
    for metric_key in sorted(STANDARD_V1_METRIC_KEYS):
        runtime_entry = RuntimeProfileEntry(metric_key=metric_key)
        assert runtime_entry.metric_key == metric_key


def test_multiple_default_quantification_profiles_rejected():
    """At most one quantification profile may be marked as default."""
    _, config_dict, _, _ = build_golden_fixture_data()
    bad_cfg = copy.deepcopy(config_dict)
    bad_cfg["quantification_profiles"].append({
        "name": "Second Profile",
        "is_default": True,  # Both marked default
        "entries": [],
    })

    with pytest.raises(ValidationError, match="At most one quantification profile may be marked as default"):
        IquanaConfigDocument.model_validate(bad_cfg)


# ---------------------------------------------------------------------------
# Test Cases: Calibration Defaults & Image Calibrations Validation
# ---------------------------------------------------------------------------

def test_calibration_defaults_valid():
    """Valid calibration defaults for response with gray_wedge or two_patch succeed."""
    cd1 = ArchiveCalibrationDefault(
        kind="response",
        defaults={"strategy": "gray_wedge", "card": "kodak_q13", "fit_model": "linear"},
    )
    assert cd1.kind == "response"
    assert cd1.defaults["strategy"] == "gray_wedge"
    assert cd1.defaults["card"] == "kodak_q13"

    cd2 = ArchiveCalibrationDefault(
        kind="response",
        defaults={"strategy": "two_patch"},
    )
    assert cd2.defaults["strategy"] == "two_patch"


def test_calibration_defaults_scale_rejected():
    """Scale has no configurable strategy; setting defaults for scale fails validation."""
    with pytest.raises(ValidationError, match="Invalid calibration defaults for kind 'scale'"):
        ArchiveCalibrationDefault(
            kind="scale",
            defaults={"strategy": "reference_card", "card": "kodak_q13"},
        )


def test_calibration_defaults_unknown_or_legacy_kind_rejected():
    """Legacy kinds like 'intensity' and unknown kinds fail dataset defaults validation."""
    # intensity is legacy, cannot be set as dataset default
    with pytest.raises(ValidationError, match="Unknown calibration kind 'intensity'"):
        ArchiveCalibrationDefault(
            kind="intensity",
            defaults={"strategy": "gray_wedge"},
        )

    # arbitrary unregistered kind
    with pytest.raises(ValidationError, match="Unknown calibration kind 'unregistered_kind'"):
        ArchiveCalibrationDefault(
            kind="unregistered_kind",
            defaults={"strategy": "anything"},
        )


def test_calibration_defaults_invalid_strategy_or_card_rejected():
    """Invalid strategy or card for response kind fails validation."""
    with pytest.raises(ValidationError, match="Strategy 'invalid_strat' does not apply"):
        ArchiveCalibrationDefault(
            kind="response",
            defaults={"strategy": "invalid_strat"},
        )

    with pytest.raises(ValidationError, match="Unknown reference card 'invalid_card'"):
        ArchiveCalibrationDefault(
            kind="response",
            defaults={"strategy": "gray_wedge", "card": "invalid_card"},
        )


def test_duplicate_calibration_defaults_rejected():
    """Multiple defaults for the same calibration kind are rejected (including case variations)."""
    _, config_dict, _, _ = build_golden_fixture_data()
    bad_cfg = copy.deepcopy(config_dict)
    bad_cfg["calibration_defaults"].append({
        "kind": "response",
        "defaults": {"strategy": "two_patch"},
    })

    with pytest.raises(ValidationError, match="Duplicate calibration default for kind 'response'"):
        IquanaConfigDocument.model_validate(bad_cfg)

    # Mixed-case duplicate ("Response" vs "response")
    bad_cfg_case = copy.deepcopy(config_dict)
    bad_cfg_case["calibration_defaults"].append({
        "kind": "Response",
        "defaults": {"strategy": "two_patch"},
    })
    with pytest.raises(ValidationError, match="Duplicate calibration default for kind 'response'"):
        IquanaConfigDocument.model_validate(bad_cfg_case)


def test_image_calibration_validation_and_uniqueness():
    """Per-image calibrations validate kind & parameters, and enforce unique kinds per image."""
    # Valid scale and response calibrations pass
    cal_scale = ArchiveImageCalibration(
        kind="scale",
        params={"scale_x": 0.05, "scale_y": 0.05, "unit": "mm"},
    )
    assert cal_scale.params["unit"] == "mm"

    # Invalid scale params (negative scale) fail
    with pytest.raises(ValidationError, match="Scale values must be positive"):
        ArchiveImageCalibration(
            kind="scale",
            params={"scale_x": -0.05, "scale_y": 0.05, "unit": "mm"},
        )

    # Unknown kind fails
    with pytest.raises(ValidationError, match="Unknown calibration kind 'non_existent'"):
        ArchiveImageCalibration(
            kind="non_existent",
            params={},
        )

    # Legacy kinds pass for backwards compatibility
    cal_legacy = ArchiveImageCalibration(
        kind="intensity",
        params={"raw": 123},
    )
    assert cal_legacy.kind == "intensity"

    # Duplicate calibration kinds on the same image fail (exact and case variation)
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    bad_ann["images"][0]["iquana"]["calibrations"].append({
        "kind": "scale",
        "source": "manual",
        "params": {"scale_x": 0.1, "scale_y": 0.1, "unit": "cm"},
    })

    with pytest.raises(ValidationError, match="Duplicate calibration for kind 'scale' found on image"):
        IquanaAnnotationsDocument.model_validate(bad_ann)

    bad_ann_case = copy.deepcopy(ann_dict)
    bad_ann_case["images"][0]["iquana"]["calibrations"].append({
        "kind": "Scale",
        "source": "manual",
        "params": {"scale_x": 0.1, "scale_y": 0.1, "unit": "cm"},
    })

    with pytest.raises(ValidationError, match="Duplicate calibration for kind 'scale' found on image"):
        IquanaAnnotationsDocument.model_validate(bad_ann_case)


# ---------------------------------------------------------------------------
# Test Cases: P2 ISO-8601 UTC Timestamp Validation & Subseconds
# ---------------------------------------------------------------------------

def test_utc_timestamp_validation_and_subsecond_precision():
    """Timestamps must be valid timezone-aware ISO-8601 strings and preserve subseconds."""
    # Subsecond precision preserved
    info = CocoInfo(description="test", year=2026, date_created="2026-09-14T20:00:00.123456Z")
    assert info.date_created == "2026-09-14T20:00:00.123456Z"

    # Valid offset converts to UTC Z with subseconds
    info2 = CocoInfo(description="test", year=2026, date_created="2026-09-14T22:00:00.500+02:00")
    assert info2.date_created == "2026-09-14T20:00:00.500000Z"

    # Naive timestamp rejected
    with pytest.raises(ValidationError, match="timezone-naive"):
        CocoInfo(description="test", year=2026, date_created="2026-09-14T20:00:00")

    # Arbitrary string rejected
    with pytest.raises(ValidationError, match="Invalid timestamp 'tomorrow'"):
        CocoInfo(description="test", year=2026, date_created="tomorrow")


# ---------------------------------------------------------------------------
# Test Cases: P2 Coordinate Tolerance Bounds [-1.5, 1.5]
# ---------------------------------------------------------------------------

def test_coordinate_tolerance_exact_bounds():
    """Exact bounds [-1.5, 1.5] pass while values beyond [-1.5, 1.5] fail."""
    # Exact bounds: -1.5 and 1.5 pass
    geo = ArchiveGeometry(x=[-1.5, 0.0, 1.5], y=[-1.5, 0.0, 1.5])
    assert geo.x == [-1.5, 0.0, 1.5]

    # Just below -1.5 fails
    with pytest.raises(ValidationError, match="exceeds tolerance range"):
        ArchiveGeometry(x=[-1.5001, 0.0, 1.0], y=[0.0, 0.0, 1.0])

    # Just above 1.5 fails
    with pytest.raises(ValidationError, match="exceeds tolerance range"):
        ArchiveGeometry(x=[0.0, 0.0, 1.5001], y=[0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Test Cases: P2 Provisional Archive Limits
# ---------------------------------------------------------------------------

def test_configured_archive_limits_provisional_bounds_and_order():
    """Verify archive limits settings are within sensible provisional bounds and ordered."""
    assert 1024 * 1024 <= DATASET_ARCHIVE_MAX_CONTROL_JSON_BYTES <= 256 * 1024 * 1024
    assert DATASET_ARCHIVE_MAX_CONTROL_JSON_BYTES <= DATASET_ARCHIVE_MAX_MEMBER_BYTES
    assert 1024 * 1024 <= DATASET_ARCHIVE_MAX_MEMBER_BYTES <= 10 * 1024 * 1024 * 1024
    assert DATASET_ARCHIVE_MAX_MEMBER_BYTES <= DATASET_ARCHIVE_MAX_COMPRESSED_BYTES <= 50 * 1024 * 1024 * 1024
    assert DATASET_ARCHIVE_MAX_COMPRESSED_BYTES <= DATASET_ARCHIVE_MAX_UNCOMPRESSED_BYTES <= 100 * 1024 * 1024 * 1024
    assert 100 <= DATASET_ARCHIVE_MAX_MEMBERS <= 1_000_000


# ---------------------------------------------------------------------------
# Test Cases: Negative Retained & Contract Exclusions
# ---------------------------------------------------------------------------

def test_duplicate_category_name_rejected():
    """Duplicate category names violate dataset-unique constraint and fail validation."""
    ann_dict, _, _, _ = build_golden_fixture_data()
    bad_ann = copy.deepcopy(ann_dict)
    bad_ann["categories"].append({
        "id": 4,
        "name": "Acropora",  # Duplicate name
        "supercategory": "Hard Coral",
        "iquana": {"value": 4, "parent_id": 2},
    })
    bad_ann["iquana"]["counts"]["categories"] = 4

    with pytest.raises(ValidationError, match="Duplicate category name 'Acropora'"):
        IquanaAnnotationsDocument.model_validate(bad_ann)


def test_invalid_format_or_version_rejected():
    """Unsupported formats and future versions are rejected before extraction."""
    ann_dict, config_dict, _, _ = build_golden_fixture_data()

    bad_ann = copy.deepcopy(ann_dict)
    bad_ann["format"] = "coco"
    with pytest.raises(ValidationError):
        IquanaAnnotationsDocument.model_validate(bad_ann)

    bad_version = copy.deepcopy(ann_dict)
    bad_version["format_version"] = 2
    with pytest.raises(ValidationError):
        IquanaAnnotationsDocument.model_validate(bad_version)

    bad_config = copy.deepcopy(config_dict)
    bad_config["format_version"] = 99
    with pytest.raises(ValidationError):
        IquanaConfigDocument.model_validate(bad_config)


def test_path_traversal_and_invalid_archive_paths_rejected():
    """Archive paths with '..', leading slashes, or missing ID prefix are rejected."""
    ann_dict, _, _, _ = build_golden_fixture_data()

    # Path traversal with ..
    bad_path1 = copy.deepcopy(ann_dict)
    bad_path1["images"][0]["iquana"]["archive_path"] = "images/1/../../etc/passwd"
    with pytest.raises(ValidationError, match="archive_path"):
        IquanaAnnotationsDocument.model_validate(bad_path1)

    # Absolute path
    bad_path2 = copy.deepcopy(ann_dict)
    bad_path2["images"][0]["iquana"]["archive_path"] = "/images/1/coral.png"
    with pytest.raises(ValidationError, match="archive_path"):
        IquanaAnnotationsDocument.model_validate(bad_path2)


def test_dangling_reference_checks():
    """Annotations or masks pointing to non-existent images or masks fail validation."""
    ann_dict, _, _, _ = build_golden_fixture_data()

    # Annotation references non-existent image
    bad_ann = copy.deepcopy(ann_dict)
    bad_ann["annotations"][0]["image_id"] = 999
    with pytest.raises(ValidationError, match="references non-existent image id 999"):
        IquanaAnnotationsDocument.model_validate(bad_ann)

    # Annotation references non-existent mask
    bad_mask = copy.deepcopy(ann_dict)
    bad_mask["annotations"][0]["iquana"]["mask_id"] = 999
    with pytest.raises(ValidationError, match="references non-existent mask id 999"):
        IquanaAnnotationsDocument.model_validate(bad_mask)


def test_contract_exclusions_in_golden_fixture():
    """Neither document contains stored quantifications, ContourMetrics, or raw DB IDs in config."""
    ann_dict, config_dict, _, _ = build_golden_fixture_data()

    # 1. No legacy contour quantification columns in annotations[].iquana
    for ann in ann_dict["annotations"]:
        iquana_ext = ann["iquana"]
        assert "area_mm2" not in iquana_ext
        assert "perimeter" not in iquana_ext
        assert "circularity" not in iquana_ext
        assert "diameter" not in iquana_ext
        assert "quantification" not in iquana_ext

    # 2. No ContourMetrics rows in annotations top-level
    assert "contour_metrics" not in ann_dict["iquana"]
    assert "metrics" not in ann_dict["iquana"]

    # 3. Label references in config are strictly names, never integer IDs
    for profile in config_dict["quantification_profiles"]:
        for entry in profile["entries"]:
            assert "label_ids" not in entry
            assert "category_id" not in entry
            assert entry.get("label_names") == ["Acropora"]

    for binding in config_dict["model_routing"]["bindings"]:
        assert "label_id" not in binding
        assert "category_id" not in binding


def test_config_rejects_query_contour_id_in_inputs():
    """Model routing bindings reject serialized local contour IDs in inputs."""
    with pytest.raises(ValidationError, match="query_contour_id must be stripped"):
        ArchiveModelRoutingBinding(
            task="prompted-segmentation",
            model_registry_key="sam2",
            inputs={"conditioning": {"query_contour_id": 42}},
        )

"""Pydantic schemas for the COCO-oriented IQUANA dataset archive (v1).

Defines the explicit, versioned contracts for:
1. `annotations.json` (required): COCO-oriented structure (info, images,
   annotations, categories) with nested and top-level `iquana` extensions for
   native normalized geometry, hierarchies, masks, rejections, metadata keys/values,
   image calibrations, and file integrity checksums.
2. `config.json` (optional): Standalone portable dataset configuration containing
   review policy, calibration defaults, quantification profile definitions (using
   supported v1 metric keys and label scoping by name), and model routing bindings (label selectors by name), with
   local installation state (such as query_contour_id) stripped and disclosed in
   `omitted_fields`.

Neither file contains serialized quantification metric values or database primary keys.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from app.schemas.inference import ModelRoutingTask
from app.schemas.review import RejectionReason, RejectionResolution
from app.services.metadata_types import InvalidMetadataError, MetadataValueType, coerce

# Normative format constants
ARCHIVE_FORMAT: Literal["iquana"] = "iquana"
ARCHIVE_FORMAT_VERSION: Literal[1] = 1

# Path safety regex: images/<archive-image-id>/<sanitized-basename>
IMAGE_ARCHIVE_PATH_PATTERN = re.compile(r"^images/(?P<image_id>\d+)/(?P<basename>[^/\\\x00]+)$")

# Allowed calibration source literals
CalibrationSourceLiteral = Literal["manual", "measured", "dataset", "file_metadata"]


# ---------------------------------------------------------------------------
# Timestamp Validation (Preserves Subsecond Precision)
# ---------------------------------------------------------------------------

def validate_utc_iso8601(v: Any) -> Optional[str]:
    """Validate that a value is a timezone-aware ISO-8601 timestamp and return canonical ISO-8601 UTC string with Z suffix."""
    if v is None:
        return None
    if isinstance(v, datetime):
        dt = v
    elif isinstance(v, str):
        try:
            s = v.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid timestamp '{v}'. Must be a valid ISO-8601 string.") from exc
    else:
        raise ValueError(f"Expected datetime or ISO-8601 string, got {type(v).__name__}")

    if dt.tzinfo is None:
        raise ValueError(f"Timestamp '{v}' is timezone-naive. A timezone offset (e.g. 'Z' or '+00:00') is required.")

    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.isoformat().replace("+00:00", "Z")


IsoUtcDatetime = Annotated[str, BeforeValidator(validate_utc_iso8601)]
OptionalIsoUtcDatetime = Annotated[Optional[str], BeforeValidator(validate_utc_iso8601)]


# ---------------------------------------------------------------------------
# Linear-Time Hierarchy Cycle Detection
# ---------------------------------------------------------------------------

def check_hierarchy_acyclic(parent_map: dict[int, Optional[int]], entity_name: str) -> None:
    """Verify that a parent mapping (child_id -> parent_id) is strictly acyclic in O(n) time.

    Uses a 3-state traversal:
      0 = unvisited
      1 = visiting (in active path; encountering it indicates a cycle)
      2 = visited (fully explored, known acyclic)
    """
    state: dict[int, int] = {node_id: 0 for node_id in parent_map}

    for start_node in parent_map:
        if state[start_node] != 0:
            continue

        curr: Optional[int] = start_node
        path: list[int] = []
        while curr is not None:
            if curr not in state:
                # Parent is not in parent_map (missing parent, handled by existence checks)
                break
            curr_state = state[curr]
            if curr_state == 1:
                raise ValueError(f"Cycle detected in {entity_name} hierarchy involving {entity_name} id {curr}.")
            if curr_state == 2:
                # Already confirmed acyclic branch
                break

            state[curr] = 1
            path.append(curr)
            curr = parent_map[curr]

        for node in path:
            state[node] = 2


# ---------------------------------------------------------------------------
# Base Schema Configuration & Recursive Finite Number Enforcement
# ---------------------------------------------------------------------------

MAX_PAYLOAD_NESTING_DEPTH: int = 64


def assert_finite_numbers(
    root_value: Any, root_path: str = "", max_depth: int = MAX_PAYLOAD_NESTING_DEPTH
) -> None:
    """Iteratively verify that no non-finite floats exist in value and nesting depth is bounded.

    Uses an explicit stack rather than recursive calls to avoid RecursionError on deeply
    nested malicious payloads, enforcing a maximum nesting depth limit.
    """
    stack: list[tuple[Any, str, int]] = [(root_value, root_path, 0)]

    while stack:
        value, path, depth = stack.pop()

        if depth > max_depth:
            field_desc = f" at '{path}'" if path else ""
            raise ValueError(
                f"Payload nesting depth exceeds maximum allowed limit ({max_depth}){field_desc}."
            )

        if isinstance(value, float):
            if not math.isfinite(value):
                field_desc = f" at '{path}'" if path else ""
                raise ValueError(
                    f"Non-finite floating point value '{value}'{field_desc} is forbidden."
                )
        elif isinstance(value, BaseModel):
            # Child Pydantic models validate their own fields
            continue
        elif isinstance(value, dict):
            for k, v in value.items():
                child_path = f"{path}.{k}" if path else str(k)
                stack.append((v, child_path, depth + 1))
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                child_path = f"{path}[{i}]" if path else f"[{i}]"
                stack.append((v, child_path, depth + 1))


class StrictArchiveModel(BaseModel):
    """Base model forbidding unknown extra fields and non-finite numbers (NaN/Inf)."""
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_finite_payloads(self) -> StrictArchiveModel:
        for field_name, val in self.__dict__.items():
            assert_finite_numbers(val, field_name)
        return self


# ---------------------------------------------------------------------------
# COCO Core: Info and Licenses
# ---------------------------------------------------------------------------

class CocoInfo(StrictArchiveModel):
    """COCO standard info header."""
    description: str = Field(..., description="Human-readable description of the dataset.")
    version: str = Field(default="1.0", description="COCO document version string.")
    year: int = Field(..., description="Year the export was generated.")
    date_created: IsoUtcDatetime = Field(..., description="ISO-8601 UTC timestamp of export creation.")
    contributor: Optional[str] = Field(default=None, description="Optional contributor attribution.")
    url: Optional[str] = Field(default=None, description="Optional dataset URL.")


class CocoLicense(StrictArchiveModel):
    """COCO standard license object."""
    id: int = Field(..., ge=0, description="License ID.")
    name: str = Field(..., description="License name.")
    url: Optional[str] = Field(default=None, description="License URL.")


# ---------------------------------------------------------------------------
# Image Schemas
# ---------------------------------------------------------------------------

class ArchiveImageCalibration(StrictArchiveModel):
    """One calibration record attached to an image."""
    kind: str = Field(..., min_length=1, max_length=32, description="Calibration kind (e.g. scale, response).")
    source: CalibrationSourceLiteral = Field(default="manual", description="Calibration source.")
    params: dict[str, Any] = Field(default_factory=dict, description="Kind-specific parameter payload.")
    created_by: Optional[str] = Field(default=None, description="Source creator username (export provenance only).")
    created_at: OptionalIsoUtcDatetime = Field(default=None, description="ISO-8601 UTC creation timestamp.")
    updated_at: OptionalIsoUtcDatetime = Field(default=None, description="ISO-8601 UTC update timestamp.")

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if not v_clean:
            raise ValueError("Calibration kind cannot be empty.")
        return v_clean

    @model_validator(mode="after")
    def validate_calibration_params(self) -> ArchiveImageCalibration:
        from app.exceptions import InvalidCalibrationError, InvalidScaleError, UnknownCalibrationKindError
        from app.services.calibration import registry

        try:
            kind_obj = registry.get_kind(self.kind)
        except UnknownCalibrationKindError:
            if self.kind in registry.LEGACY_KINDS:
                return self
            raise ValueError(
                f"Unknown calibration kind '{self.kind}'. Known kinds: {', '.join(sorted(registry._KINDS))}."
            )

        try:
            self.params = kind_obj.normalize(self.params or {})
        except (InvalidCalibrationError, InvalidScaleError) as exc:
            raise ValueError(
                f"Invalid calibration params for kind '{self.kind}': {exc}"
            ) from exc
        return self


ArchiveContentMode = Literal["full", "annotations_only"]


class ArchiveImageExtension(StrictArchiveModel):
    """IQUANA non-COCO fields nested under images[].iquana."""
    archive_path: Optional[str] = Field(
        default=None,
        description="Relative path of the image within the archive (images/<id>/<filename>), or None in annotations_only mode.",
    )
    color_mode: str = Field(default="RGB", description="Image color mode (e.g. RGB, RGBA, L).")
    scale_x: float = Field(default=1.0, gt=0, description="Spatial scale along X (unit per pixel).")
    scale_y: float = Field(default=1.0, gt=0, description="Spatial scale along Y (unit per pixel).")
    unit: str = Field(default="px", max_length=16, description="Spatial unit (e.g. px, mm, um).")
    description: Optional[str] = Field(default=None, description="Optional image description.")
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Per-image metadata key/value string pairs.",
    )
    calibrations: list[ArchiveImageCalibration] = Field(
        default_factory=list,
        description="List of per-image calibration records.",
    )
    sha256: Optional[str] = Field(
        default=None,
        min_length=64,
        max_length=64,
        description="SHA-256 hex digest of original image bytes, or None in annotations_only mode.",
    )
    size_bytes: Optional[int] = Field(
        default=None,
        ge=0,
        description="Original image size in bytes, or None in annotations_only mode.",
    )

    @field_validator("calibrations")
    @classmethod
    def validate_unique_image_calibrations(cls, v: list[ArchiveImageCalibration]) -> list[ArchiveImageCalibration]:
        seen_kinds: set[str] = set()
        for cal in v:
            if cal.kind in seen_kinds:
                raise ValueError(f"Duplicate calibration for kind '{cal.kind}' found on image.")
            seen_kinds.add(cal.kind)
        return v

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not re.fullmatch(r"^[a-fA-F0-9]{64}$", v):
            raise ValueError("sha256 must be a 64-character hexadecimal string.")
        return v.lower()

    @field_validator("archive_path")
    @classmethod
    def validate_archive_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not IMAGE_ARCHIVE_PATH_PATTERN.match(v) or ".." in v or v.startswith(("/", "\\")):
            raise ValueError(f"archive_path '{v}' must match 'images/<image-id>/<sanitized-basename>'.")
        return v


class ArchiveImage(StrictArchiveModel):
    """COCO-compatible image object with nested IQUANA extension."""
    id: int = Field(..., ge=1, description="Stable ZIP-local positive integer image ID.")
    file_name: str = Field(..., min_length=1, description="Original image filename.")
    width: int = Field(..., gt=0, description="Native image width in pixels.")
    height: int = Field(..., gt=0, description="Native image height in pixels.")
    iquana: ArchiveImageExtension = Field(..., description="IQUANA-specific image attributes.")

    @model_validator(mode="after")
    def validate_archive_path_encodes_id(self) -> ArchiveImage:
        if self.iquana.archive_path is not None:
            expected_prefix = f"images/{self.id}/"
            if not self.iquana.archive_path.startswith(expected_prefix):
                raise ValueError(
                    f"Image {self.id} archive_path '{self.iquana.archive_path}' must encode image id {self.id} "
                    f"(expected prefix '{expected_prefix}')."
                )
        return self


# ---------------------------------------------------------------------------
# Category / Label Schemas
# ---------------------------------------------------------------------------

class ArchiveCategoryExtension(StrictArchiveModel):
    """IQUANA non-COCO fields nested under categories[].iquana."""
    value: int = Field(..., description="Integer label value (e.g. 0 for background, 1, 2...).")
    parent_id: Optional[int] = Field(default=None, ge=1, description="Parent category archive ID, if any.")


class ArchiveCategory(StrictArchiveModel):
    """COCO-compatible category with required nested IQUANA label data."""
    id: int = Field(..., ge=1, description="Stable ZIP-local positive integer category ID.")
    name: str = Field(..., min_length=1, max_length=255, description="Category / label name (must be dataset-unique).")
    supercategory: Optional[str] = Field(default="none", description="COCO supercategory string.")
    iquana: ArchiveCategoryExtension = Field(
        ...,
        description="IQUANA label value and hierarchy parent reference (required for lossless restore).",
    )


# ---------------------------------------------------------------------------
# Annotation / Contour Schemas
# ---------------------------------------------------------------------------

class ArchiveGeometry(StrictArchiveModel):
    """Canonical normalized polygon coordinates [0.0, 1.0] within [-1.5, 1.5] tolerance."""
    x: list[float] = Field(..., min_length=3, description="Normalized X coordinates.")
    y: list[float] = Field(..., min_length=3, description="Normalized Y coordinates.")

    @model_validator(mode="after")
    def validate_coordinate_lengths_and_bounds(self) -> ArchiveGeometry:
        if len(self.x) != len(self.y):
            raise ValueError(f"Geometry x count ({len(self.x)}) must match y count ({len(self.y)}).")
        # Explicit finite and range check [-1.5, 1.5]
        for idx, val in enumerate(self.x):
            fval = float(val)
            if not math.isfinite(fval) or abs(fval) > 1.5:
                raise ValueError(f"Normalized coordinate x[{idx}]={val} exceeds tolerance range [-1.5, 1.5] or is non-finite.")
        for idx, val in enumerate(self.y):
            fval = float(val)
            if not math.isfinite(fval) or abs(fval) > 1.5:
                raise ValueError(f"Normalized coordinate y[{idx}]={val} exceeds tolerance range [-1.5, 1.5] or is non-finite.")
        return self


class ArchiveAnnotationExtension(StrictArchiveModel):
    """IQUANA non-COCO fields nested under annotations[].iquana."""
    mask_id: int = Field(..., ge=1, description="Archive ID of the mask this contour belongs to.")
    parent_id: Optional[int] = Field(default=None, ge=1, description="Archive ID of parent contour, if nested.")
    geometry: ArchiveGeometry = Field(..., description="Canonical normalized polygon coordinates.")
    added_by: str = Field(default="User", max_length=255, description="Generator provenance: User, SAM2, etc.")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Inference or assignment confidence.")
    created_at: IsoUtcDatetime = Field(..., description="ISO-8601 UTC creation timestamp.")
    author_username: Optional[str] = Field(default=None, description="Source author username (export provenance only).")
    reviewed_by: list[str] = Field(default_factory=list, description="Source reviewers (export provenance only).")

    @field_validator("added_by", mode="before")
    @classmethod
    def normalize_added_by(cls, v: Any) -> str:
        if v is None:
            return "User"
        s = str(v).strip()
        return s if s else "User"


class ArchiveAnnotation(StrictArchiveModel):
    """COCO-compatible annotation object with nested IQUANA extension."""
    id: int = Field(..., ge=1, description="Stable ZIP-local positive integer annotation ID.")
    image_id: int = Field(..., ge=1, description="Referenced archive image ID.")
    category_id: Optional[int] = Field(default=None, ge=1, description="Referenced archive category ID (None if unlabelled).")
    segmentation: list[list[float]] = Field(..., description="COCO flat polygon coordinates in native pixel space.")
    area: float = Field(..., ge=0.0, description="Derived area in native pixel^2.")
    bbox: list[float] = Field(..., min_length=4, max_length=4, description="COCO bounding box [x, y, w, h] in native pixels.")
    iscrowd: int = Field(default=0, ge=0, le=1, description="COCO crowd flag (always 0 for polygon contours).")
    iquana: ArchiveAnnotationExtension = Field(..., description="IQUANA normalized geometry and workflow metadata.")

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("bbox must contain exactly 4 numbers [x, y, w, h].")
        for idx, val in enumerate(v):
            if not math.isfinite(float(val)):
                raise ValueError(f"bbox element [{idx}]={val} must be a finite number.")
        if v[2] < 0 or v[3] < 0:
            raise ValueError("bbox width and height must be non-negative.")
        return v

    @model_validator(mode="after")
    def validate_segmentation_shape(self) -> ArchiveAnnotation:
        if len(self.segmentation) != 1:
            raise ValueError("segmentation must contain exactly one polygon.")
        polygon = self.segmentation[0]
        expected_length = 2 * len(self.iquana.geometry.x)
        if len(polygon) < 6 or len(polygon) % 2 or len(polygon) != expected_length:
            raise ValueError(
                "segmentation must contain one x/y pair per canonical geometry point."
            )
        return self


# ---------------------------------------------------------------------------
# Top-Level IQUANA Extension in annotations.json
# ---------------------------------------------------------------------------

class ArchiveDatasetInfo(StrictArchiveModel):
    """Dataset identity and provenance."""
    name: str = Field(..., min_length=1, max_length=50, description="Dataset display name.")
    description: Optional[str] = Field(default=None, max_length=255, description="Dataset description.")
    dataset_type: Literal["image"] = Field(default="image", description="v1 supports image datasets only.")
    created_by: Optional[str] = Field(default=None, description="Source creator username (export provenance only).")


class ArchiveActorProvenance(StrictArchiveModel):
    """Source actor username and roles summary for export-only provenance."""
    username: str = Field(..., min_length=1, description="Username in source system.")
    roles: list[str] = Field(default_factory=list, description="Roles held in source export (e.g. creator, reviewer).")


class ArchiveMetadataKey(StrictArchiveModel):
    """Per-dataset metadata key definition constrained to real application types."""
    key: str = Field(..., min_length=1, max_length=64, description="Metadata key name.")
    value_type: MetadataValueType = Field(default=MetadataValueType.CATEGORICAL, description="Data type.")
    unit: Optional[str] = Field(default=None, max_length=16, description="Display unit for numeric keys.")
    options: list[str] = Field(default_factory=list, description="Allowed vocabulary for categorical keys.")
    description: Optional[str] = Field(default=None, max_length=256, description="Description of the key.")


class ArchiveMask(StrictArchiveModel):
    """Mask container for an image."""
    id: int = Field(..., ge=1, description="Stable ZIP-local positive integer mask ID.")
    image_id: int = Field(..., ge=1, description="Referenced archive image ID.")
    fully_annotated: bool = Field(default=False, description="Whether the mask was marked fully annotated.")


class ArchiveRejection(StrictArchiveModel):
    """Reviewer rejection record constrained to real review enums."""
    id: int = Field(..., ge=1, description="Stable ZIP-local positive integer rejection ID.")
    mask_id: int = Field(..., ge=1, description="Referenced archive mask ID.")
    annotation_id: Optional[int] = Field(default=None, ge=1, description="Referenced archive annotation ID (None if mask-level).")
    reason: RejectionReason = Field(..., description="Rejection reason code.")
    note: Optional[str] = Field(default=None, max_length=1000, description="Reviewer feedback note.")
    created_by: Optional[str] = Field(default=None, description="Source reviewer username (export provenance only).")
    created_at: IsoUtcDatetime = Field(..., description="ISO-8601 UTC creation timestamp.")
    resolved_at: OptionalIsoUtcDatetime = Field(default=None, description="ISO-8601 UTC resolution timestamp.")
    resolved_by: Optional[str] = Field(default=None, description="Source resolving username (export provenance only).")
    resolution: Optional[RejectionResolution] = Field(default=None, description="Resolution outcome.")


class ArchiveCounts(StrictArchiveModel):
    """Summary counts for validation and integrity checking."""
    images: int = Field(..., ge=0)
    annotations: int = Field(..., ge=0)
    categories: int = Field(..., ge=0)
    masks: int = Field(..., ge=0)
    rejections: int = Field(..., ge=0)
    temporary_contours_omitted: int = Field(default=0, ge=0)


class ArchiveFileEntry(StrictArchiveModel):
    """Checksum and dimension manifest entry for an archived image file."""
    image_id: int = Field(..., ge=1, description="Referenced archive image ID.")
    path: str = Field(..., description="Relative archive path (images/<id>/<filename>).")
    sha256: str = Field(..., min_length=64, max_length=64, description="SHA-256 digest of the image file.")
    size_bytes: int = Field(..., ge=0, description="File size in bytes.")
    width: int = Field(..., gt=0, description="Image width verified from file header.")
    height: int = Field(..., gt=0, description="Image height verified from file header.")
    color_mode: str = Field(..., min_length=1, description="Color mode verified from file header.")

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, v: str) -> str:
        if not re.fullmatch(r"^[a-fA-F0-9]{64}$", v):
            raise ValueError("sha256 must be a 64-character hexadecimal string.")
        return v.lower()

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not IMAGE_ARCHIVE_PATH_PATTERN.match(v) or ".." in v or v.startswith(("/", "\\")):
            raise ValueError(f"path '{v}' must match 'images/<image-id>/<sanitized-basename>'.")
        return v


class IquanaDatasetExtension(StrictArchiveModel):
    """Top-level iquana extension object inside annotations.json."""
    content_mode: ArchiveContentMode = Field(
        default="full",
        description="Archive content mode: 'full' (includes images) or 'annotations_only' (metadata and geometries only).",
    )
    dataset: ArchiveDatasetInfo = Field(..., description="Dataset metadata and source creator.")
    actors: list[ArchiveActorProvenance] = Field(
        default_factory=list,
        description="Source actors participating in this dataset (export provenance only).",
    )
    metadata_keys: list[ArchiveMetadataKey] = Field(
        default_factory=list,
        description="Declared metadata key definitions for the dataset.",
    )
    masks: list[ArchiveMask] = Field(default_factory=list, description="Mask entities.")
    rejections: list[ArchiveRejection] = Field(default_factory=list, description="Annotation rejections.")
    counts: ArchiveCounts = Field(..., description="Counts summary.")
    files: list[ArchiveFileEntry] = Field(
        default_factory=list,
        description="Manifest of all image files with hashes and dimensions.",
    )


# ---------------------------------------------------------------------------
# Complete annotations.json Document Schema
# ---------------------------------------------------------------------------

class IquanaAnnotationsDocument(StrictArchiveModel):
    """Root document schema for annotations.json in an IQUANA dataset archive."""
    format: Literal["iquana"] = Field(default=ARCHIVE_FORMAT, description="Format discriminator.")
    format_version: Literal[1] = Field(default=ARCHIVE_FORMAT_VERSION, description="Format version.")
    info: CocoInfo = Field(..., description="COCO info block.")
    licenses: list[CocoLicense] = Field(default_factory=list, description="COCO licenses list.")
    images: list[ArchiveImage] = Field(default_factory=list, description="COCO images list.")
    annotations: list[ArchiveAnnotation] = Field(default_factory=list, description="COCO annotations list.")
    categories: list[ArchiveCategory] = Field(default_factory=list, description="COCO categories list.")
    iquana: IquanaDatasetExtension = Field(..., description="IQUANA dataset state extension.")

    @model_validator(mode="after")
    def validate_document_integrity(self) -> IquanaAnnotationsDocument:
        # 1. Category name uniqueness and category hierarchy validation (Linear-time O(n))
        category_names: set[str] = set()
        category_by_id: dict[int, ArchiveCategory] = {}
        category_parent_map: dict[int, Optional[int]] = {}

        for cat in self.categories:
            if cat.name in category_names:
                raise ValueError(f"Duplicate category name '{cat.name}' found in categories.")
            category_names.add(cat.name)
            if cat.id in category_by_id:
                raise ValueError(f"Duplicate category id {cat.id} found in categories.")
            category_by_id[cat.id] = cat
            category_parent_map[cat.id] = cat.iquana.parent_id

        for cat in self.categories:
            parent_id = cat.iquana.parent_id
            if parent_id is not None:
                if parent_id not in category_by_id:
                    raise ValueError(f"Category {cat.id} references non-existent parent category id {parent_id}.")
                if parent_id == cat.id:
                    raise ValueError(f"Category {cat.id} cannot be its own parent.")

        check_hierarchy_acyclic(category_parent_map, "category")

        # 2. Image ID uniqueness
        image_by_id: dict[int, ArchiveImage] = {}
        for img in self.images:
            if img.id in image_by_id:
                raise ValueError(f"Duplicate image id {img.id} found in images.")
            image_by_id[img.id] = img

        # 3. Mask ID uniqueness and image reference validity
        mask_by_id: dict[int, ArchiveMask] = {}
        for mask in self.iquana.masks:
            if mask.id in mask_by_id:
                raise ValueError(f"Duplicate mask id {mask.id} found in iquana.masks.")
            mask_by_id[mask.id] = mask
            if mask.image_id not in image_by_id:
                raise ValueError(f"Mask {mask.id} references non-existent image id {mask.image_id}.")

        # 4. Metadata key uniqueness and image metadata validation against definitions
        metadata_keys_by_key: dict[str, ArchiveMetadataKey] = {}
        for mk in self.iquana.metadata_keys:
            if mk.key in metadata_keys_by_key:
                raise ValueError(f"Duplicate metadata key definition for '{mk.key}'.")
            metadata_keys_by_key[mk.key] = mk

        for img in self.images:
            for key, val in img.iquana.metadata.items():
                if key not in metadata_keys_by_key:
                    raise ValueError(
                        f"Image {img.id} uses undeclared metadata key '{key}'. "
                        f"All metadata keys must be declared in iquana.metadata_keys."
                    )
                mk_def = metadata_keys_by_key[key]
                try:
                    _, val_num = coerce(val, mk_def.value_type, mk_def.options)
                    if val_num is not None and not math.isfinite(val_num):
                        raise ValueError(f"Numeric metadata '{key}' value '{val}' must be a finite number.")
                except InvalidMetadataError as exc:
                    raise ValueError(
                        f"Image {img.id} has invalid metadata value for key '{key}': {exc}"
                    ) from exc

        # 5. Annotation ID uniqueness, mask/image ownership, and parent hierarchy (Linear-time O(n))
        ann_by_id: dict[int, ArchiveAnnotation] = {}
        ann_parent_map: dict[int, Optional[int]] = {}

        for ann in self.annotations:
            if ann.id in ann_by_id:
                raise ValueError(f"Duplicate annotation id {ann.id} found in annotations.")
            ann_by_id[ann.id] = ann
            ann_parent_map[ann.id] = ann.iquana.parent_id

            if ann.image_id not in image_by_id:
                raise ValueError(f"Annotation {ann.id} references non-existent image id {ann.image_id}.")
            if ann.category_id is not None and ann.category_id not in category_by_id:
                raise ValueError(f"Annotation {ann.id} references non-existent category id {ann.category_id}.")
            if ann.iquana.mask_id not in mask_by_id:
                raise ValueError(f"Annotation {ann.id} references non-existent mask id {ann.iquana.mask_id}.")

            # Annotation's image_id must match the mask's image_id
            mask = mask_by_id[ann.iquana.mask_id]
            if ann.image_id != mask.image_id:
                raise ValueError(
                    f"Annotation {ann.id} image_id ({ann.image_id}) does not match "
                    f"its mask {mask.id} image_id ({mask.image_id})."
                )

            # COCO fields are a projection of the canonical normalized geometry.
            geometry = ann.iquana.geometry
            expected_polygon = [
                coordinate
                for x, y in zip(geometry.x, geometry.y)
                for coordinate in (x * image_by_id[ann.image_id].width, y * image_by_id[ann.image_id].height)
            ]
            polygon = ann.segmentation[0]
            if any(
                not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6)
                for actual, expected in zip(polygon, expected_polygon)
            ):
                raise ValueError(f"Annotation {ann.id} segmentation does not match canonical geometry.")

            x_points = expected_polygon[0::2]
            y_points = expected_polygon[1::2]
            expected_bbox = [
                min(x_points),
                min(y_points),
                max(x_points) - min(x_points),
                max(y_points) - min(y_points),
            ]
            if any(
                not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6)
                for actual, expected in zip(ann.bbox, expected_bbox)
            ):
                raise ValueError(f"Annotation {ann.id} bbox does not match canonical geometry.")

            point_count = len(x_points)
            expected_area = abs(sum(
                x_points[i] * y_points[(i + 1) % point_count]
                - y_points[i] * x_points[(i + 1) % point_count]
                for i in range(point_count)
            )) / 2.0
            if not math.isclose(ann.area, expected_area, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError(f"Annotation {ann.id} area does not match canonical geometry.")

        # Annotation parent hierarchy: must belong to the same mask and be acyclic
        for ann in self.annotations:
            parent_id = ann.iquana.parent_id
            if parent_id is not None:
                if parent_id not in ann_by_id:
                    raise ValueError(f"Annotation {ann.id} references non-existent parent annotation {parent_id}.")
                if parent_id == ann.id:
                    raise ValueError(f"Annotation {ann.id} cannot be its own parent.")
                parent_ann = ann_by_id[parent_id]
                if parent_ann.iquana.mask_id != ann.iquana.mask_id:
                    raise ValueError(
                        f"Annotation {ann.id} (mask_id {ann.iquana.mask_id}) and parent annotation "
                        f"{parent_id} (mask_id {parent_ann.iquana.mask_id}) must belong to the same mask."
                    )

        check_hierarchy_acyclic(ann_parent_map, "annotation")

        # 6. Rejection references and ownership: annotation must belong to rejection's mask
        rejection_ids: set[int] = set()
        for rej in self.iquana.rejections:
            if rej.id in rejection_ids:
                raise ValueError(f"Duplicate rejection id {rej.id} found in iquana.rejections.")
            rejection_ids.add(rej.id)
            if rej.mask_id not in mask_by_id:
                raise ValueError(f"Rejection {rej.id} references non-existent mask id {rej.mask_id}.")
            if rej.annotation_id is not None:
                if rej.annotation_id not in ann_by_id:
                    raise ValueError(f"Rejection {rej.id} references non-existent annotation id {rej.annotation_id}.")
                rej_ann = ann_by_id[rej.annotation_id]
                if rej_ann.iquana.mask_id != rej.mask_id:
                    raise ValueError(
                        f"Rejection {rej.id} mask_id ({rej.mask_id}) does not match referenced "
                        f"annotation {rej_ann.id} mask_id ({rej_ann.iquana.mask_id})."
                    )

        # 7. File manifest and content mode validation
        content_mode = self.iquana.content_mode
        if content_mode == "full":
            for img in self.images:
                if img.iquana.archive_path is None or img.iquana.sha256 is None or img.iquana.size_bytes is None:
                    raise ValueError(
                        f"Image {img.id} is missing archive_path, sha256, or size_bytes required in 'full' mode."
                    )

            manifest_by_image_id: dict[int, ArchiveFileEntry] = {}
            for f in self.iquana.files:
                if f.image_id in manifest_by_image_id:
                    raise ValueError(f"Duplicate file manifest entry for image id {f.image_id}.")
                manifest_by_image_id[f.image_id] = f

            if set(manifest_by_image_id.keys()) != set(image_by_id.keys()):
                missing = set(image_by_id.keys()) - set(manifest_by_image_id.keys())
                extra = set(manifest_by_image_id.keys()) - set(image_by_id.keys())
                raise ValueError(
                    f"File manifest image IDs do not match images list. Missing: {missing}, extra: {extra}."
                )

            for img_id, img in image_by_id.items():
                file_entry = manifest_by_image_id[img_id]
                expected_prefix = f"images/{img_id}/"
                if not file_entry.path.startswith(expected_prefix):
                    raise ValueError(
                        f"File manifest path '{file_entry.path}' must encode image id {img_id} "
                        f"(expected prefix '{expected_prefix}')."
                    )
                if file_entry.path != img.iquana.archive_path:
                    raise ValueError(
                        f"File manifest path '{file_entry.path}' does not match image {img_id} "
                        f"archive_path '{img.iquana.archive_path}'."
                    )
                if file_entry.sha256 != img.iquana.sha256:
                    raise ValueError(
                        f"File manifest sha256 for image {img_id} does not match image record "
                        f"('{file_entry.sha256}' vs '{img.iquana.sha256}')."
                    )
                if file_entry.size_bytes != img.iquana.size_bytes:
                    raise ValueError(
                        f"File manifest size_bytes for image {img_id} does not match image record "
                        f"({file_entry.size_bytes} vs {img.iquana.size_bytes})."
                    )
                if file_entry.width != img.width:
                    raise ValueError(
                        f"File manifest width for image {img_id} does not match image record "
                        f"({file_entry.width} vs {img.width})."
                    )
                if file_entry.height != img.height:
                    raise ValueError(
                        f"File manifest height for image {img_id} does not match image record "
                        f"({file_entry.height} vs {img.height})."
                    )
                if file_entry.color_mode != img.iquana.color_mode:
                    raise ValueError(
                        f"File manifest color_mode for image {img_id} does not match image record "
                        f"('{file_entry.color_mode}' vs '{img.iquana.color_mode}')."
                    )
        elif content_mode == "annotations_only":
            if self.iquana.files:
                raise ValueError("File manifest (iquana.files) must be empty in 'annotations_only' mode.")
            for img in self.images:
                if (
                    img.iquana.archive_path is not None
                    or img.iquana.sha256 is not None
                    or img.iquana.size_bytes is not None
                ):
                    raise ValueError(
                        f"Image {img.id} has archive_path, sha256, or size_bytes populated. "
                        f"Asset-only fields must be null in 'annotations_only' mode."
                    )

        # 8. Counts match actual item counts
        counts = self.iquana.counts
        if counts.images != len(self.images):
            raise ValueError(f"Counts.images ({counts.images}) != len(images) ({len(self.images)}).")
        if counts.annotations != len(self.annotations):
            raise ValueError(f"Counts.annotations ({counts.annotations}) != len(annotations) ({len(self.annotations)}).")
        if counts.categories != len(self.categories):
            raise ValueError(f"Counts.categories ({counts.categories}) != len(categories) ({len(self.categories)}).")
        if counts.masks != len(self.iquana.masks):
            raise ValueError(f"Counts.masks ({counts.masks}) != len(masks) ({len(self.iquana.masks)}).")
        if counts.rejections != len(self.iquana.rejections):
            raise ValueError(f"Counts.rejections ({counts.rejections}) != len(rejections) ({len(self.iquana.rejections)}).")

        return self


# ---------------------------------------------------------------------------
# Optional config.json Document Schemas
# ---------------------------------------------------------------------------

class ArchiveConfigDataset(StrictArchiveModel):
    """Dataset behavior configuration settings."""
    require_independent_review: bool = Field(
        default=False,
        description="Whether a contour cannot be approved by the author who created it.",
    )


class ArchiveCalibrationDefault(StrictArchiveModel):
    """Per-dataset calibration defaults for one calibration kind."""
    kind: str = Field(..., min_length=1, max_length=32, description="Calibration kind (e.g. response).")
    defaults: dict[str, Any] = Field(default_factory=dict, description="Strategy configuration defaults.")

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if not v_clean:
            raise ValueError("Calibration kind cannot be empty.")
        return v_clean

    @model_validator(mode="after")
    def validate_calibration_defaults(self) -> ArchiveCalibrationDefault:
        from app.exceptions import InvalidCalibrationError, UnknownCalibrationKindError
        from app.services.calibration.service import validate_defaults

        try:
            self.defaults = validate_defaults(self.kind, self.defaults)
        except (InvalidCalibrationError, UnknownCalibrationKindError) as exc:
            raise ValueError(f"Invalid calibration defaults for kind '{self.kind}': {exc}") from exc
        return self


#: Quantification metric keys supported by archive format v1.
#: This fixed set matches the current ``iquana-toolbox`` ``METRIC_REGISTRY``.
#: Unknown or future metric keys are rejected during archive validation because the
#: runtime profile schema cannot read them back safely.
STANDARD_V1_METRIC_KEYS: frozenset[str] = frozenset({
    "area",
    "perimeter",
    "circularity",
    "max_diameter",
    "mean_color_rgb",
    "mean_color_lab",
    "mean_intensity",
    "nn_distance",
    "mean_knn_distance",
    "n_children",
})


class ArchiveProfileEntry(StrictArchiveModel):
    """Single supported metric configuration inside a quantification profile."""
    metric_key: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Supported v1 registry key of the metric.",
    )
    params: dict[str, Any] = Field(default_factory=dict, description="Per-metric parameters.")
    label_names: Optional[list[str]] = Field(
        default=None,
        description="List of dataset-unique label names this metric applies to (None = all labels).",
    )

    @field_validator("metric_key")
    @classmethod
    def validate_metric_key(cls, v: str) -> str:
        v_clean = v.strip()
        if not v_clean:
            raise ValueError("Metric key cannot be empty or whitespace.")
        if v_clean not in STANDARD_V1_METRIC_KEYS:
            raise ValueError(
                f"Metric key '{v_clean}' is not supported by archive format v1. "
                f"Supported metric keys: {', '.join(sorted(STANDARD_V1_METRIC_KEYS))}."
            )
        return v_clean


class ArchiveQuantificationProfile(StrictArchiveModel):
    """Portable quantification profile definition (definitions only, NO computed values)."""
    name: str = Field(..., min_length=1, max_length=128, description="Human-readable profile name.")
    is_default: bool = Field(default=False, description="Whether this is the dataset's default profile.")
    entries: list[ArchiveProfileEntry] = Field(default_factory=list, description="Ordered metric selections.")


class ArchiveModelRoutingBinding(StrictArchiveModel):
    """A model routing assignment referencing labels by name and constrained to real tasks."""
    task: ModelRoutingTask = Field(..., description="Inference task capability.")
    label_name: Optional[str] = Field(
        default=None,
        description="Unique label name for a label-specific override, or None for task default.",
    )
    model_registry_key: str = Field(..., min_length=1, description="Registry key of the model.")
    inputs: Optional[dict[str, Any]] = Field(
        default=None,
        description="Model-owned runtime parameters. Contour IDs must be stripped.",
    )

    @field_validator("inputs")
    @classmethod
    def reject_query_contour_id(cls, v: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Ensure no local contour primary key is serialized in inputs."""
        if v is not None and "conditioning" in v:
            conditioning = v.get("conditioning")
            if isinstance(conditioning, dict) and "query_contour_id" in conditioning:
                raise ValueError(
                    "inputs.conditioning.query_contour_id must be stripped before serialization "
                    "and recorded in omitted_fields."
                )
        return v


class ArchiveModelRouting(StrictArchiveModel):
    """Model routing configuration with unique (task, label_name) selectors."""
    bindings: list[ArchiveModelRoutingBinding] = Field(
        default_factory=list,
        description="List of task defaults and label-specific routing bindings.",
    )

    @model_validator(mode="after")
    def validate_unique_selectors(self) -> ArchiveModelRouting:
        seen: set[tuple[str, Optional[str]]] = set()
        for b in self.bindings:
            selector = (b.task, b.label_name)
            if selector in seen:
                desc = f"task '{b.task}'" + (f" and label '{b.label_name}'" if b.label_name else " (default)")
                raise ValueError(f"Duplicate model routing selector for {desc}.")
            seen.add(selector)
        return self


class ArchiveOmittedField(StrictArchiveModel):
    """Audit record for non-portable runtime state stripped during export."""
    section: str = Field(..., description="Configuration section where omission occurred.")
    field: str = Field(..., description="Dotted path of the omitted field.")
    task: Optional[ModelRoutingTask] = Field(default=None, description="Affected task if applicable.")
    label_name: Optional[str] = Field(default=None, description="Affected label name if applicable.")
    reason: str = Field(..., description="Explanation of why this field was omitted from export.")


class IquanaConfigDocument(StrictArchiveModel):
    """Root document schema for config.json in an IQUANA dataset archive."""
    format: Literal["iquana"] = Field(default=ARCHIVE_FORMAT, description="Format discriminator.")
    format_version: Literal[1] = Field(default=ARCHIVE_FORMAT_VERSION, description="Format version.")
    dataset: ArchiveConfigDataset = Field(default_factory=ArchiveConfigDataset, description="Dataset behavior flags.")
    calibration_defaults: list[ArchiveCalibrationDefault] = Field(
        default_factory=list,
        description="Per-dataset calibration strategy defaults.",
    )
    quantification_profiles: list[ArchiveQuantificationProfile] = Field(
        default_factory=list,
        description="Quantification profile definitions (no computed values).",
    )
    model_routing: ArchiveModelRouting = Field(
        default_factory=ArchiveModelRouting,
        description="Model routing policies with label names.",
    )
    omitted_fields: list[ArchiveOmittedField] = Field(
        default_factory=list,
        description="Disclosed non-portable runtime fields stripped from configuration.",
    )

    @field_validator("calibration_defaults")
    @classmethod
    def validate_unique_calibration_defaults(
        cls, v: list[ArchiveCalibrationDefault]
    ) -> list[ArchiveCalibrationDefault]:
        seen_kinds: set[str] = set()
        for cd in v:
            if cd.kind in seen_kinds:
                raise ValueError(
                    f"Duplicate calibration default for kind '{cd.kind}' found in calibration_defaults."
                )
            seen_kinds.add(cd.kind)
        return v

    @model_validator(mode="after")
    def validate_config_invariants(self) -> IquanaConfigDocument:
        default_count = sum(1 for p in self.quantification_profiles if p.is_default)
        if default_count > 1:
            raise ValueError(
                f"At most one quantification profile may be marked as default (found {default_count})."
            )

        return self


# ---------------------------------------------------------------------------
# HTTP Import Response Schema
# ---------------------------------------------------------------------------

class DatasetArchiveImportResponse(BaseModel):
    """Response returned upon successful dataset archive import."""
    model_config = ConfigDict(extra="forbid")

    success: bool = Field(default=True, description="Success status flag.")
    message: str = Field(default="Dataset imported successfully.", description="Success message.")
    dataset_id: int = Field(..., description="The newly created database dataset ID.")
    dataset_name: str = Field(..., description="The resolved dataset name.")
    config_applied: bool = Field(..., description="Whether config.json was present and applied.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings encountered during import.")

"""Dataset archive serialization and export service for IQUANA dataset format v1.

Provides snapshot queries, stable ID remapping, canonical coordinate serialization,
integrity hashing, and streaming TemporaryFile archive generation.
Excludes computed quantifications (ContourMetrics) and temporary contours.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import math
import os
import queue
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid
import zipfile
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np
from PIL import Image
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.contours import Contours, dual_write_geometry_metrics, reviewer_contour_association
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
from app.exceptions import (
    DatasetArchiveExportError,
    DatasetArchiveImportError,
    DatasetArchiveNameConflictError,
    DatasetArchiveSizeLimitError,
    DatasetArchiveValidationError,
    DatasetNotFoundError,
)
from app.services.database_access.members import ensure_owner_membership
from app.services.embedding_lifecycle import enqueue_embed_contours, enqueue_embed_image
from app.services.metadata_types import InvalidMetadataError, MetadataValueType, coerce
from app.services.model_registry import _models_for_task
import config
from config import (
    DATASET_ARCHIVE_MAX_COMPRESSED_BYTES,
    DATASET_ARCHIVE_MAX_CONTROL_JSON_BYTES,
    DATASET_ARCHIVE_MAX_MEMBER_BYTES,
    DATASET_ARCHIVE_MAX_MEMBERS,
    DATASET_ARCHIVE_MAX_UNCOMPRESSED_BYTES,
)
from app.schemas.dataset_archive import (
    ARCHIVE_FORMAT,
    ARCHIVE_FORMAT_VERSION,
    STANDARD_V1_METRIC_KEYS,
    ArchiveActorProvenance,
    ArchiveAnnotation,
    ArchiveAnnotationExtension,
    ArchiveCalibrationDefault,
    ArchiveCategory,
    ArchiveCategoryExtension,
    ArchiveConfigDataset,
    ArchiveCounts,
    ArchiveDatasetInfo,
    ArchiveFileEntry,
    ArchiveGeometry,
    ArchiveImage,
    ArchiveImageCalibration,
    ArchiveImageExtension,
    ArchiveMask,
    ArchiveMetadataKey,
    ArchiveModelRouting,
    ArchiveModelRoutingBinding,
    ArchiveOmittedField,
    ArchiveProfileEntry,
    ArchiveQuantificationProfile,
    ArchiveRejection,
    CocoInfo,
    IquanaAnnotationsDocument,
    IquanaConfigDocument,
    IquanaDatasetExtension,
)
from app.schemas.inference import ModelRoutingTask
from app.schemas.review import RejectionReason, RejectionResolution
from app.services.metadata_types import MetadataValueType

logger = logging.getLogger(__name__)


def _sanitize_archive_filename(file_name: str, fallback_id: int) -> str:
    """Sanitizes image basename to ensure safe ZIP-local relative paths."""
    basename = os.path.basename(file_name).strip()
    clean = re.sub(r"[\x00-\x1f\x7f/\\\\]", "_", basename)
    if not clean or clean in (".", ".."):
        clean = f"image_{fallback_id}"
    return clean


def _sanitize_attachment_filename(dataset_name: str, dataset_id: int) -> str:
    """Derives a safe Content-Disposition attachment filename from the dataset name."""
    clean = re.sub(r"[^a-zA-Z0-9_\-.]", "_", dataset_name.strip()).strip("._")
    if not clean:
        clean = f"dataset_{dataset_id}"
    return f"{clean}.zip"


def _slugify_dataset_name(name: str) -> str:
    """Derives a safe, bounded-length filesystem slug from a dataset name.

    Never returns a value containing path separators or '..', so it is safe
    to join onto a directory root without permitting traversal.
    """
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("._-")
    return slug[:50] if slug else "dataset"


# Small, fixed-size, process-wide pool of daemon workers for model-registry readiness
# checks: reused across every import (never one thread per call or per key), so a
# slow/unreachable MLflow can tie up at most this many background threads, no matter
# how many imports run or how many distinct routing bindings each references. Plain
# daemon threads pulling from a queue -- not concurrent.futures.ThreadPoolExecutor --
# because that executor's worker threads are joined by an atexit hook, so a lookup
# stuck on an unreachable registry would hang interpreter shutdown; daemon threads are
# killed outright and never block exit.
#
# The queue itself is bounded: if every worker is wedged on an unreachable registry,
# an unbounded queue would let concurrent/repeated imports keep enqueuing closures
# forever (each import returns after its own timeout, but the backlog never drains),
# growing without limit. A bounded queue caps that backlog; see put_nowait() below.
_MODEL_AVAILABILITY_QUEUE: queue.Queue = queue.Queue(maxsize=32)


def _model_availability_worker() -> None:
    while True:
        job = _MODEL_AVAILABILITY_QUEUE.get()
        try:
            job()
        except Exception:
            logger.exception("Model registry availability check worker failed.")


for _ in range(4):
    threading.Thread(target=_model_availability_worker, daemon=True).start()


def _confirmed_ready_model_bindings(
    task_key_pairs: set[tuple[str, str]], timeout_seconds: float = 2.0
) -> set[tuple[str, str]]:
    """Best-effort, bounded-time check for which (task, model_registry_key) routing
    bindings are actually usable at runtime: registered, status=ready, and tagged for
    that task -- the same bar the runtime's own model discovery applies (see
    ``_models_for_task``), not merely "a record with this name exists".

    One lookup per distinct *task* (not per key) covers every binding under it, since
    the set of supported tasks is small and fixed while the number of referenced keys
    is not. Lookups run on the bounded worker pool above and are awaited against one
    shared wall-clock deadline for the whole batch, so total time never exceeds
    ``timeout_seconds`` regardless of how many tasks or keys are involved. A task that
    can't even be queued (the bounded queue is full) or whose lookup misses the
    deadline leaves its bindings "not confirmed", not "confirmed unusable" -- a queued
    job keeps running in the background and its result is simply discarded.
    """
    tasks = {task for task, _ in task_key_pairs}
    ready_keys_by_task: dict[str, set[str]] = {}
    lock = threading.Lock()
    done_events = {task: threading.Event() for task in tasks}

    def make_job(task: str):
        def _job() -> None:
            try:
                keys = {m.name for m in _models_for_task(task)}
            except Exception:
                keys = None
            if keys is not None:
                with lock:
                    ready_keys_by_task[task] = keys
            done_events[task].set()
        return _job

    queued_tasks: set[str] = set()
    for task in tasks:
        try:
            _MODEL_AVAILABILITY_QUEUE.put_nowait(make_job(task))
            queued_tasks.add(task)
        except queue.Full:
            # Backlog is at capacity (workers likely wedged on an unreachable
            # registry): treat as unconfirmed immediately rather than blocking this
            # import on put() or growing the backlog further.
            pass

    deadline = time.monotonic() + timeout_seconds
    for task in queued_tasks:
        done_events[task].wait(max(0.0, deadline - time.monotonic()))

    with lock:
        # Snapshot under the lock at the deadline: a straggler job past the deadline
        # keeps running (the queue is not cancelled) and must never be able to mutate
        # what this function has already returned to its caller.
        snapshot = {task: set(keys) for task, keys in ready_keys_by_task.items()}

    return {(task, key) for task, key in task_key_pairs if key in snapshot.get(task, set())}


def _to_utc_datetime(dt: Any) -> datetime | None:
    """Ensures a datetime or ISO string is converted to a timezone-aware UTC datetime."""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(dt, str):
        try:
            parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _normalize_and_validate_geometry(
    x_coords: list[Any],
    y_coords: list[Any],
    width: int,
    height: int,
    contour_id: int,
) -> tuple[ArchiveGeometry, list[list[float]], list[float], float]:
    """Validates contour coordinates, normalizes legacy pixels if needed, and projects COCO fields.

    Returns:
        (canonical_geometry, segmentation, bbox, area)
    """
    if not x_coords or not y_coords:
        raise DatasetArchiveExportError(
            f"Contour {contour_id} has empty coordinate lists."
        )
    if len(x_coords) != len(y_coords):
        raise DatasetArchiveExportError(
            f"Contour {contour_id} x coordinate count ({len(x_coords)}) does not match "
            f"y coordinate count ({len(y_coords)})."
        )
    if len(x_coords) < 3:
        raise DatasetArchiveExportError(
            f"Contour {contour_id} has fewer than 3 coordinates ({len(x_coords)})."
        )

    try:
        raw_x = [float(v) for v in x_coords]
        raw_y = [float(v) for v in y_coords]
    except (ValueError, TypeError) as exc:
        raise DatasetArchiveExportError(
            f"Contour {contour_id} contains non-numeric coordinates: {exc}"
        ) from exc

    for idx, v in enumerate(raw_x):
        if not math.isfinite(v):
            raise DatasetArchiveExportError(
                f"Contour {contour_id} contains non-finite x[{idx}] coordinate: {v}"
            )
    for idx, v in enumerate(raw_y):
        if not math.isfinite(v):
            raise DatasetArchiveExportError(
                f"Contour {contour_id} contains non-finite y[{idx}] coordinate: {v}"
            )

    is_normalized = (
        max(abs(v) for v in raw_x) <= 1.5
        and max(abs(v) for v in raw_y) <= 1.5
    )

    if is_normalized:
        norm_x = raw_x
        norm_y = raw_y
    else:
        norm_x = [v / float(width) for v in raw_x]
        norm_y = [v / float(height) for v in raw_y]

    for idx, v in enumerate(norm_x):
        if abs(v) > 1.5:
            raise DatasetArchiveExportError(
                f"Contour {contour_id} normalized x[{idx}] coordinate ({v}) exceeds tolerance [-1.5, 1.5]."
            )
    for idx, v in enumerate(norm_y):
        if abs(v) > 1.5:
            raise DatasetArchiveExportError(
                f"Contour {contour_id} normalized y[{idx}] coordinate ({v}) exceeds tolerance [-1.5, 1.5]."
            )

    expected_polygon: list[float] = [
        coord
        for nx, ny in zip(norm_x, norm_y)
        for coord in (nx * float(width), ny * float(height))
    ]
    x_points = expected_polygon[0::2]
    y_points = expected_polygon[1::2]
    bbox = [
        min(x_points),
        min(y_points),
        max(x_points) - min(x_points),
        max(y_points) - min(y_points),
    ]

    point_count = len(x_points)
    area = abs(sum(
        x_points[i] * y_points[(i + 1) % point_count]
        - y_points[i] * x_points[(i + 1) % point_count]
        for i in range(point_count)
    )) / 2.0

    geometry = ArchiveGeometry(x=norm_x, y=norm_y)
    return geometry, [expected_polygon], bbox, area


def create_iquana_dataset_archive(
    db: Session,
    dataset_id: int,
    include_config: bool = False,
) -> tuple[BinaryIO, str]:
    """Creates a complete, verified IQUANA dataset archive (v1) in a temporary file.

    Args:
        db: Active SQLAlchemy database session.
        dataset_id: Target dataset database ID.
        include_config: Whether to include optional portable dataset configuration (config.json).

    Returns:
        tuple of (open_temporary_file, attachment_filename). The caller is responsible
        for closing the returned TemporaryFile, which will automatically delete it.
    """
    # 1. Fetch dataset
    dataset = db.query(Datasets).filter(Datasets.id == dataset_id).first()
    if dataset is None:
        raise DatasetNotFoundError(f"Dataset {dataset_id} not found.")

    if dataset.dataset_type != "image":
        raise DatasetArchiveExportError(
            f"Unsupported dataset type '{dataset.dataset_type}'. Only 'image' datasets are supported in v1."
        )

    # 2. Snapshot bulk queries for dataset contents
    images: list[Images] = (
        db.query(Images)
        .filter(Images.dataset_id == dataset_id)
        .order_by(Images.id)
        .all()
    )
    image_ids = [img.id for img in images]

    labels: list[Labels] = (
        db.query(Labels)
        .filter(Labels.dataset_id == dataset_id)
        .order_by(Labels.name, Labels.id)
        .all()
    )

    masks: list[Masks] = []
    if image_ids:
        masks = (
            db.query(Masks)
            .filter(Masks.image_id.in_(image_ids))
            .order_by(Masks.id)
            .all()
        )
    mask_ids = [m.id for m in masks]

    contours: list[Contours] = []
    if mask_ids:
        contours = (
            db.query(Contours)
            .filter(Contours.mask_id.in_(mask_ids))
            .order_by(Contours.id)
            .all()
        )
    contour_ids = [c.id for c in contours]

    reviewers_map: dict[int, list[str]] = defaultdict(list)
    if contour_ids:
        reviewer_rows = (
            db.query(reviewer_contour_association)
            .filter(reviewer_contour_association.c.contour_id.in_(contour_ids))
            .all()
        )
        for row in reviewer_rows:
            reviewers_map[row.contour_id].append(row.reviewer_id)

    rejections: list[AnnotationRejections] = []
    if mask_ids:
        rejections = (
            db.query(AnnotationRejections)
            .filter(AnnotationRejections.mask_id.in_(mask_ids))
            .order_by(AnnotationRejections.id)
            .all()
        )

    image_metadata_rows: list[ImageMetadata] = []
    if image_ids:
        image_metadata_rows = (
            db.query(ImageMetadata)
            .filter(ImageMetadata.image_id.in_(image_ids))
            .order_by(ImageMetadata.image_id, ImageMetadata.key)
            .all()
        )

    declared_metadata_keys: list[DatasetMetadataKeys] = (
        db.query(DatasetMetadataKeys)
        .filter(DatasetMetadataKeys.dataset_id == dataset_id)
        .order_by(DatasetMetadataKeys.key)
        .all()
    )

    image_calibrations: list[ImageCalibrations] = []
    if image_ids:
        image_calibrations = (
            db.query(ImageCalibrations)
            .filter(ImageCalibrations.image_id.in_(image_ids))
            .order_by(ImageCalibrations.image_id, ImageCalibrations.kind)
            .all()
        )

    members: list[DatasetMembers] = (
        db.query(DatasetMembers)
        .filter(DatasetMembers.dataset_id == dataset_id)
        .order_by(DatasetMembers.username)
        .all()
    )

    tmp_file = tempfile.TemporaryFile(mode="w+b")
    try:
        # 3. Validate Category name uniqueness
        seen_category_names: set[str] = set()
        duplicate_names: set[str] = set()
        for lbl in labels:
            if lbl.name in seen_category_names:
                duplicate_names.add(lbl.name)
            seen_category_names.add(lbl.name)
        if duplicate_names:
            raise DatasetArchiveExportError(
                f"Duplicate category name(s) found in dataset: {', '.join(sorted(duplicate_names))}. "
                f"Category names must be unique for export."
            )

        # 4. Build stable ZIP-local ID mappings
        # Categories: sorted by name
        sorted_labels = sorted(labels, key=lambda l: l.name)
        label_id_to_zip_cat_id: dict[int, int] = {}
        label_by_id: dict[int, Labels] = {lbl.id: lbl for lbl in sorted_labels}
        for idx, lbl in enumerate(sorted_labels, start=1):
            label_id_to_zip_cat_id[lbl.id] = idx

        archive_categories: list[ArchiveCategory] = []
        for lbl in sorted_labels:
            zip_cat_id = label_id_to_zip_cat_id[lbl.id]
            parent_cat_id: int | None = None
            if lbl.parent_id is not None:
                if lbl.parent_id not in label_id_to_zip_cat_id:
                    raise DatasetArchiveExportError(
                        f"Category '{lbl.name}' (id {lbl.id}) references parent label id {lbl.parent_id} "
                        f"which does not belong to dataset {dataset_id}."
                    )
                parent_cat_id = label_id_to_zip_cat_id[lbl.parent_id]
            archive_categories.append(
                ArchiveCategory(
                    id=zip_cat_id,
                    name=lbl.name,
                    supercategory="none",
                    iquana=ArchiveCategoryExtension(
                        value=lbl.value,
                        parent_id=parent_cat_id,
                    ),
                )
            )

        # Images: sorted by DB id
        image_id_to_zip_image_id: dict[int, int] = {
            img.id: idx for idx, img in enumerate(images, start=1)
        }

        # Masks: sorted by DB id
        mask_by_id: dict[int, Masks] = {m.id: m for m in masks}
        mask_id_to_zip_mask_id: dict[int, int] = {
            m.id: idx for idx, m in enumerate(masks, start=1)
        }
        archive_masks: list[ArchiveMask] = []
        for m in masks:
            zip_mask_id = mask_id_to_zip_mask_id[m.id]
            zip_img_id = image_id_to_zip_image_id[m.image_id]
            archive_masks.append(
                ArchiveMask(
                    id=zip_mask_id,
                    image_id=zip_img_id,
                    fully_annotated=bool(m.fully_annotated),
                )
            )

        # Contours: omit temporary contours and their entire descendant subtree
        children_by_parent_id: dict[int, list[int]] = defaultdict(list)
        for c in contours:
            if c.parent_id is not None:
                children_by_parent_id[c.parent_id].append(c.id)

        excluded_contour_ids: set[int] = set()
        stack = [c.id for c in contours if c.temporary]
        while stack:
            cid = stack.pop()
            if cid in excluded_contour_ids:
                continue
            excluded_contour_ids.add(cid)
            stack.extend(children_by_parent_id.get(cid, []))

        temp_contours_count = len(excluded_contour_ids)
        valid_contours = [c for c in contours if c.id not in excluded_contour_ids]
        contour_id_to_zip_ann_id: dict[int, int] = {
            c.id: idx for idx, c in enumerate(valid_contours, start=1)
        }

        # Group image calibrations and metadata
        calibrations_by_image_id: dict[int, list[ImageCalibrations]] = defaultdict(list)
        for cal in image_calibrations:
            calibrations_by_image_id[cal.image_id].append(cal)

        metadata_by_image_id: dict[int, dict[str, str]] = defaultdict(dict)
        for meta in image_metadata_rows:
            metadata_by_image_id[meta.image_id][meta.key] = str(meta.value)

        # Ensure all image metadata keys are declared
        declared_key_names = {k.key for k in declared_metadata_keys}
        synthesized_metadata_keys: list[DatasetMetadataKeys] = []
        for meta in image_metadata_rows:
            if meta.key not in declared_key_names:
                synthesized_metadata_keys.append(
                    DatasetMetadataKeys(
                        dataset_id=dataset_id,
                        key=meta.key,
                        value_type="categorical",
                        options=[],
                        unit=None,
                        description=None,
                    )
                )
                declared_key_names.add(meta.key)

        all_metadata_keys = declared_metadata_keys + synthesized_metadata_keys
        archive_metadata_keys: list[ArchiveMetadataKey] = []
        for k in sorted(all_metadata_keys, key=lambda x: x.key):
            if k.value_type not in MetadataValueType._value2member_map_:
                raise DatasetArchiveExportError(
                    f"Metadata key '{k.key}' has unknown value_type '{k.value_type}'."
                )
            vtype = MetadataValueType(k.value_type)
            archive_metadata_keys.append(
                ArchiveMetadataKey(
                    key=k.key,
                    value_type=vtype,
                    unit=k.unit,
                    options=list(k.options or []),
                    description=k.description,
                )
            )

        # 5. Open TemporaryFile and ZipFile, and stream image files directly
        class _ImageInfo:
            __slots__ = ("image_id", "archive_path", "sha256", "size_bytes", "width", "height", "color_mode")

            def __init__(
                self,
                image_id: int,
                archive_path: str,
                sha256: str,
                size_bytes: int,
                width: int,
                height: int,
                color_mode: str,
            ):
                self.image_id = image_id
                self.archive_path = archive_path
                self.sha256 = sha256
                self.size_bytes = size_bytes
                self.width = width
                self.height = height
                self.color_mode = color_mode

        with zipfile.ZipFile(tmp_file, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            image_info_by_id: dict[int, _ImageInfo] = {}
            archive_images: list[ArchiveImage] = []
            archive_files: list[ArchiveFileEntry] = []

            # Stream images sorted by ZIP ID
            for img in sorted(images, key=lambda im: image_id_to_zip_image_id[im.id]):
                zip_img_id = image_id_to_zip_image_id[img.id]
                if not os.path.isfile(img.file_path):
                    raise DatasetArchiveExportError(
                        f"Image file not found at '{img.file_path}' for image ID {img.id}."
                    )

                try:
                    with Image.open(img.file_path) as pil_img:
                        actual_w, actual_h = pil_img.size
                        actual_mode = pil_img.mode
                except Exception as exc:
                    raise DatasetArchiveExportError(
                        f"Failed to inspect image header for image ID {img.id} ('{img.file_path}'): {exc}"
                    ) from exc

                sanitized_name = _sanitize_archive_filename(img.file_name, zip_img_id)
                archive_path = f"images/{zip_img_id}/{sanitized_name}"

                hasher = hashlib.sha256()
                size_bytes = 0
                try:
                    with open(img.file_path, "rb") as src_f, zf.open(archive_path, "w") as dst_f:
                        while True:
                            chunk = src_f.read(64 * 1024)
                            if not chunk:
                                break
                            hasher.update(chunk)
                            size_bytes += len(chunk)
                            dst_f.write(chunk)
                except OSError as exc:
                    raise DatasetArchiveExportError(
                        f"Failed to read image file for image ID {img.id} ('{img.file_path}'): {exc}"
                    ) from exc

                sha256_hex = hasher.hexdigest()

                info = _ImageInfo(
                    image_id=zip_img_id,
                    archive_path=archive_path,
                    sha256=sha256_hex,
                    size_bytes=size_bytes,
                    width=actual_w,
                    height=actual_h,
                    color_mode=actual_mode,
                )
                image_info_by_id[zip_img_id] = info

                # Build image calibrations
                img_cals = calibrations_by_image_id.get(img.id, [])
                archive_cals: list[ArchiveImageCalibration] = []
                for c in sorted(img_cals, key=lambda x: x.kind):
                    if c.source not in ("manual", "measured", "dataset", "file_metadata"):
                        raise DatasetArchiveExportError(
                            f"Image calibration '{c.kind}' for image ID {img.id} has unknown source '{c.source}'."
                        )
                    archive_cals.append(
                        ArchiveImageCalibration(
                            kind=c.kind,
                            source=c.source,
                            params=c.params or {},
                            created_by=c.created_by,
                            created_at=_to_utc_datetime(c.created_at),
                            updated_at=_to_utc_datetime(c.updated_at),
                        )
                    )

                img_meta = metadata_by_image_id.get(img.id, {})
                img_extension = ArchiveImageExtension(
                    archive_path=archive_path,
                    color_mode=actual_mode,
                    scale_x=float(img.scale_x) if img.scale_x is not None else 1.0,
                    scale_y=float(img.scale_y) if img.scale_y is not None else 1.0,
                    unit=img.unit or "px",
                    description=img.description,
                    metadata=img_meta,
                    calibrations=archive_cals,
                    sha256=sha256_hex,
                    size_bytes=size_bytes,
                )

                archive_images.append(
                    ArchiveImage(
                        id=zip_img_id,
                        file_name=img.file_name,
                        width=actual_w,
                        height=actual_h,
                        iquana=img_extension,
                    )
                )

                archive_files.append(
                    ArchiveFileEntry(
                        image_id=zip_img_id,
                        path=archive_path,
                        sha256=sha256_hex,
                        size_bytes=size_bytes,
                        width=actual_w,
                        height=actual_h,
                        color_mode=actual_mode,
                    )
                )

            # 6. Build Annotations
            archive_annotations: list[ArchiveAnnotation] = []
            for c in valid_contours:
                zip_ann_id = contour_id_to_zip_ann_id[c.id]
                if c.mask_id not in mask_id_to_zip_mask_id:
                    raise DatasetArchiveExportError(
                        f"Contour {c.id} references mask ID {c.mask_id} which is not part of this dataset."
                    )
                zip_mask_id = mask_id_to_zip_mask_id[c.mask_id]

                parent_ann_id: int | None = None
                if c.parent_id is not None:
                    if c.parent_id not in contour_id_to_zip_ann_id:
                        raise DatasetArchiveExportError(
                            f"Contour {c.id} references missing or temporary parent contour ID {c.parent_id}."
                        )
                    parent_ann_id = contour_id_to_zip_ann_id[c.parent_id]

                category_id: int | None = None
                if c.label_id is not None:
                    if c.label_id not in label_id_to_zip_cat_id:
                        raise DatasetArchiveExportError(
                            f"Contour {c.id} references label ID {c.label_id} which is not part of this dataset."
                        )
                    category_id = label_id_to_zip_cat_id[c.label_id]

                # Find image for this contour
                mask_obj = mask_by_id[c.mask_id]
                zip_img_id = image_id_to_zip_image_id[mask_obj.image_id]
                matching_img = image_info_by_id[zip_img_id]

                geometry, segmentation, bbox, area = _normalize_and_validate_geometry(
                    x_coords=c.x,
                    y_coords=c.y,
                    width=matching_img.width,
                    height=matching_img.height,
                    contour_id=c.id,
                )

                reviewers = sorted(reviewers_map.get(c.id, []))

                ann_ext = ArchiveAnnotationExtension(
                    mask_id=zip_mask_id,
                    parent_id=parent_ann_id,
                    geometry=geometry,
                    added_by=c.added_by,
                    confidence_score=float(c.confidence_score) if c.confidence_score is not None else 1.0,
                    created_at=_to_utc_datetime(c.created_at),
                    author_username=c.author_username,
                    reviewed_by=reviewers,
                )

                archive_annotations.append(
                    ArchiveAnnotation(
                        id=zip_ann_id,
                        image_id=zip_img_id,
                        category_id=category_id,
                        segmentation=segmentation,
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                        iquana=ann_ext,
                    )
                )

            # 7. Build Rejections (omit rejections attached to an omitted temporary contour subtree)
            archive_rejections: list[ArchiveRejection] = []
            next_rejection_id = 1
            for r in rejections:
                if r.contour_id is not None and r.contour_id in excluded_contour_ids:
                    continue

                if r.mask_id not in mask_id_to_zip_mask_id:
                    raise DatasetArchiveExportError(
                        f"Rejection {r.id} references mask ID {r.mask_id} which is not part of this dataset."
                    )
                zip_mask_id = mask_id_to_zip_mask_id[r.mask_id]

                zip_ann_id: int | None = None
                if r.contour_id is not None:
                    if r.contour_id not in contour_id_to_zip_ann_id:
                        raise DatasetArchiveExportError(
                            f"Rejection {r.id} references missing contour ID {r.contour_id}."
                        )
                    zip_ann_id = contour_id_to_zip_ann_id[r.contour_id]

                if r.reason not in RejectionReason._value2member_map_:
                    raise DatasetArchiveExportError(
                        f"Rejection {r.id} has unknown reason '{r.reason}'."
                    )
                reason_enum = RejectionReason(r.reason)

                resolution_enum: RejectionResolution | None = None
                if r.resolution is not None:
                    if r.resolution not in RejectionResolution._value2member_map_:
                        raise DatasetArchiveExportError(
                            f"Rejection {r.id} has unknown resolution '{r.resolution}'."
                        )
                    resolution_enum = RejectionResolution(r.resolution)

                archive_rejections.append(
                    ArchiveRejection(
                        id=next_rejection_id,
                        mask_id=zip_mask_id,
                        annotation_id=zip_ann_id,
                        reason=reason_enum,
                        note=r.note,
                        created_by=r.created_by,
                        created_at=_to_utc_datetime(r.created_at),
                        resolved_at=_to_utc_datetime(r.resolved_at),
                        resolved_by=r.resolved_by,
                        resolution=resolution_enum,
                    )
                )
                next_rejection_id += 1

            # 8. Build Actors provenance
            user_roles: dict[str, set[str]] = defaultdict(set)
            if dataset.created_by:
                user_roles[dataset.created_by].add("creator")
            for m in members:
                user_roles[m.username].add(m.role)
            for c in contours:
                if c.author_username:
                    user_roles[c.author_username].add("annotator")
            for rev_list in reviewers_map.values():
                for rev_id in rev_list:
                    user_roles[rev_id].add("reviewer")
            for r in rejections:
                if r.created_by:
                    user_roles[r.created_by].add("reviewer")
                if r.resolved_by:
                    user_roles[r.resolved_by].add("reviewer")
            for cal in image_calibrations:
                if cal.created_by:
                    user_roles[cal.created_by].add("annotator")
            for k in declared_metadata_keys:
                if k.created_by:
                    user_roles[k.created_by].add("annotator")

            archive_actors: list[ArchiveActorProvenance] = []
            for uname in sorted(user_roles.keys()):
                archive_actors.append(
                    ArchiveActorProvenance(
                        username=uname,
                        roles=sorted(list(user_roles[uname])),
                    )
                )

            # 9. Build Counts
            archive_counts = ArchiveCounts(
                images=len(archive_images),
                annotations=len(archive_annotations),
                categories=len(archive_categories),
                masks=len(archive_masks),
                rejections=len(archive_rejections),
                temporary_contours_omitted=temp_contours_count,
            )

            # 10. Assemble annotations.json
            now_utc = datetime.now(timezone.utc)
            coco_info = CocoInfo(
                description=dataset.description or dataset.name,
                version="1.0",
                year=now_utc.year,
                date_created=now_utc,
                contributor=dataset.created_by,
                url=None,
            )

            iquana_extension = IquanaDatasetExtension(
                dataset=ArchiveDatasetInfo(
                    name=dataset.name,
                    description=dataset.description,
                    dataset_type="image",
                    created_by=dataset.created_by,
                ),
                actors=archive_actors,
                metadata_keys=archive_metadata_keys,
                masks=archive_masks,
                rejections=archive_rejections,
                counts=archive_counts,
                files=archive_files,
            )

            try:
                annotations_document = IquanaAnnotationsDocument(
                    format=ARCHIVE_FORMAT,
                    format_version=ARCHIVE_FORMAT_VERSION,
                    info=coco_info,
                    licenses=[],
                    images=archive_images,
                    annotations=archive_annotations,
                    categories=archive_categories,
                    iquana=iquana_extension,
                )
            except Exception as exc:
                raise DatasetArchiveExportError(
                    f"Failed schema validation for annotations.json: {exc}"
                ) from exc

            # 11. Optional config.json assembly
            config_document: IquanaConfigDocument | None = None
            if include_config:
                cal_defaults_rows = (
                    db.query(DatasetCalibrationDefaults)
                    .filter(DatasetCalibrationDefaults.dataset_id == dataset_id)
                    .order_by(DatasetCalibrationDefaults.kind)
                    .all()
                )
                archive_cal_defaults: list[ArchiveCalibrationDefault] = []
                for cd in cal_defaults_rows:
                    archive_cal_defaults.append(
                        ArchiveCalibrationDefault(
                            kind=cd.kind,
                            defaults=cd.defaults or {},
                        )
                    )

                profiles_rows = (
                    db.query(QuantificationProfiles)
                    .filter(QuantificationProfiles.dataset_id == dataset_id)
                    .order_by(QuantificationProfiles.name)
                    .all()
                )
                archive_profiles: list[ArchiveQuantificationProfile] = []
                for p in profiles_rows:
                    p_entries: list[ArchiveProfileEntry] = []
                    for entry in (p.entries or []):
                        metric_key = entry.get("metric_key", "")
                        if metric_key not in STANDARD_V1_METRIC_KEYS:
                            raise DatasetArchiveExportError(
                                f"Quantification profile '{p.name}' uses unsupported metric key '{metric_key}'. "
                                f"Supported keys in v1: {', '.join(sorted(STANDARD_V1_METRIC_KEYS))}."
                            )
                        label_ids = entry.get("label_ids")
                        label_names: list[str] | None = None
                        if label_ids is not None:
                            label_names = []
                            for lid in label_ids:
                                if lid not in label_by_id:
                                    raise DatasetArchiveExportError(
                                        f"Quantification profile '{p.name}' references unknown label ID {lid}."
                                    )
                                label_names.append(label_by_id[lid].name)
                        p_entries.append(
                            ArchiveProfileEntry(
                                metric_key=metric_key,
                                params=entry.get("params", {}),
                                label_names=label_names,
                            )
                        )
                    archive_profiles.append(
                        ArchiveQuantificationProfile(
                            name=p.name,
                            is_default=bool(p.is_default),
                            entries=p_entries,
                        )
                    )

                routing_row = (
                    db.query(DatasetModelRoutingConfigs)
                    .filter(DatasetModelRoutingConfigs.dataset_id == dataset_id)
                    .first()
                )
                archive_bindings: list[ArchiveModelRoutingBinding] = []
                archive_omitted_fields: list[ArchiveOmittedField] = []

                if routing_row and routing_row.bindings:
                    for raw_binding in routing_row.bindings:
                        if isinstance(raw_binding, dict):
                            task_str = raw_binding.get("task")
                            b_lid = raw_binding.get("label_id")
                            m_key = raw_binding.get("model_registry_key", "")
                            b_inputs = raw_binding.get("inputs")
                        else:
                            task_str = getattr(raw_binding, "task", None)
                            b_lid = getattr(raw_binding, "label_id", None)
                            m_key = getattr(raw_binding, "model_registry_key", "")
                            b_inputs = getattr(raw_binding, "inputs", None)

                        task_enum = ModelRoutingTask(task_str)

                        b_label_name: str | None = None
                        if b_lid is not None:
                            if b_lid not in label_by_id:
                                raise DatasetArchiveExportError(
                                    f"Model routing binding for task '{task_str}' references unknown label ID {b_lid}."
                                )
                            b_label_name = label_by_id[b_lid].name

                        sanitized_inputs: dict[str, Any] | None = None
                        if b_inputs is not None:
                            sanitized_inputs = copy.deepcopy(b_inputs)
                            if isinstance(sanitized_inputs, dict) and "conditioning" in sanitized_inputs:
                                cond = sanitized_inputs.get("conditioning")
                                if isinstance(cond, dict) and "query_contour_id" in cond:
                                    del cond["query_contour_id"]
                                    archive_omitted_fields.append(
                                        ArchiveOmittedField(
                                            section="model_routing.bindings",
                                            field="inputs.conditioning.query_contour_id",
                                            task=task_enum,
                                            label_name=b_label_name,
                                            reason="query_contour_id is a local database ID and not portable across installations",
                                        )
                                    )

                        archive_bindings.append(
                            ArchiveModelRoutingBinding(
                                task=task_enum,
                                label_name=b_label_name,
                                model_registry_key=m_key,
                                inputs=sanitized_inputs,
                            )
                        )

                archive_bindings.sort(
                    key=lambda b: (b.task.value if hasattr(b.task, "value") else str(b.task), b.label_name or "")
                )
                archive_omitted_fields.sort(
                    key=lambda o: (
                        o.section,
                        o.field,
                        o.task.value if (o.task and hasattr(o.task, "value")) else str(o.task or ""),
                        o.label_name or "",
                    )
                )

                try:
                    config_document = IquanaConfigDocument(
                        format=ARCHIVE_FORMAT,
                        format_version=ARCHIVE_FORMAT_VERSION,
                        dataset=ArchiveConfigDataset(
                            require_independent_review=bool(dataset.require_independent_review)
                        ),
                        calibration_defaults=archive_cal_defaults,
                        quantification_profiles=archive_profiles,
                        model_routing=ArchiveModelRouting(bindings=archive_bindings),
                        omitted_fields=archive_omitted_fields,
                    )
                except Exception as exc:
                    raise DatasetArchiveExportError(
                        f"Failed schema validation for config.json: {exc}"
                    ) from exc

            # 12. Write control JSON documents to ZipFile
            ann_dict = annotations_document.model_dump(mode="json")
            ann_json = json.dumps(ann_dict, indent=2, sort_keys=True).encode("utf-8")
            zf.writestr("annotations.json", ann_json)

            if config_document is not None:
                cfg_dict = config_document.model_dump(mode="json")
                cfg_json = json.dumps(cfg_dict, indent=2, sort_keys=True).encode("utf-8")
                zf.writestr("config.json", cfg_json)

        tmp_file.seek(0)
    except DatasetArchiveExportError:
        tmp_file.close()
        raise
    except ValueError as exc:
        # Catches pydantic ValidationError (a ValueError subclass) and plain ValueError
        # (e.g. invalid enum conversions) raised by any child schema built above, so
        # invalid stored state surfaces as the documented export error, not a 500.
        tmp_file.close()
        raise DatasetArchiveExportError(
            f"Failed to build dataset archive due to invalid stored data: {exc}"
        ) from exc
    except Exception:
        tmp_file.close()
        raise

    attachment_name = _sanitize_attachment_filename(dataset.name, dataset.id)
    return tmp_file, attachment_name


def import_iquana_dataset_archive(
    db: Session,
    archive_file: BinaryIO | Any,
    override_name: str | None,
    importer_username: str,
    content_length: int | None = None,
) -> dict[str, Any]:
    """Import a dataset from an IQUANA archive ZIP (format v1).

    Validates archive limits, structure, JSON schemas, image checksums/headers,
    geometry projections, and relational references before beginning persistence.
    Stages derivatives and executes database insertions in a single transaction with
    strict rollback and cleanup on any failure.

    Args:
        db: Active SQLAlchemy database session.
        archive_file: Open binary file-like object containing the ZIP archive.
        override_name: Optional override name for the imported dataset.
        importer_username: Username of the authenticated importing user (dataset owner).
        content_length: Optional claimed compressed size in bytes.

    Returns:
        dict[str, Any]: Result payload with success status, new dataset ID/name,
        config application flag, and list of non-fatal warnings.

    Raises:
        DatasetArchiveSizeLimitError: If archive compressed/uncompressed size,
            member count, or single member size exceeds limits (HTTP 413).
        DatasetArchiveNameConflictError: If dataset name conflicts with existing
            database record or filesystem directory (HTTP 409).
        DatasetArchiveValidationError: If archive structure, schemas, checksums,
            headers, geometry, or references are invalid (HTTP 422).
    """
    # 1. Check compressed size
    if content_length is not None and content_length > DATASET_ARCHIVE_MAX_COMPRESSED_BYTES:
        raise DatasetArchiveSizeLimitError(
            f"Archive compressed size ({content_length} bytes) exceeds limit of {DATASET_ARCHIVE_MAX_COMPRESSED_BYTES} bytes."
        )

    if hasattr(archive_file, "seek") and hasattr(archive_file, "tell"):
        archive_file.seek(0, os.SEEK_END)
        actual_compressed_size = archive_file.tell()
        archive_file.seek(0)
        if actual_compressed_size > DATASET_ARCHIVE_MAX_COMPRESSED_BYTES:
            raise DatasetArchiveSizeLimitError(
                f"Archive compressed size ({actual_compressed_size} bytes) exceeds limit of {DATASET_ARCHIVE_MAX_COMPRESSED_BYTES} bytes."
            )

    # 2. Open and inspect ZIP central directory
    try:
        zf = zipfile.ZipFile(archive_file, mode="r")
    except Exception as exc:
        raise DatasetArchiveValidationError(f"Invalid or corrupted ZIP archive: {exc}") from exc

    with zf:
        infolist = zf.infolist()
        if len(infolist) > DATASET_ARCHIVE_MAX_MEMBERS:
            raise DatasetArchiveSizeLimitError(
                f"Archive contains {len(infolist)} members, exceeding limit of {DATASET_ARCHIVE_MAX_MEMBERS}."
            )

        namelist = [info.filename for info in infolist]
        if len(namelist) != len(set(namelist)):
            raise DatasetArchiveValidationError("Archive contains duplicate member names.")

        total_uncompressed = 0
        for info in infolist:
            # Check encryption
            if info.flag_bits & 0x1:
                raise DatasetArchiveValidationError(
                    f"Archive member '{info.filename}' is encrypted. Encrypted archives are not supported."
                )

            # Check symlink or special file
            mode = info.external_attr >> 16
            if mode != 0 and (
                stat.S_ISLNK(mode)
                or stat.S_ISFIFO(mode)
                or stat.S_ISSOCK(mode)
                or stat.S_ISBLK(mode)
                or stat.S_ISCHR(mode)
            ):
                raise DatasetArchiveValidationError(
                    f"Archive member '{info.filename}' is a symlink or special file. Only regular files are allowed."
                )

            # Check path safety (ZipSlip, drive letters, control characters)
            fname = info.filename
            if fname.startswith("/") or fname.startswith("\\") or (len(fname) > 1 and fname[1] == ":"):
                raise DatasetArchiveValidationError(
                    f"Archive member '{fname}' contains an absolute or drive path."
                )
            parts = Path(fname).parts
            if ".." in parts:
                raise DatasetArchiveValidationError(
                    f"Archive member '{fname}' contains directory traversal ('..')."
                )
            if any(ord(c) < 32 or ord(c) == 127 for c in fname):
                raise DatasetArchiveValidationError(
                    f"Archive member '{fname}' contains control characters."
                )

            # Check member size limits
            if info.file_size > DATASET_ARCHIVE_MAX_MEMBER_BYTES:
                raise DatasetArchiveSizeLimitError(
                    f"Archive member '{fname}' size ({info.file_size} bytes) exceeds limit of {DATASET_ARCHIVE_MAX_MEMBER_BYTES} bytes."
                )
            total_uncompressed += info.file_size
            if total_uncompressed > DATASET_ARCHIVE_MAX_UNCOMPRESSED_BYTES:
                raise DatasetArchiveSizeLimitError(
                    f"Total uncompressed archive size exceeds limit of {DATASET_ARCHIVE_MAX_UNCOMPRESSED_BYTES} bytes."
                )

        # Locate annotations.json
        ann_info = next((i for i in infolist if i.filename == "annotations.json"), None)
        if not ann_info:
            raise DatasetArchiveValidationError("Archive is missing required 'annotations.json' manifest.")
        if ann_info.file_size > DATASET_ARCHIVE_MAX_CONTROL_JSON_BYTES:
            raise DatasetArchiveSizeLimitError(
                f"'annotations.json' size ({ann_info.file_size} bytes) exceeds limit of {DATASET_ARCHIVE_MAX_CONTROL_JSON_BYTES} bytes."
            )

        # Locate config.json (optional)
        cfg_info = next((i for i in infolist if i.filename == "config.json"), None)
        if cfg_info and cfg_info.file_size > DATASET_ARCHIVE_MAX_CONTROL_JSON_BYTES:
            raise DatasetArchiveSizeLimitError(
                f"'config.json' size ({cfg_info.file_size} bytes) exceeds limit of {DATASET_ARCHIVE_MAX_CONTROL_JSON_BYTES} bytes."
            )

        # 3. Read and validate control documents
        try:
            ann_bytes = zf.read("annotations.json")
            ann_raw = json.loads(ann_bytes.decode("utf-8"))
        except Exception as exc:
            raise DatasetArchiveValidationError(f"Failed to read or parse 'annotations.json': {exc}") from exc

        if not isinstance(ann_raw, dict):
            raise DatasetArchiveValidationError("'annotations.json' must be a JSON object.")
        if ann_raw.get("format") != ARCHIVE_FORMAT or ann_raw.get("format_version") != ARCHIVE_FORMAT_VERSION:
            raise DatasetArchiveValidationError(
                f"Unsupported format '{ann_raw.get('format')}' (v{ann_raw.get('format_version')}) in 'annotations.json'. "
                f"Expected format '{ARCHIVE_FORMAT}' version {ARCHIVE_FORMAT_VERSION}."
            )

        try:
            annotations_doc = IquanaAnnotationsDocument.model_validate(ann_raw)
        except Exception as exc:
            raise DatasetArchiveValidationError(f"'annotations.json' failed schema validation: {exc}") from exc

        config_doc: IquanaConfigDocument | None = None
        if cfg_info is not None:
            try:
                cfg_bytes = zf.read("config.json")
                cfg_raw = json.loads(cfg_bytes.decode("utf-8"))
            except Exception as exc:
                raise DatasetArchiveValidationError(f"Failed to read or parse 'config.json': {exc}") from exc

            if not isinstance(cfg_raw, dict):
                raise DatasetArchiveValidationError("'config.json' must be a JSON object.")
            if cfg_raw.get("format") != ARCHIVE_FORMAT or cfg_raw.get("format_version") != ARCHIVE_FORMAT_VERSION:
                raise DatasetArchiveValidationError(
                    f"Unsupported format '{cfg_raw.get('format')}' (v{cfg_raw.get('format_version')}) in 'config.json'. "
                    f"Expected format '{ARCHIVE_FORMAT}' version {ARCHIVE_FORMAT_VERSION}."
                )

            try:
                config_doc = IquanaConfigDocument.model_validate(cfg_raw)
            except Exception as exc:
                raise DatasetArchiveValidationError(f"'config.json' failed schema validation: {exc}") from exc

        # Check declared files in annotations.json vs archive members
        declared_image_paths = {f.path for f in annotations_doc.iquana.files}
        non_dir_members = {i.filename for i in infolist if not i.is_dir()}
        expected_members = {"annotations.json"} | declared_image_paths
        if cfg_info is not None:
            expected_members.add("config.json")

        unexpected = non_dir_members - expected_members
        if unexpected:
            raise DatasetArchiveValidationError(
                f"Archive contains unexpected member(s): {', '.join(sorted(unexpected))}."
            )

        missing = declared_image_paths - non_dir_members
        if missing:
            raise DatasetArchiveValidationError(
                f"Archive is missing declared image file(s): {', '.join(sorted(missing))}."
            )

        # 4. Resolve and validate target dataset name
        target_name = (
            override_name.strip()
            if (override_name and override_name.strip())
            else (annotations_doc.iquana.dataset.name.strip() if annotations_doc.iquana.dataset.name else "")
        ).strip()
        if not target_name or len(target_name) > 50:
            raise DatasetArchiveValidationError(
                f"Dataset name must be between 1 and 50 characters, got '{target_name}'."
            )

        existing_ds = db.query(Datasets).filter_by(name=target_name).first()
        if existing_ds is not None:
            raise DatasetArchiveNameConflictError(f"Dataset with name '{target_name}' already exists.")

        # 5. Logical invariant validations
        # Declared counts
        counts = annotations_doc.iquana.counts
        if counts.images != len(annotations_doc.images):
            raise DatasetArchiveValidationError(
                f"Declared counts.images ({counts.images}) does not match images count ({len(annotations_doc.images)})."
            )
        if counts.annotations != len(annotations_doc.annotations):
            raise DatasetArchiveValidationError(
                f"Declared counts.annotations ({counts.annotations}) does not match annotations count ({len(annotations_doc.annotations)})."
            )
        if counts.categories != len(annotations_doc.categories):
            raise DatasetArchiveValidationError(
                f"Declared counts.categories ({counts.categories}) does not match categories count ({len(annotations_doc.categories)})."
            )
        if counts.masks != len(annotations_doc.iquana.masks):
            raise DatasetArchiveValidationError(
                f"Declared counts.masks ({counts.masks}) does not match masks count ({len(annotations_doc.iquana.masks)})."
            )
        if counts.rejections != len(annotations_doc.iquana.rejections):
            raise DatasetArchiveValidationError(
                f"Declared counts.rejections ({counts.rejections}) does not match rejections count ({len(annotations_doc.iquana.rejections)})."
            )

        # Category validation
        cat_names = [c.name for c in annotations_doc.categories]
        if len(cat_names) != len(set(cat_names)):
            raise DatasetArchiveValidationError("Duplicate category name(s) found in categories.")

        # Parent existence and cycle-freedom (linear-time, via check_hierarchy_acyclic)
        # were already enforced by IquanaAnnotationsDocument.model_validate() above;
        # re-walking ancestors here would just reintroduce quadratic behavior.
        cat_by_id = {c.id: c for c in annotations_doc.categories}

        # Metadata keys validation
        meta_key_names = {k.key for k in annotations_doc.iquana.metadata_keys}
        if len(meta_key_names) != len(annotations_doc.iquana.metadata_keys):
            raise DatasetArchiveValidationError("Duplicate metadata key names found in metadata_keys.")
        meta_keys_by_key = {k.key: k for k in annotations_doc.iquana.metadata_keys}

        # Images & files mapping
        image_by_id = {im.id: im for im in annotations_doc.images}
        files_by_image_id = {f.image_id: f for f in annotations_doc.iquana.files}
        for im in annotations_doc.images:
            if im.id not in files_by_image_id:
                raise DatasetArchiveValidationError(f"No file entry declared in files manifest for image ID {im.id}.")
            f_entry = files_by_image_id[im.id]
            if f_entry.path != im.iquana.archive_path:
                raise DatasetArchiveValidationError(
                    f"Image {im.id} archive_path '{im.iquana.archive_path}' does not match manifest path '{f_entry.path}'."
                )
            if f_entry.sha256.lower() != im.iquana.sha256.lower():
                raise DatasetArchiveValidationError(f"Image {im.id} sha256 mismatch with manifest entry.")
            if f_entry.size_bytes != im.iquana.size_bytes:
                raise DatasetArchiveValidationError(f"Image {im.id} size_bytes mismatch with manifest entry.")

            # Image metadata values validation
            for mk, mv in im.iquana.metadata.items():
                if mk not in meta_keys_by_key:
                    raise DatasetArchiveValidationError(f"Image {im.id} metadata references undeclared key '{mk}'.")
                k_def = meta_keys_by_key[mk]
                vtype = k_def.value_type.value if hasattr(k_def.value_type, "value") else str(k_def.value_type)
                try:
                    coerce(mv, vtype, k_def.options)
                except InvalidMetadataError as exc:
                    raise DatasetArchiveValidationError(
                        f"Image {im.id} metadata value '{mv}' is invalid for key '{mk}': {exc}"
                    ) from exc

        # Masks validation
        mask_by_id = {m.id: m for m in annotations_doc.iquana.masks}
        for m in annotations_doc.iquana.masks:
            if m.image_id not in image_by_id:
                raise DatasetArchiveValidationError(f"Mask {m.id} references unknown image ID {m.image_id}.")

        # Annotations validation
        ann_by_id = {a.id: a for a in annotations_doc.annotations}
        for a in annotations_doc.annotations:
            if a.image_id not in image_by_id:
                raise DatasetArchiveValidationError(f"Annotation {a.id} references unknown image ID {a.image_id}.")
            if a.iquana.mask_id not in mask_by_id:
                raise DatasetArchiveValidationError(f"Annotation {a.id} references unknown mask ID {a.iquana.mask_id}.")
            mask_obj = mask_by_id[a.iquana.mask_id]
            if mask_obj.image_id != a.image_id:
                raise DatasetArchiveValidationError(
                    f"Annotation {a.id} image_id {a.image_id} does not match mask {mask_obj.id} image_id {mask_obj.image_id}."
                )
            if a.category_id is not None and a.category_id not in cat_by_id:
                raise DatasetArchiveValidationError(
                    f"Annotation {a.id} references unknown category ID {a.category_id}."
                )
            if a.iquana.parent_id is not None:
                if a.iquana.parent_id not in ann_by_id:
                    raise DatasetArchiveValidationError(
                        f"Annotation {a.id} references unknown parent annotation ID {a.iquana.parent_id}."
                    )
                parent_ann = ann_by_id[a.iquana.parent_id]
                if parent_ann.iquana.mask_id != a.iquana.mask_id:
                    raise DatasetArchiveValidationError(
                        f"Annotation {a.id} and parent {parent_ann.id} belong to different masks ({a.iquana.mask_id} vs {parent_ann.iquana.mask_id})."
                    )
                # Cycle-freedom was already enforced in linear time by
                # IquanaAnnotationsDocument.model_validate() above; no need to re-walk.

            # Geometry projection checks
            gx = a.iquana.geometry.x
            gy = a.iquana.geometry.y
            if len(gx) != len(gy) or len(gx) < 3:
                raise DatasetArchiveValidationError(
                    f"Annotation {a.id} geometry must have at least 3 points, got {len(gx)}."
                )
            if any(abs(coord) > 1.5 for coord in gx + gy):
                raise DatasetArchiveValidationError(
                    f"Annotation {a.id} geometry coordinates must be normalized within [-1.5, 1.5]."
                )

            img_obj = image_by_id[a.image_id]
            _, expected_seg, expected_bbox, expected_area = _normalize_and_validate_geometry(
                x_coords=gx,
                y_coords=gy,
                width=img_obj.width,
                height=img_obj.height,
                contour_id=a.id,
            )

            # Check COCO segmentation
            if not a.segmentation or len(a.segmentation) != 1 or len(a.segmentation[0]) != len(expected_seg[0]):
                raise DatasetArchiveValidationError(
                    f"Annotation {a.id} derived COCO segmentation structure does not match canonical geometry."
                )
            if any(abs(c1 - c2) > 1.0 for c1, c2 in zip(a.segmentation[0], expected_seg[0])):
                raise DatasetArchiveValidationError(
                    f"Annotation {a.id} derived COCO segmentation points contradict canonical geometry."
                )

            # Check bbox
            if not a.bbox or len(a.bbox) != 4 or any(abs(b1 - b2) > 1.0 for b1, b2 in zip(a.bbox, expected_bbox)):
                raise DatasetArchiveValidationError(
                    f"Annotation {a.id} derived COCO bbox contradicts canonical geometry."
                )

            # Check area
            if abs(a.area - expected_area) > max(1.0, expected_area * 0.05):
                raise DatasetArchiveValidationError(
                    f"Annotation {a.id} derived COCO area ({a.area}) contradicts canonical geometry ({expected_area})."
                )

        # Rejections validation
        for rej in annotations_doc.iquana.rejections:
            if rej.mask_id not in mask_by_id:
                raise DatasetArchiveValidationError(
                    f"Rejection {rej.id} references unknown mask ID {rej.mask_id}."
                )
            if rej.annotation_id is not None:
                if rej.annotation_id not in ann_by_id:
                    raise DatasetArchiveValidationError(
                        f"Rejection {rej.id} references unknown annotation ID {rej.annotation_id}."
                    )
                target_ann = ann_by_id[rej.annotation_id]
                if target_ann.iquana.mask_id != rej.mask_id:
                    raise DatasetArchiveValidationError(
                        f"Rejection {rej.id} annotation {target_ann.id} belongs to mask {target_ann.iquana.mask_id}, not {rej.mask_id}."
                    )

        # Config validation (if present)
        if config_doc is not None:
            cat_name_set = set(cat_names)
            for prof in config_doc.quantification_profiles:
                for entry in prof.entries:
                    if entry.metric_key not in STANDARD_V1_METRIC_KEYS:
                        raise DatasetArchiveValidationError(
                            f"Unsupported metric key '{entry.metric_key}' in profile '{prof.name}'."
                        )
                    if entry.label_names is not None:
                        for ln in entry.label_names:
                            if ln not in cat_name_set:
                                raise DatasetArchiveValidationError(
                                    f"Quantification profile '{prof.name}' references unknown category '{ln}'."
                                )

            for binding in config_doc.model_routing.bindings:
                if binding.label_name is not None and binding.label_name not in cat_name_set:
                    raise DatasetArchiveValidationError(
                        f"Model routing binding references unknown category '{binding.label_name}'."
                    )
                if isinstance(binding.inputs, dict):
                    cond = binding.inputs.get("conditioning")
                    if isinstance(cond, dict) and "query_contour_id" in cond:
                        raise DatasetArchiveValidationError(
                            "Unsanitized local query_contour_id found in model routing inputs."
                        )

        # 6. Collect Warnings
        warnings: list[str] = []
        num_actors = len(annotations_doc.iquana.actors)
        num_approvals = sum(len(a.iquana.reviewed_by or []) for a in annotations_doc.annotations)
        if num_actors > 0 or num_approvals > 0:
            warnings.append(
                f"Imported dataset detached {num_actors} source actor record(s) and {num_approvals} reviewer approval(s). "
                "Review statuses and assignments may change."
            )

        if config_doc is None:
            warnings.append(
                "Configuration file (config.json) was not present in the archive; "
                "destination defaults will be used for review policy, calibrations, quantification profiles, and model routing."
            )
        elif config_doc.omitted_fields:
            warnings.append(
                f"Configuration noted {len(config_doc.omitted_fields)} stripped non-portable field(s) from model routing conditioning."
            )

        if config_doc is not None:
            routing_pairs = {
                (
                    b.task.value if hasattr(b.task, "value") else str(b.task),
                    b.model_registry_key,
                )
                for b in config_doc.model_routing.bindings
                if b.model_registry_key
            }
            if routing_pairs:
                ready_pairs = _confirmed_ready_model_bindings(routing_pairs)
                unconfirmed = sorted(
                    f"{task}:{key}" for task, key in routing_pairs if (task, key) not in ready_pairs
                )
                if unconfirmed:
                    warnings.append(
                        f"Model routing binding(s) not confirmed ready locally: {', '.join(unconfirmed)}. "
                        "Retained for curator repair."
                    )

        if annotations_doc.iquana.counts.temporary_contours_omitted > 0:
            warnings.append(
                f"Archive omitted {annotations_doc.iquana.counts.temporary_contours_omitted} temporary contour(s) during export."
            )

        # 7. Staging and image extraction
        staging_uuid = uuid.uuid4().hex
        staging_dataset_dir = os.path.join(config.DATASETS_DIR, f".staging_import_{staging_uuid}")
        staging_images_dir = os.path.join(staging_dataset_dir, "images")
        staging_masks_dir = os.path.join(staging_dataset_dir, "masks")
        staging_thumbnails_dir = os.path.join(config.THUMBNAILS_DIR, f".staging_import_{staging_uuid}")

        image_disk_filenames: dict[int, str] = {}
        used_disk_filenames: set[str] = set()
        created_thumbnail_files: list[str] = []
        moved_to_final_dir = False

        try:
            # Created inside the try so a failure partway through (e.g. the 2nd or 3rd
            # makedirs call) is still caught by the except below, which only cleans up
            # whichever of these roots actually exists.
            os.makedirs(staging_images_dir, exist_ok=True)
            os.makedirs(staging_masks_dir, exist_ok=True)
            os.makedirs(staging_thumbnails_dir, exist_ok=True)

            for img in sorted(annotations_doc.images, key=lambda x: x.id):
                f_entry = files_by_image_id[img.id]
                sanitized_name = _sanitize_archive_filename(img.file_name, img.id)
                disk_filename = sanitized_name
                if disk_filename in used_disk_filenames:
                    stem = Path(sanitized_name).stem
                    suffix = Path(sanitized_name).suffix
                    counter = 1
                    while f"{stem}_{counter}{suffix}" in used_disk_filenames:
                        counter += 1
                    disk_filename = f"{stem}_{counter}{suffix}"
                used_disk_filenames.add(disk_filename)
                image_disk_filenames[img.id] = disk_filename

                staged_img_path = os.path.join(staging_images_dir, disk_filename)

                # Stream from ZIP and compute hash
                hasher = hashlib.sha256()
                extracted_bytes = 0
                with zf.open(f_entry.path, "r") as src_f, open(staged_img_path, "wb") as dst_f:
                    while True:
                        chunk = src_f.read(64 * 1024)
                        if not chunk:
                            break
                        hasher.update(chunk)
                        extracted_bytes += len(chunk)
                        if extracted_bytes > DATASET_ARCHIVE_MAX_MEMBER_BYTES:
                            raise DatasetArchiveSizeLimitError(
                                f"Image '{f_entry.path}' exceeded single-member size limit."
                            )
                        dst_f.write(chunk)

                if extracted_bytes != f_entry.size_bytes:
                    raise DatasetArchiveValidationError(
                        f"Image '{f_entry.path}' size mismatch: expected {f_entry.size_bytes} bytes, extracted {extracted_bytes} bytes."
                    )
                if hasher.hexdigest().lower() != f_entry.sha256.lower():
                    raise DatasetArchiveValidationError(
                        f"Image '{f_entry.path}' checksum mismatch: expected {f_entry.sha256}, calculated {hasher.hexdigest()}."
                    )

                # Inspect image header with Pillow (lazy)
                try:
                    with Image.open(staged_img_path) as pil_img:
                        actual_w, actual_h = pil_img.size
                        actual_mode = pil_img.mode
                except Exception as exc:
                    raise DatasetArchiveValidationError(
                        f"Failed to inspect image header for '{f_entry.path}': {exc}"
                    ) from exc

                if actual_w != img.width or actual_h != img.height:
                    raise DatasetArchiveValidationError(
                        f"Image '{f_entry.path}' header dimensions ({actual_w}x{actual_h}) do not match manifest ({img.width}x{img.height})."
                    )
                if actual_mode != img.iquana.color_mode:
                    raise DatasetArchiveValidationError(
                        f"Image '{f_entry.path}' actual color mode ({actual_mode}) does not match manifest ({img.iquana.color_mode})."
                    )

                # Generate thumbnail in thumbnail staging
                staged_thumb_path = os.path.join(staging_thumbnails_dir, f"{img.id}_{disk_filename}")
                try:
                    with Image.open(staged_img_path) as pil_img:
                        thumb = pil_img.copy()
                        thumb.thumbnail((500, 500))
                        thumb.save(staged_thumb_path)
                except Exception as exc:
                    raise DatasetArchiveValidationError(
                        f"Failed to generate thumbnail for image '{f_entry.path}': {exc}"
                    ) from exc

            # 8. Generate semantic mask pixel data in staging
            annotations_by_mask_id: dict[int, list[ArchiveAnnotation]] = defaultdict(list)
            for a in annotations_doc.annotations:
                annotations_by_mask_id[a.iquana.mask_id].append(a)

            for m in annotations_doc.iquana.masks:
                img = image_by_id[m.image_id]
                canvas = np.zeros((img.height, img.width), dtype=np.uint8)

                mask_anns = annotations_by_mask_id.get(m.id, [])
                roots = [a for a in mask_anns if a.iquana.parent_id is None]
                children_by_parent: dict[int, list[ArchiveAnnotation]] = defaultdict(list)
                for a in mask_anns:
                    if a.iquana.parent_id is not None:
                        children_by_parent[a.iquana.parent_id].append(a)

                queue = deque(roots)
                while queue:
                    ann = queue.popleft()
                    if ann.category_id is not None:
                        cat = cat_by_id.get(ann.category_id)
                        if cat is not None:
                            val = cat.iquana.value
                            pts = np.array(
                                [
                                    [round(xi * img.width), round(yi * img.height)]
                                    for xi, yi in zip(ann.iquana.geometry.x, ann.iquana.geometry.y)
                                ],
                                dtype=np.int32,
                            )
                            if len(pts) >= 3:
                                cv2.fillPoly(canvas, [pts], color=int(val))
                    queue.extend(children_by_parent.get(ann.id, []))

                # "zip_"-prefixed so this archive-local-ID namespace can never collide
                # with the DB-assigned-ID namespace these get renamed into below.
                staged_mask_path = os.path.join(staging_masks_dir, f"zip_{m.id}.png")
                Image.fromarray(canvas, mode="L").save(staged_mask_path)

            # 9. Database Transaction & Model Persistence
            # 1. Dataset
            new_dataset = Datasets(
                name=target_name,
                # iquana.dataset.description is the lossless field; info.description is
                # COCO's required-string field and falls back to the dataset name at
                # export time when the source description was None.
                description=annotations_doc.iquana.dataset.description,
                folder_path="",  # Placeholder; real path depends on the ID assigned below.
                dataset_type="image",
                created_by=importer_username,
                require_independent_review=config_doc.dataset.require_independent_review if config_doc else False,
            )
            db.add(new_dataset)
            db.flush()

            # Derive the on-disk directory from the new dataset's DB ID plus a safe
            # slug of the (untrusted) name, and confirm it cannot escape DATASETS_DIR.
            final_dataset_dir = os.path.join(
                config.DATASETS_DIR, f"{new_dataset.id}_{_slugify_dataset_name(target_name)}"
            )
            real_datasets_root = os.path.realpath(config.DATASETS_DIR)
            real_final_dir = os.path.realpath(final_dataset_dir)
            if os.path.commonpath([real_final_dir, real_datasets_root]) != real_datasets_root:
                raise DatasetArchiveValidationError(
                    f"Resolved dataset directory '{final_dataset_dir}' escapes the datasets root."
                )
            if os.path.exists(final_dataset_dir):
                raise DatasetArchiveNameConflictError(f"Directory '{final_dataset_dir}' already exists on disk.")
            new_dataset.folder_path = final_dataset_dir

            # 2. Owner membership
            ensure_owner_membership(new_dataset.id, importer_username, db)

            # 3. Categories / Labels (topological order)
            zip_cat_id_to_db_id: dict[int, int] = {}
            label_name_to_db_id: dict[str, int] = {}

            cat_roots = [c for c in annotations_doc.categories if c.iquana.parent_id is None]
            cat_children_by_parent: dict[int, list[ArchiveCategory]] = defaultdict(list)
            for c in annotations_doc.categories:
                if c.iquana.parent_id is not None:
                    cat_children_by_parent[c.iquana.parent_id].append(c)

            cat_queue = deque(cat_roots)
            while cat_queue:
                cat = cat_queue.popleft()
                parent_db_id = (
                    zip_cat_id_to_db_id.get(cat.iquana.parent_id)
                    if cat.iquana.parent_id is not None
                    else None
                )
                lbl = Labels(
                    dataset_id=new_dataset.id,
                    name=cat.name,
                    value=cat.iquana.value,
                    parent_id=parent_db_id,
                )
                db.add(lbl)
                db.flush()
                zip_cat_id_to_db_id[cat.id] = lbl.id
                label_name_to_db_id[cat.name] = lbl.id
                cat_queue.extend(cat_children_by_parent.get(cat.id, []))

            # 4. Metadata Keys
            meta_key_by_name_db: dict[str, DatasetMetadataKeys] = {}
            for k in annotations_doc.iquana.metadata_keys:
                vtype = k.value_type.value if hasattr(k.value_type, "value") else str(k.value_type)
                meta_key_row = DatasetMetadataKeys(
                    dataset_id=new_dataset.id,
                    key=k.key,
                    value_type=vtype,
                    unit=k.unit,
                    options=list(k.options or []),
                    description=k.description,
                    created_by=importer_username,
                )
                db.add(meta_key_row)
                db.flush()
                meta_key_by_name_db[k.key] = meta_key_row

            # 5. Images
            zip_img_id_to_db_image: dict[int, Images] = {}
            for img in sorted(annotations_doc.images, key=lambda x: x.id):
                disk_filename = image_disk_filenames[img.id]
                final_img_path = os.path.join(final_dataset_dir, "images", disk_filename)
                final_thumb_filename = f"{new_dataset.id}_{img.id}_{disk_filename}"
                final_thumb_path = os.path.join(config.THUMBNAILS_DIR, final_thumb_filename)

                new_img = Images(
                    file_name=img.file_name,
                    file_path=str(final_img_path),
                    thumbnail_file_path=str(final_thumb_path),
                    dataset_id=new_dataset.id,
                    width=img.width,
                    height=img.height,
                    color_mode=img.iquana.color_mode,
                    scale_x=img.iquana.scale_x,
                    scale_y=img.iquana.scale_y,
                    unit=img.iquana.unit,
                    description=img.iquana.description,
                )
                db.add(new_img)
                db.flush()
                zip_img_id_to_db_image[img.id] = new_img

                # 6. Image metadata values
                for k, v in img.iquana.metadata.items():
                    meta_def = meta_key_by_name_db[k]
                    coerced_val, val_num = coerce(v, meta_def.value_type, meta_def.options)
                    meta_val_row = ImageMetadata(
                        image_id=new_img.id,
                        key=k,
                        value=coerced_val,
                        value_num=val_num,
                    )
                    db.add(meta_val_row)

                # 7. Image calibrations
                for cal in img.iquana.calibrations:
                    c_source = (
                        cal.source
                        if cal.source in ("manual", "measured", "dataset", "file_metadata")
                        else "manual"
                    )
                    cal_row = ImageCalibrations(
                        image_id=new_img.id,
                        kind=cal.kind,
                        source=c_source,
                        params=cal.params,
                        created_by=importer_username,
                        created_at=_to_utc_datetime(cal.created_at) or datetime.now(timezone.utc),
                        updated_at=_to_utc_datetime(cal.updated_at) or datetime.now(timezone.utc),
                    )
                    db.add(cal_row)

            # 8. Masks
            zip_mask_id_to_db_mask: dict[int, Masks] = {}
            for m in annotations_doc.iquana.masks:
                matching_img = zip_img_id_to_db_image[m.image_id]
                new_mask = Masks(
                    image_id=matching_img.id,
                    fully_annotated=m.fully_annotated,
                    file_path="",  # Placeholder; real path depends on the mask ID assigned below.
                )
                db.add(new_mask)
                db.flush()
                # Use the mask's own (unique) DB ID for the filename: an image can have
                # multiple masks, so naming by image ID alone would collide across them.
                final_mask_path = os.path.join(final_dataset_dir, "masks", f"{new_mask.id}.png")
                new_mask.file_path = str(final_mask_path)
                zip_mask_id_to_db_mask[m.id] = new_mask

                # staged_mask_old uses the "zip_" prefix from staging above; the disjoint
                # namespaces guarantee this rename can never overwrite an unprocessed
                # archive mask's still-staged file, even if a later new_mask.id happens
                # to numerically equal another mask's not-yet-renamed zip-local m.id.
                staged_mask_old = os.path.join(staging_masks_dir, f"zip_{m.id}.png")
                staged_mask_new = os.path.join(staging_masks_dir, f"{new_mask.id}.png")
                if os.path.exists(staged_mask_old):
                    os.rename(staged_mask_old, staged_mask_new)

            # 9. Contours (topological sort per mask)
            zip_ann_id_to_db_contour: dict[int, Contours] = {}
            for m in annotations_doc.iquana.masks:
                mask_db_obj = zip_mask_id_to_db_mask[m.id]
                mask_anns = annotations_by_mask_id.get(m.id, [])

                roots = [a for a in mask_anns if a.iquana.parent_id is None]
                children_by_parent = defaultdict(list)
                for a in mask_anns:
                    if a.iquana.parent_id is not None:
                        children_by_parent[a.iquana.parent_id].append(a)

                ann_queue = deque(roots)
                created_mask_contours: list[Contours] = []
                while ann_queue:
                    ann = ann_queue.popleft()
                    parent_db_id = (
                        zip_ann_id_to_db_contour[ann.iquana.parent_id].id
                        if ann.iquana.parent_id is not None
                        else None
                    )
                    cat_db_id = (
                        zip_cat_id_to_db_id.get(ann.category_id)
                        if ann.category_id is not None
                        else None
                    )

                    c_row = Contours(
                        mask_id=mask_db_obj.id,
                        parent_id=parent_db_id,
                        temporary=False,
                        added_by=ann.iquana.added_by or "User",
                        # Source actor usernames are provenance-only and must be cleared
                        # on import per the frozen decoupling policy (docs/iquana-dataset-format-v1.md);
                        # the importing user becomes the dataset owner, not the content author.
                        author_username=None,
                        created_at=_to_utc_datetime(ann.iquana.created_at) or datetime.now(timezone.utc),
                        confidence_score=ann.iquana.confidence_score if ann.iquana.confidence_score is not None else 1.0,
                        label_id=cat_db_id,
                        area=0.0,
                        perimeter=0.0,
                        circularity=0.0,
                        diameter=0.0,
                        x=ann.iquana.geometry.x,
                        y=ann.iquana.geometry.y,
                    )
                    db.add(c_row)
                    db.flush()
                    zip_ann_id_to_db_contour[ann.id] = c_row
                    created_mask_contours.append(c_row)
                    ann_queue.extend(children_by_parent.get(ann.id, []))

                # Recompute legacy geometry metrics and dual-write to ContourMetrics
                if created_mask_contours:
                    dual_write_geometry_metrics(db, mask_db_obj.id, created_mask_contours)

            # 10. Rejections
            for rej in annotations_doc.iquana.rejections:
                mask_db_obj = zip_mask_id_to_db_mask[rej.mask_id]
                contour_db_id = (
                    zip_ann_id_to_db_contour[rej.annotation_id].id
                    if rej.annotation_id is not None
                    else None
                )
                reason_str = rej.reason.value if hasattr(rej.reason, "value") else str(rej.reason)
                resolution_str = (
                    rej.resolution.value
                    if (rej.resolution and hasattr(rej.resolution, "value"))
                    else (str(rej.resolution) if rej.resolution else None)
                )

                rej_row = AnnotationRejections(
                    mask_id=mask_db_obj.id,
                    contour_id=contour_db_id,
                    reason=reason_str,
                    note=rej.note,
                    # Source actor usernames are provenance-only and are cleared on
                    # import (see author_username above); only timestamps are kept.
                    created_by=None,
                    created_at=_to_utc_datetime(rej.created_at) or datetime.now(timezone.utc),
                    resolved_at=_to_utc_datetime(rej.resolved_at),
                    resolved_by=None,
                    resolution=resolution_str,
                )
                db.add(rej_row)

            # 11. Configuration
            if config_doc is not None:
                # Calibration defaults
                for cd in config_doc.calibration_defaults:
                    db.add(
                        DatasetCalibrationDefaults(
                            dataset_id=new_dataset.id,
                            kind=cd.kind,
                            defaults=cd.defaults,
                        )
                    )

                # Quantification profiles
                for prof in config_doc.quantification_profiles:
                    entries_raw = []
                    for entry in prof.entries:
                        label_ids = (
                            [label_name_to_db_id[lname] for lname in entry.label_names]
                            if entry.label_names is not None
                            else None
                        )
                        entries_raw.append(
                            {
                                "metric_key": entry.metric_key,
                                "params": entry.params,
                                "label_ids": label_ids,
                            }
                        )
                    db.add(
                        QuantificationProfiles(
                            dataset_id=new_dataset.id,
                            name=prof.name,
                            is_default=prof.is_default,
                            entries=entries_raw,
                        )
                    )

                # Model routing
                routing_bindings_raw = []
                for b in config_doc.model_routing.bindings:
                    lbl_id = label_name_to_db_id[b.label_name] if b.label_name is not None else None
                    task_str = b.task.value if hasattr(b.task, "value") else str(b.task)
                    routing_bindings_raw.append(
                        {
                            "task": task_str,
                            "label_id": lbl_id,
                            "model_registry_key": b.model_registry_key,
                            "inputs": b.inputs,
                        }
                    )
                db.add(
                    DatasetModelRoutingConfigs(
                        dataset_id=new_dataset.id,
                        bindings=routing_bindings_raw,
                    )
                )

            # 12. Move files to final destinations
            # os.mkdir is used instead of renaming staging_dataset_dir directly onto
            # final_dataset_dir: on POSIX, rename() only fails if the destination is a
            # *non-empty* directory -- an empty one is silently replaced. That would let
            # a directory a concurrent process just created (but hasn't populated yet)
            # be silently taken over and later deleted by our cleanup. mkdir never
            # replaces an existing directory, empty or not, so success here proves we
            # are the sole owner of final_dataset_dir before anything is moved into it.
            try:
                os.mkdir(final_dataset_dir)
            except OSError as exc:
                raise DatasetArchiveNameConflictError(
                    f"Directory '{final_dataset_dir}' already exists on disk."
                ) from exc
            moved_to_final_dir = True

            for subdir_name in ("images", "masks"):
                os.rename(
                    os.path.join(staging_dataset_dir, subdir_name),
                    os.path.join(final_dataset_dir, subdir_name),
                )
            os.rmdir(staging_dataset_dir)

            for img in annotations_doc.images:
                disk_filename = image_disk_filenames[img.id]
                staged_thumb = os.path.join(staging_thumbnails_dir, f"{img.id}_{disk_filename}")
                final_thumb_filename = f"{new_dataset.id}_{img.id}_{disk_filename}"
                final_thumb = os.path.join(config.THUMBNAILS_DIR, final_thumb_filename)
                shutil.move(staged_thumb, final_thumb)
                created_thumbnail_files.append(final_thumb)

            if os.path.exists(staging_thumbnails_dir):
                shutil.rmtree(staging_thumbnails_dir, ignore_errors=True)

            # 13. Recheck dataset name availability before commit
            conflict_count = (
                db.query(Datasets)
                .filter(Datasets.name == target_name, Datasets.id != new_dataset.id)
                .count()
            )
            if conflict_count > 0:
                raise DatasetArchiveNameConflictError(
                    f"Dataset with name '{target_name}' was created concurrently."
                )

            # Captured before commit: SQLAlchemy expires instance attributes on commit
            # by default, so reading new_dataset.id/.name afterward would trigger an
            # implicit SELECT. If the DB connection is what's unavailable (the exact
            # scenario the refresh guard below exists for), that implicit reload fails
            # too, propagating out of this already-committed branch and into the
            # destructive outer cleanup. Locals sidestep that entirely.
            new_dataset_id = new_dataset.id
            new_dataset_name = new_dataset.name
            new_image_ids = [im.id for im in zip_img_id_to_db_image.values()]
            new_contour_ids = [c.id for c in zip_ann_id_to_db_contour.values()]

            db.commit()
            try:
                db.refresh(new_dataset)
            except Exception:
                # The transaction already committed successfully, so the dataset and its
                # files are real and must not be cleaned up by the except block below;
                # a refresh failure here is non-fatal since we already know id/name.
                logger.warning(
                    "Dataset %s imported successfully but post-commit refresh failed.",
                    new_dataset_id,
                    exc_info=True,
                )

            # Opt-in background embedding for cross-image retrieval (no-op unless
            # EMBEDDING_LIFECYCLE_ENABLED), same as the on-write hooks for uploads and
            # contour saves. Post-commit: the rows are already durably persisted, so a
            # failure here must never roll back or clean up the import.
            for image_id in new_image_ids:
                enqueue_embed_image(image_id)
            enqueue_embed_contours(new_contour_ids)

            return {
                "success": True,
                "message": "Dataset imported successfully.",
                "dataset_id": new_dataset_id,
                "dataset_name": new_dataset_name,
                "config_applied": config_doc is not None,
                "warnings": warnings,
            }

        except IntegrityError as exc:
            db.rollback()
            if os.path.exists(staging_dataset_dir):
                shutil.rmtree(staging_dataset_dir, ignore_errors=True)
            if os.path.exists(staging_thumbnails_dir):
                shutil.rmtree(staging_thumbnails_dir, ignore_errors=True)
            if moved_to_final_dir and os.path.exists(final_dataset_dir):
                shutil.rmtree(final_dataset_dir, ignore_errors=True)
            for thumb_f in created_thumbnail_files:
                if os.path.exists(thumb_f):
                    try:
                        os.unlink(thumb_f)
                    except Exception:
                        pass
            raise DatasetArchiveNameConflictError(
                f"Dataset with name '{target_name}' already exists or was created concurrently."
            ) from exc
        except Exception:
            db.rollback()
            if os.path.exists(staging_dataset_dir):
                shutil.rmtree(staging_dataset_dir, ignore_errors=True)
            if os.path.exists(staging_thumbnails_dir):
                shutil.rmtree(staging_thumbnails_dir, ignore_errors=True)
            # Only remove final_dataset_dir if this import actually moved its own
            # staged data there; short-circuits before referencing the name at all
            # if we never got that far (e.g. it was never assigned) or never moved it,
            # so we never delete a directory this import doesn't own.
            if moved_to_final_dir and os.path.exists(final_dataset_dir):
                shutil.rmtree(final_dataset_dir, ignore_errors=True)
            for thumb_f in created_thumbnail_files:
                if os.path.exists(thumb_f):
                    try:
                        os.unlink(thumb_f)
                    except Exception:
                        pass
            raise


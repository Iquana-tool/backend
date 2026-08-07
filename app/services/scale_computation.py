"""Scale computation service for the IQuana backend.

Provides pure service functions (no FastAPI objects, no HTTP handling) for
reading and writing image scale data. Routes call these functions and map
domain exceptions to HTTP status codes.

All DB mutations live here, not in the route layer (RULE 2).
Domain-specific exceptions are raised for expected error conditions so routes
can catch them precisely without silently swallowing unexpected bugs (RULE 1).
"""
import logging
from typing import Iterable

import numpy as np
from sqlalchemy.orm import Session

from app.database.contour_metrics import ContourMetrics
from app.database.images import Images
from app.database.masks import Masks
from app.database.contours import Contours
from app.exceptions import DatasetNotFoundError, ImageNotFoundError, InvalidScaleError
from app.services.calibration import store as calibration_store
from app.services.calibration.registry import SCALE_STALE_METRIC_KEYS, CalibrationSource

logger = logging.getLogger(__name__)

# Geometry and contextual metric keys that are scale-dependent.
# Circularity is dimensionless (scale cancels out) and appearance metrics are
# pixel-color based — neither group needs recomputation when scale changes.
# Scale is one calibration kind among several now, so the list lives with the
# other kinds' dependency declarations in the calibration registry.
_SCALE_DEPENDENT_METRIC_KEYS: tuple[str, ...] = SCALE_STALE_METRIC_KEYS


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_image_scale(db: Session, image_id: int) -> dict:
    """Return the current scale information for an image.

    Args:
        db: SQLAlchemy session.
        image_id: Primary key of the image row.

    Returns:
        dict with keys ``scale_x``, ``scale_y``, ``unit``.

    Raises:
        ImageNotFoundError: If ``image_id`` does not exist.
    """
    image = db.query(Images).filter_by(id=image_id).first()
    if not image:
        raise ImageNotFoundError(f"Image {image_id} not found.")
    return {
        "scale_x": image.scale_x,
        "scale_y": image.scale_y,
        "unit": image.unit,
    }


# ---------------------------------------------------------------------------
# Write — single image
# ---------------------------------------------------------------------------

def set_image_scale(
        db: Session,
        image_id: int,
        scale_x: float,
        scale_y: float,
        unit: str,
        source: str = CalibrationSource.MANUAL,
        username: str | None = None,
) -> dict:
    """Persist a new physical scale for an image and mark dependent metrics stale.

    The function validates inputs first, then updates the image row, then marks
    only scale-dependent metric rows as stale so they are recomputed on the next
    batch quantification run. Scale-independent metrics (circularity, appearance)
    are intentionally left untouched.

    It also records the change as a ``scale`` row in ``image_calibrations``, which
    is where the Calibrate tab reads provenance from. The ``images`` columns stay
    the value — see ``app.services.calibration.registry`` — so the two cannot
    disagree; the row only adds who set it, when, and how.

    Args:
        db: SQLAlchemy session (caller does NOT need to commit separately —
            this function commits after the update).
        image_id: Primary key of the image to update.
        scale_x: Physical size of one pixel along the x-axis in ``unit`` units.
            Must be positive.
        scale_y: Physical size of one pixel along the y-axis in ``unit`` units.
            Must be positive.
        unit: Length unit string (e.g. ``"mm"``, ``"µm"``). Must not be empty.
        source: How the scale was obtained, see ``CalibrationSource``.
        username: Account making the change, recorded on the calibration row.

    Returns:
        dict with keys ``scale_x``, ``scale_y``, ``unit``.

    Raises:
        InvalidScaleError: If scale values are non-positive or unit is empty.
        ImageNotFoundError: If ``image_id`` does not exist.
    """
    if not unit or not unit.strip():
        raise InvalidScaleError("Unit must be a non-empty string (e.g. 'mm', 'µm').")
    if scale_x <= 0 or scale_y <= 0:
        raise InvalidScaleError(
            f"Scale values must be positive (got scale_x={scale_x}, scale_y={scale_y})."
        )

    image = db.query(Images).filter_by(id=image_id).first()
    if not image:
        raise ImageNotFoundError(f"Image {image_id} not found.")

    if image.scale_x != scale_x or image.scale_y != scale_y or image.unit != unit:
        logger.info(
            "Updating scale for image %d: scale_x=%s→%s, scale_y=%s→%s, unit=%s→%s",
            image_id, image.scale_x, scale_x, image.scale_y, scale_y, image.unit, unit,
        )
        image.scale_x = scale_x
        image.scale_y = scale_y
        image.unit = unit
        _mark_scale_dependent_metrics_stale(db, image_id)
        calibration_store.upsert(
            db, image_id, "scale",
            {"scale_x": scale_x, "scale_y": scale_y, "unit": unit},
            source, username,
        )
        db.commit()
    else:
        logger.debug("Scale for image %d unchanged; no update performed.", image_id)

    return {"scale_x": scale_x, "scale_y": scale_y, "unit": unit}


# ---------------------------------------------------------------------------
# Write — drawn-line calibration
# ---------------------------------------------------------------------------

def set_scale_from_drawn_line(
        db: Session,
        image_id: int,
        p1: tuple[float, float],
        p2: tuple[float, float],
        known_distance: float,
        unit: str,
        username: str | None = None,
) -> dict:
    """Compute the physical scale from a user-drawn calibration line and persist it.

    The user draws a line between two known points on the image (e.g. two ruler
    ticks) and provides the real-world distance between them. This function derives
    the mm/pixel (or chosen unit/pixel) scale and delegates to :func:`set_image_scale`.

    Args:
        db: SQLAlchemy session.
        image_id: Primary key of the image.
        p1: (x1, y1) coordinates of the first point in *image pixels* (not
            normalised fractions).
        p2: (x2, y2) coordinates of the second point in *image pixels*.
        known_distance: Real-world distance between the two points in ``unit``.
            Must be positive.
        unit: Length unit of ``known_distance`` (e.g. ``"mm"``).
        username: Account drawing the line, recorded on the calibration row.

    Returns:
        dict with keys ``scale_x``, ``scale_y``, ``unit``, ``pixel_distance``.

    Raises:
        InvalidScaleError: If ``known_distance`` ≤ 0 or the two points are identical.
        ImageNotFoundError: If ``image_id`` does not exist.
    """
    if known_distance <= 0:
        raise InvalidScaleError(
            f"known_distance must be positive (got {known_distance})."
        )

    scale_per_pixel = compute_pixel_scale_from_points(p1, p2, known_distance)
    result = set_image_scale(
        db, image_id, scale_per_pixel, scale_per_pixel, unit,
        source=CalibrationSource.MEASURED, username=username,
    )
    pixel_distance = float(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))
    return {**result, "pixel_distance": pixel_distance}


# ---------------------------------------------------------------------------
# Write — bulk apply to dataset
# ---------------------------------------------------------------------------

def apply_scale_to_dataset(
        db: Session,
        dataset_id: int,
        scale_x: float,
        scale_y: float,
        unit: str,
        username: str | None = None,
) -> dict:
    """Apply the same physical scale to all images in a dataset.

    This is the common case for datasets captured at a fixed magnification: one
    calibration is measured once and propagated to every image. Validation is
    performed once before any DB writes.

    Args:
        db: SQLAlchemy session.
        dataset_id: Primary key of the dataset.
        scale_x: Physical size of one pixel along x in ``unit`` units. Must be > 0.
        scale_y: Physical size of one pixel along y in ``unit`` units. Must be > 0.
        unit: Length unit string.

    Returns:
        dict with ``images_updated`` (count) and the applied scale/unit values.

    Raises:
        InvalidScaleError: If scale values are non-positive or unit is empty.
        DatasetNotFoundError: If no images belong to the dataset (or dataset missing).
    """
    # Validate once before touching the DB.
    if not unit or not unit.strip():
        raise InvalidScaleError("Unit must be a non-empty string (e.g. 'mm', 'µm').")
    if scale_x <= 0 or scale_y <= 0:
        raise InvalidScaleError(
            f"Scale values must be positive (got scale_x={scale_x}, scale_y={scale_y})."
        )

    images = db.query(Images).filter_by(dataset_id=dataset_id).all()
    if not images:
        raise DatasetNotFoundError(
            f"Dataset {dataset_id} not found or contains no images."
        )

    updated = 0
    for image in images:
        if image.scale_x != scale_x or image.scale_y != scale_y or image.unit != unit:
            image.scale_x = scale_x
            image.scale_y = scale_y
            image.unit = unit
            _mark_scale_dependent_metrics_stale(db, image.id)
            updated += 1
        # Written for every image, not only the changed ones: an image that
        # already happened to carry these values still gains the provenance row
        # saying the value came from a dataset-wide apply.
        calibration_store.upsert(
            db, image.id, "scale",
            {"scale_x": scale_x, "scale_y": scale_y, "unit": unit},
            CalibrationSource.DATASET, username,
        )

    # Unconditional: even when no image's value changed, every one of them just
    # gained (or had refreshed) its scale provenance row.
    db.commit()
    if updated:
        logger.info(
            "Applied scale (scale_x=%s, scale_y=%s, unit=%s) to %d/%d images in dataset %d.",
            scale_x, scale_y, unit, updated, len(images), dataset_id,
        )

    return {
        "images_updated": updated,
        "images_total": len(images),
        "scale_x": scale_x,
        "scale_y": scale_y,
        "unit": unit,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mark_scale_dependent_metrics_stale(db: Session, image_id: int) -> int:
    """Mark scale-dependent metric rows as stale for all contours of an image.

    Only geometry and contextual metrics are updated — circularity (dimensionless)
    and appearance metrics (pixel-color based) are unaffected by a scale change
    and are intentionally left untouched to avoid unnecessary recomputation.

    Args:
        db: SQLAlchemy session (caller is responsible for commit).
        image_id: Primary key of the image whose contour metrics should be invalidated.

    Returns:
        The number of rows marked stale.
    """
    contour_ids: list[int] = [
        row[0]
        for row in (
            db.query(Contours.id)
            .join(Masks, Masks.id == Contours.mask_id)
            .filter(Masks.image_id == image_id)
            .all()
        )
    ]
    if not contour_ids:
        return 0

    updated = (
        db.query(ContourMetrics)
        .filter(
            ContourMetrics.contour_id.in_(contour_ids),
            ContourMetrics.metric_key.in_(_SCALE_DEPENDENT_METRIC_KEYS),
        )
        .update({ContourMetrics.stale: True}, synchronize_session=False)
    )
    logger.debug(
        "Marked %d metric rows stale for image %d after scale change.",
        updated, image_id,
    )
    return updated


def compute_pixel_scale_from_points(
        p1: tuple[float, float],
        p2: tuple[float, float],
        known_distance: float,
) -> float:
    """Derive the real-world scale (unit/pixel) from two points and a known distance.

    Args:
        p1: (x1, y1) coordinates of the first point in image pixels.
        p2: (x2, y2) coordinates of the second point in image pixels.
        known_distance: Real-world distance between the points in the caller's unit.

    Returns:
        Scale value in unit/pixel (identical for x and y — isotropic assumption).

    Raises:
        InvalidScaleError: If the two points are identical (zero pixel distance).
    """
    x1, y1 = p1
    x2, y2 = p2
    pixel_distance = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    if pixel_distance == 0:
        raise InvalidScaleError(
            "Cannot compute scale from two identical points. "
            "Choose two distinct points on the image."
        )
    return known_distance / pixel_distance

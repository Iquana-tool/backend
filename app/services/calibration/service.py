"""Calibration orchestration: read, set, clear, propagate, sample, apply.

Everything here is kind-agnostic — it asks the registry what a kind means and
does the same thing for all of them. That is the whole point of the split: a new
calibration kind is a registry entry, not a change to this module.

The two contracts worth knowing:

* **Nothing is baked into the image.** Setting a calibration stores parameters and
  marks the dependent measurements stale. The pixel transform happens at compute
  time in :func:`load_calibrated_image_rgb`, which the appearance-metric batch
  uses in place of the raw loader. A calibration can therefore be revised or
  removed at any time, and the original file is never touched.
* **Sampling replays the pipeline.** :func:`sample_patch` applies every stage
  ordered *before* the kind being calibrated, so a colour patch is read in the
  same tone space the colour gains will later act on. Sampling raw and applying
  corrected is the subtle way this kind of feature goes wrong.
"""
from logging import getLogger
from typing import Iterable

import numpy as np
from sqlalchemy.orm import Session

from app.database.dataset_calibration_defaults import DatasetCalibrationDefaults
from app.database.image_calibrations import ImageCalibrations
from app.database.images import Images
from app.exceptions import DatasetNotFoundError, ImageNotFoundError, InvalidCalibrationError
from app.services.calibration import cards, registry, store, strategies
from app.services.calibration.registry import CalibrationKind, CalibrationSource

logger = getLogger(__name__)

#: Default radius, in image pixels, of the disc a reference patch is averaged over.
#: Large enough to survive sensor noise and JPEG blocking, small enough to sit
#: inside a typical colour-chart patch at full resolution.
DEFAULT_SAMPLE_RADIUS = 8

#: Ceiling on the sample radius, so a request cannot turn into a full-image mean.
MAX_SAMPLE_RADIUS = 256


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def _require_image(db: Session, image_id: int) -> Images:
    image = db.query(Images).filter_by(id=image_id).first()
    if image is None:
        raise ImageNotFoundError(f"Image {image_id} not found.")
    return image


def kinds_metadata() -> list[dict]:
    """Registry metadata for every kind, in pipeline (and display) order."""
    return [kind.as_dict() for kind in registry.all_kinds()]


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _read_params(image: Images, kind: CalibrationKind, row) -> dict | None:
    """The current parameters of one kind, or None when it is not calibrated.

    A kind with a ``read`` hook (scale) is answered from its mirror columns, so an
    image calibrated before this table existed still reads as calibrated.
    """
    if kind.read is not None:
        return kind.read(image)
    return dict(row.params) if row is not None and row.params else None


def _entry(image: Images, kind: CalibrationKind, row) -> dict:
    """One calibration card's worth of state: registry metadata plus what is set."""
    params = _read_params(image, kind, row)
    entry = kind.as_dict()
    entry.update({
        "calibrated": params is not None,
        "params": params,
        "description": kind.describe(params) if params else None,
        # Provenance comes from the row even when the value comes from the mirror
        # columns; a legacy image has the value but no row, hence the None-safety.
        "source": row.source if row is not None else None,
        "created_by": row.created_by if row is not None else None,
        "updated_at": row.updated_at.isoformat() if row is not None and row.updated_at else None,
    })
    return entry


def get_calibration_state(db: Session, image_id: int) -> dict:
    """Every kind's state for one image — calibrated or not, with provenance.

    Returns one entry per *registered* kind rather than per stored row, so the
    client can render a card for a kind that has never been set without having to
    know the registry itself. Each entry also carries the dataset's default
    strategy for that kind, which is what an uncalibrated card should open on.
    """
    image = _require_image(db, image_id)
    rows = {row.kind: row for row in store.list_rows(db, image_id)}
    defaults = get_dataset_defaults(db, image.dataset_id)

    entries = []
    for kind in registry.all_kinds():
        entry = _entry(image, kind, rows.get(kind.key))
        entry["dataset_defaults"] = defaults.get(kind.key) or _fallback_defaults(kind)
        entries.append(entry)

    return {
        "image_id": image_id,
        "dataset_id": image.dataset_id,
        "calibrations": entries,
        "calibrated_count": sum(1 for e in entries if e["calibrated"]),
        "total_count": len(entries),
    }


def calibrated_counts(
        db: Session,
        images: Iterable[Images],
) -> dict[int, tuple[int, int]]:
    """``{image_id: (kinds calibrated, kinds registered)}`` for a batch of images.

    The batch form exists because the Calibrate phase of every image in a dataset
    is wanted at once (the gallery, the progress bars), and asking
    :func:`get_calibration_state` per image would be one query per image plus a
    dataset-defaults lookup nobody reads. Here it is two queries total.

    Counting *registered* kinds, not stored rows, is what makes "fully calibrated"
    mean the same thing for every image: a kind nobody has set yet is a kind that
    is missing, not a kind that does not exist. Legacy rows for retired kinds are
    ignored for the same reason.
    """
    images = list(images)
    if not images:
        return {}

    kinds = registry.all_kinds()
    image_ids = [image.id for image in images]
    rows_by_image: dict[int, dict[str, object]] = {image_id: {} for image_id in image_ids}
    for row in (
            db.query(ImageCalibrations)
            .filter(ImageCalibrations.image_id.in_(image_ids))
            .all()
    ):
        rows_by_image.setdefault(row.image_id, {})[row.kind] = row

    counts: dict[int, tuple[int, int]] = {}
    for image in images:
        rows = rows_by_image.get(image.id, {})
        calibrated = sum(
            1 for kind in kinds
            if _read_params(image, kind, rows.get(kind.key)) is not None
        )
        counts[image.id] = (calibrated, len(kinds))
    return counts


# ---------------------------------------------------------------------------
# Dataset defaults
# ---------------------------------------------------------------------------

def _fallback_defaults(kind: CalibrationKind) -> dict | None:
    """The configuration a kind starts from when the dataset has set none."""
    if not kind.strategy_keys:
        return None
    strategy_key = kind.strategy_keys[0]
    defaults = {"strategy": strategy_key}
    if strategies.get_strategy(strategy_key).requires_card:
        defaults["card"] = cards.DEFAULT_CARD
    return defaults


def validate_defaults(kind_key: str, defaults: dict) -> dict:
    """Check a strategy configuration without needing any measurements.

    Deliberately separate from the kind's ``normalize``: choosing *how* a dataset
    will calibrate has to be possible before anything has been measured, so this
    validates the choice alone.
    """
    kind = registry.get_kind(kind_key)
    if not kind.strategy_keys:
        raise InvalidCalibrationError(
            f"Calibration kind '{kind_key}' has only one way of being measured, so "
            f"it has nothing to configure."
        )

    strategy_key = str((defaults or {}).get("strategy") or kind.strategy_keys[0])
    if strategy_key not in kind.strategy_keys:
        raise InvalidCalibrationError(
            f"Strategy '{strategy_key}' does not apply to '{kind_key}'. Available: "
            f"{', '.join(kind.strategy_keys)}."
        )
    strategy = strategies.get_strategy(strategy_key)

    validated = {"strategy": strategy_key}
    if strategy.requires_card:
        card_key = str((defaults or {}).get("card") or cards.DEFAULT_CARD)
        validated["card"] = cards.get_card(card_key).key
    if (defaults or {}).get("fit_model") is not None:
        fit_model = str(defaults["fit_model"])
        allowed = [model["key"] for model in strategy.fit_models]
        if allowed and fit_model not in allowed:
            raise InvalidCalibrationError(
                f"Unknown fit model '{fit_model}' for '{strategy_key}'. "
                f"Available: {', '.join(allowed)}."
            )
        validated["fit_model"] = fit_model
    return validated


def get_dataset_defaults(db: Session, dataset_id: int) -> dict[str, dict]:
    """Every kind's default strategy configuration for a dataset, keyed by kind."""
    rows = (
        db.query(DatasetCalibrationDefaults)
        .filter_by(dataset_id=dataset_id)
        .all()
    )
    return {row.kind: dict(row.defaults or {}) for row in rows}


def set_dataset_defaults(
        db: Session,
        dataset_id: int,
        kind_key: str,
        defaults: dict,
        username: str | None = None,
) -> dict:
    """Set how a dataset calibrates one kind. Does not touch existing calibrations.

    Changing the default is a choice about future work, not a correction to past
    work — silently re-estimating every image against a new strategy would discard
    measurements nobody asked to discard.
    """
    validated = validate_defaults(kind_key, defaults)

    row = (
        db.query(DatasetCalibrationDefaults)
        .filter_by(dataset_id=dataset_id, kind=kind_key)
        .first()
    )
    if row is None:
        row = DatasetCalibrationDefaults(
            dataset_id=dataset_id, kind=kind_key,
            defaults=validated, updated_by=username,
        )
        db.add(row)
    else:
        row.defaults = validated
        row.updated_by = username
    db.commit()

    logger.info("Dataset %d now calibrates '%s' with %s (set by %s).",
                dataset_id, kind_key, validated, username)
    return {"dataset_id": dataset_id, "kind": kind_key, "defaults": validated}


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def set_calibration(
        db: Session,
        image_id: int,
        kind_key: str,
        params: dict,
        source: str = CalibrationSource.MANUAL,
        username: str | None = None,
) -> dict:
    """Validate, store and apply one calibration to one image.

    Raises:
        UnknownCalibrationKindError: If ``kind_key`` is not registered.
        ImageNotFoundError: If ``image_id`` does not exist.
        InvalidCalibrationError / InvalidScaleError: If the parameters are invalid
            for the kind (the scale kind keeps the older exception type).
    """
    kind = registry.get_kind(kind_key)
    image = _require_image(db, image_id)
    normalized = kind.normalize(params or {})

    if kind.persist is not None:
        kind.persist(db, image, normalized)
    store.upsert(db, image_id, kind_key, normalized, source, username)
    invalidated = store.mark_metrics_stale_for_image(db, image_id, kind.stale_metric_keys)
    db.commit()

    logger.info("Set %s calibration on image %d (source=%s, by=%s); %d metric row(s) stale.",
                kind_key, image_id, source, username, invalidated)
    return {
        "image_id": image_id,
        "kind": kind_key,
        "params": normalized,
        "description": kind.describe(normalized),
        "source": source,
        "metrics_invalidated": invalidated,
    }


def clear_calibration(db: Session, image_id: int, kind_key: str) -> dict:
    """Remove a calibration, restoring the uncalibrated reading of the image.

    For a kind with mirror columns (scale) this also resets them to the
    uncalibrated default — ``unit = "px"`` at unit scale — so the value and the
    row disappear together rather than leaving the old scale silently in force.
    """
    kind = registry.get_kind(kind_key)
    image = _require_image(db, image_id)

    removed = store.delete_row(db, image_id, kind_key)
    had_mirror = False
    if kind.read is not None and kind.read(image) is not None:
        had_mirror = True
        if kind.key == "scale":
            image.scale_x = 1.0
            image.scale_y = 1.0
            image.unit = "px"

    if not removed and not had_mirror:
        return {"image_id": image_id, "kind": kind_key, "cleared": False,
                "metrics_invalidated": 0}

    invalidated = store.mark_metrics_stale_for_image(db, image_id, kind.stale_metric_keys)
    db.commit()
    logger.info("Cleared %s calibration on image %d; %d metric row(s) stale.",
                kind_key, image_id, invalidated)
    return {"image_id": image_id, "kind": kind_key, "cleared": True,
            "metrics_invalidated": invalidated}


def apply_to_dataset(
        db: Session,
        dataset_id: int,
        kind_key: str,
        params: dict,
        username: str | None = None,
) -> dict:
    """Apply one calibration to every image in a dataset, in one transaction.

    The common case for a dataset captured in one session: the reference is
    measured once and propagated. The rows are written with source ``dataset`` so
    a later reader can tell a propagated calibration from a measured one — the
    distinction matters when the acquisition conditions were not in fact constant.

    Raises:
        DatasetNotFoundError: If the dataset has no images (or does not exist).
    """
    kind = registry.get_kind(kind_key)
    if not kind.dataset_propagatable:
        raise ValueError(f"Calibration kind '{kind_key}' cannot be applied dataset-wide.")
    normalized = kind.normalize(params or {})

    images = db.query(Images).filter_by(dataset_id=dataset_id).all()
    if not images:
        raise DatasetNotFoundError(
            f"Dataset {dataset_id} not found or contains no images."
        )

    invalidated = 0
    for image in images:
        if kind.persist is not None:
            kind.persist(db, image, normalized)
        store.upsert(db, image.id, kind_key, normalized, CalibrationSource.DATASET, username)
        invalidated += store.mark_metrics_stale_for_image(db, image.id, kind.stale_metric_keys)
    db.commit()

    logger.info("Applied %s calibration to %d image(s) in dataset %d; %d metric row(s) stale.",
                kind_key, len(images), dataset_id, invalidated)
    return {
        "dataset_id": dataset_id,
        "kind": kind_key,
        "params": normalized,
        "description": kind.describe(normalized),
        "images_updated": len(images),
        "metrics_invalidated": invalidated,
    }


# ---------------------------------------------------------------------------
# The pixel pipeline
# ---------------------------------------------------------------------------

def _active_stages(db: Session, image_id: int, below_order: int | None = None
                   ) -> list[tuple[CalibrationKind, dict]]:
    """The pixel-transforming calibrations set on an image, in pipeline order.

    ``below_order`` restricts the result to stages that run *before* a given kind,
    which is what patch sampling needs so a reference is read in the same space
    the calibration being measured will act on.
    """
    rows = {row.kind: row for row in store.list_rows(db, image_id)}
    stages: list[tuple[CalibrationKind, dict]] = []
    for kind in registry.pixel_stages():
        row = rows.get(kind.key)
        if row is None or not row.params:
            continue
        if below_order is not None and kind.order >= below_order:
            continue
        stages.append((kind, dict(row.params)))
    return stages


def apply_calibration_pipeline(
        db: Session,
        image_id: int,
        rgb: np.ndarray,
        below_order: int | None = None,
) -> np.ndarray:
    """Run an image's pixel calibrations over an RGB array.

    Works in float32 across the whole chain and converts back to ``uint8`` once at
    the end, so a two-stage pipeline does not quantise twice.

    A stage that raises is logged and skipped rather than allowed to abort the
    caller — the appearance batch processes a whole dataset, and one malformed
    calibration row should cost that one correction, not every image after it.
    """
    stages = _active_stages(db, image_id, below_order=below_order)
    if not stages:
        return rgb

    work = rgb.astype(np.float32)
    for kind, params in stages:
        try:
            work = kind.apply(work, params)
        except Exception:
            logger.exception(
                "Calibration stage '%s' failed on image %s; skipping that stage. "
                "Params: %s", kind.key, image_id, params,
            )
    return np.clip(work, 0.0, 255.0).round().astype(np.uint8)


def load_calibrated_image_rgb(db: Session, image: Images) -> np.ndarray | None:
    """Load an image's pixels with its radiometric calibrations applied.

    The loader the appearance-metric batch uses. Returns ``None`` for an
    unreadable file, exactly like the raw loader it wraps, so a single missing
    file still only skips one image.

    Imported lazily to keep ``quantification`` importable without this package
    (the raw loader has no calibration dependency).
    """
    from app.services.quantification import load_image_rgb

    raw = load_image_rgb(image)
    if raw is None:
        return None
    return apply_calibration_pipeline(db, image.id, raw)


# ---------------------------------------------------------------------------
# Reference sampling
# ---------------------------------------------------------------------------

def _measure_disc(pixels: np.ndarray, x: float, y: float, radius: int) -> dict:
    """Statistics of one disc of an already-loaded, already-corrected image.

    Split out from :func:`sample_patch` so a whole reference card can be read
    without decoding and correcting the image once per patch — twenty patches
    through the single-point path would mean twenty full decodes.
    """
    height, width = pixels.shape[:2]
    cx, cy = float(x), float(y)
    if not (0 <= cx < width and 0 <= cy < height):
        raise InvalidCalibrationError(
            f"Sample point ({cx:.1f}, {cy:.1f}) lies outside the {width}x{height} image."
        )

    x0, x1 = max(0, int(cx - radius)), min(width, int(cx + radius) + 1)
    y0, y1 = max(0, int(cy - radius)), min(height, int(cy + radius) + 1)
    window = pixels[y0:y1, x0:x1].astype(np.float32)

    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside = ((xs - cx) ** 2 + (ys - cy) ** 2) <= radius ** 2
    if not inside.any():
        # Only reachable for a sub-pixel radius at the very edge of the frame.
        inside[:] = True
    selected = window[inside]

    # The median is the value callers should use, and what the original MATLAB
    # utility used: a reference patch is meant to be uniform, so a specular glint,
    # a speck of sediment, or a sliver of the card's border caught in the window
    # is contamination rather than signal. The mean and the spread come back too,
    # because a large gap between them is exactly how a bad patch announces itself.
    median_rgb = np.median(selected, axis=0)
    mean_rgb = selected.mean(axis=0)
    std_rgb = selected.std(axis=0)
    return {
        "x": cx,
        "y": cy,
        "radius": radius,
        "n_pixels": int(selected.shape[0]),
        "median_rgb": [float(v) for v in median_rgb],
        "mean_rgb": [float(v) for v in mean_rgb],
        "std_rgb": [float(v) for v in std_rgb],
        # Unweighted channel average, matching how the response transform treats
        # the three channels independently rather than through a luminance
        # weighting that no stage would then honour.
        "median_intensity": float(median_rgb.mean()),
        "mean_intensity": float(mean_rgb.mean()),
    }


def _prepare_sampling(db: Session, image_id: int, radius: int, for_kind: str | None):
    """Load an image and apply the stages that precede ``for_kind``. Shared setup."""
    from app.services.quantification import load_image_rgb

    image = _require_image(db, image_id)
    if radius < 1 or radius > MAX_SAMPLE_RADIUS:
        raise InvalidCalibrationError(
            f"Sample radius must lie in 1-{MAX_SAMPLE_RADIUS} px (got {radius})."
        )

    raw = load_image_rgb(image)
    if raw is None:
        raise ImageNotFoundError(
            f"Image {image_id} could not be read from disk; cannot sample a patch."
        )

    below_order = registry.get_kind(for_kind).order if for_kind else None
    stages = _active_stages(db, image_id, below_order=below_order)
    pixels = apply_calibration_pipeline(db, image_id, raw, below_order=below_order)
    return pixels, [kind.key for kind, _ in stages]


def sample_patches(
        db: Session,
        image_id: int,
        points: list[tuple[float, float]],
        radius: int = DEFAULT_SAMPLE_RADIUS,
        for_kind: str | None = None,
) -> dict:
    """Read every patch of a reference card in one pass.

    The card path: one decode, one pipeline application, N disc measurements. Same
    semantics as :func:`sample_patch` per point, including the stage replay.

    Raises:
        InvalidCalibrationError: If no points were given, or any lies outside the
            image. A partial card reading is worse than none — it would be saved
            as a complete one.
    """
    if not points:
        raise InvalidCalibrationError("No sample points were given.")
    if len(points) > 256:
        raise InvalidCalibrationError(
            f"Refusing to sample {len(points)} points in one request; no reference "
            f"card has that many patches."
        )

    pixels, stages = _prepare_sampling(db, image_id, radius, for_kind)
    samples = [_measure_disc(pixels, x, y, radius) for x, y in points]
    return {"image_id": image_id, "samples": samples, "stages_applied": stages}


def sample_patch(
        db: Session,
        image_id: int,
        x: float,
        y: float,
        radius: int = DEFAULT_SAMPLE_RADIUS,
        for_kind: str | None = None,
) -> dict:
    """Average a disc of pixels around (x, y), for use as a calibration reference.

    Sampling happens server-side against the original file rather than in the
    client against the rendered ``<img>``: the browser shows a scaled, possibly
    re-encoded copy, and reading a reference patch off that would calibrate the
    display rather than the data.

    Args:
        db: SQLAlchemy session.
        image_id: The image to sample.
        x: Sample centre, in image pixels.
        y: Sample centre, in image pixels.
        radius: Radius of the averaged disc, in image pixels.
        for_kind: The kind being calibrated. When given, every calibration stage
            ordered before it is applied first, so the returned value is in the
            space that kind's parameters will act on.

    Returns:
        dict with ``median_rgb``, ``mean_rgb``, ``std_rgb``, ``median_intensity``,
        ``mean_intensity``, ``n_pixels`` and ``stages_applied``.

    Raises:
        ImageNotFoundError: If the image row or its file is missing.
        InvalidCalibrationError: If the point lies outside the image, or the radius
            is not usable.
    """
    pixels, stages = _prepare_sampling(db, image_id, radius, for_kind)
    return {
        "image_id": image_id,
        **_measure_disc(pixels, x, y, radius),
        "stages_applied": stages,
    }

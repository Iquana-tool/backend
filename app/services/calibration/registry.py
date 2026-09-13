"""The calibration kind registry — what a calibration *is*, one entry per kind.

Everything that varies between calibration kinds is declared here, so adding a
kind (flat-field, lens distortion, perspective, densitometry, ...) is one
:class:`CalibrationKind` entry plus a UI card, with no schema migration and no
change to the service, the routes or the quantification hook.

A kind declares:

``normalize``
    Validates the caller's parameters and fills in whatever can be derived from
    them. Raises :class:`~app.exceptions.InvalidCalibrationError` — or, for scale,
    :class:`~app.exceptions.InvalidScaleError`, so the older /scale routes keep
    their existing 422 mapping.
``apply``
    The pixel transform, or ``None`` for kinds that do not touch pixels (scale
    only changes what a pixel *means*, not what it contains). Takes and returns
    float32 RGB in the 0-255 domain; the pipeline converts to and from uint8 once
    at the ends, so a chain of stages does not accumulate rounding error.
``order``
    Position in the pixel pipeline, and the display order of the UI cards. Not
    cosmetic: the sampling endpoint replays every lower-ordered stage before
    returning a patch value, so a reference is always read in the space the
    calibration being measured will act on.
``stale_metric_keys``
    Which ``contour_metrics`` rows a change invalidates. This is the mechanism
    that keeps stored measurements honest — the same one a scale change has
    always used, generalised.
``strategies``
    Ways of *estimating* this kind's parameters (see ``strategies.py``). A kind
    with more than one lets the user pick; the choice is stored on the row, and a
    dataset-level default means it is normally picked once, not per image.
``persist`` / ``read``
    Optional column mirror. Only scale has them: it predates this table and lives
    in ``images.scale_x`` / ``scale_y`` / ``unit``, which remain the read path for
    the quantification contexts and the COCO export. Treating those columns as the
    value and the row as its provenance means the two can never drift, and that
    scale set through the older ``/scale`` routes — or on an image that predates
    this table entirely — still shows up here as calibrated. The generic upsert,
    staleness and commit still run for such a kind; the hooks only add the mirror.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.exceptions import (
    InvalidCalibrationError,
    InvalidScaleError,
    UnknownCalibrationKindError,
)
from app.schemas.permissions import Permission
from app.services.calibration import cards, strategies

#: Geometry and contextual metrics whose value depends on the physical scale.
#: Circularity is dimensionless (the scale cancels) and appearance metrics are
#: pixel-value based, so neither is invalidated by a scale change.
SCALE_STALE_METRIC_KEYS: tuple[str, ...] = (
    "area",
    "perimeter",
    "max_diameter",
    "nn_distance",
    "mean_knn_distance",
)

#: Metrics computed from the image's pixel values, and therefore invalidated by
#: anything that changes those values (the tone/colour response, and any future
#: radiometric kind such as flat-field correction).
APPEARANCE_STALE_METRIC_KEYS: tuple[str, ...] = (
    "mean_color_rgb",
    "mean_color_lab",
    "mean_intensity",
)

#: Kinds that existed before tone and colour were unified into ``response``.
#: Rows carrying them are inert (the service iterates registered kinds), but they
#: are real user data until ``scripts/migrate_response_calibration.py`` converts
#: them, so the orphan-pruning migration must leave them alone.
LEGACY_KINDS: frozenset[str] = frozenset({"intensity", "color"})


class CalibrationSource:
    """How a calibration was obtained. Stored on the row; shown in the UI."""

    MANUAL = "manual"                  # parameters typed in directly
    MEASURED = "measured"              # derived from a reference in this frame
    DATASET = "dataset"                # propagated from a dataset-wide apply
    FILE_METADATA = "file_metadata"    # read out of the image file's own metadata

    ALL = (MANUAL, MEASURED, DATASET, FILE_METADATA)


@dataclass(frozen=True)
class ParamField:
    """One parameter of a kind, described well enough for a client to render it."""

    name: str
    label: str
    type: str                      # 'number' | 'text' | 'vector3' | 'enum'
    help: str = ""
    unit: str | None = None
    choices: tuple[str, ...] | None = None
    optional: bool = False

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "help": self.help,
            "unit": self.unit,
            "choices": list(self.choices) if self.choices else None,
            "optional": self.optional,
        }


@dataclass(frozen=True)
class CalibrationKind:
    """Everything the generic calibration machinery needs to know about a kind."""

    key: str
    label: str
    summary: str
    order: int
    permission: Permission
    stale_metric_keys: tuple[str, ...]
    normalize: Callable[[dict], dict]
    describe: Callable[[dict], str]
    fields: tuple[ParamField, ...] = ()
    apply: Callable[[np.ndarray, dict], np.ndarray] | None = None
    persist: Callable | None = None
    read: Callable | None = None
    dataset_propagatable: bool = True
    #: Strategy keys this kind can be estimated by, first being the default.
    #: Empty for a kind measured exactly one way (scale draws a line).
    strategy_keys: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        """Registry metadata for ``GET /calibration/kinds``."""
        strategy_list = [
            strategies.get_strategy(key).as_dict() for key in self.strategy_keys
        ]
        needs_card = any(entry["requires_card"] for entry in strategy_list)
        return {
            "kind": self.key,
            "label": self.label,
            "summary": self.summary,
            "order": self.order,
            "permission": str(self.permission),
            "affects_metrics": list(self.stale_metric_keys),
            "transforms_pixels": self.apply is not None,
            "dataset_propagatable": self.dataset_propagatable,
            "fields": [field.as_dict() for field in self.fields],
            "strategies": strategy_list,
            "default_strategy": self.strategy_keys[0] if self.strategy_keys else None,
            "cards": [card.as_dict() for card in cards.all_cards()] if needs_card else [],
            "default_card": cards.DEFAULT_CARD if needs_card else None,
        }


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------

def _normalize_scale(params: dict) -> dict:
    """Validate a physical scale. Raises InvalidScaleError, matching /scale."""
    unit = params.get("unit")
    if not unit or not str(unit).strip():
        raise InvalidScaleError("Unit must be a non-empty string (e.g. 'mm', 'µm').")
    try:
        scale_x = float(params["scale_x"])
        scale_y = float(params["scale_y"])
    except (KeyError, TypeError, ValueError):
        raise InvalidScaleError("scale_x and scale_y must be numbers.")
    if not np.isfinite(scale_x) or not np.isfinite(scale_y):
        raise InvalidScaleError("scale_x and scale_y must be finite.")
    if scale_x <= 0 or scale_y <= 0:
        raise InvalidScaleError(
            f"Scale values must be positive (got scale_x={scale_x}, scale_y={scale_y})."
        )
    return {"scale_x": scale_x, "scale_y": scale_y, "unit": str(unit).strip()}


def _describe_scale(params: dict) -> str:
    return f"1 px = {params.get('scale_x', 0):.6g} {params.get('unit', '?')}"


def _read_scale(image) -> dict | None:
    """Read the scale calibration out of the legacy ``images`` columns.

    ``unit == "px"`` is the default an image is created with and means "no
    real-world scale has been set", so it reads as uncalibrated rather than as a
    scale of one pixel per pixel.
    """
    unit = (getattr(image, "unit", None) or "px").strip()
    if unit == "px":
        return None
    scale_x = float(image.scale_x or 0)
    scale_y = float(image.scale_y or 0)
    if scale_x <= 0 or scale_y <= 0:
        return None
    return {"scale_x": scale_x, "scale_y": scale_y, "unit": unit}


def _persist_scale(db, image, params: dict) -> None:
    """Mirror a scale calibration into the legacy ``images`` columns.

    Only the mirror — the service still upserts the calibration row, marks the
    dependent metrics stale and commits, exactly as it does for every other kind.
    Does not commit.
    """
    image.scale_x = params["scale_x"]
    image.scale_y = params["scale_y"]
    image.unit = params["unit"]


# ---------------------------------------------------------------------------
# response (tone + colour, formerly the separate `intensity` and `color` kinds)
# ---------------------------------------------------------------------------

#: Inputs each strategy needs before it can estimate. When they are absent but
#: anchors are present, the parameters arrived pre-computed (a dataset-wide apply
#: of a calibration whose per-image samples do not travel) and are used as given.
_STRATEGY_INPUTS: dict[str, tuple[str, ...]] = {
    "gray_wedge": ("samples",),
    "two_patch": ("black_level", "white_level"),
}


def _normalize_response(params: dict) -> dict:
    """Validate a tone/colour response and resolve it to per-channel anchors.

    Tone and colour are one calibration, not two applied in sequence. They were
    split at first — a black/white/gamma stage and a per-channel gain stage — but
    both are estimates of the same thing: how this camera, under this light,
    turned reference values into pixel values. Keeping them separate meant they
    could be set inconsistently, or stacked on top of a card-based calibration
    that already accounts for both.

    So the stored form is the transform itself (anchors per channel), and how it
    was arrived at is a *strategy*. See ``strategies.py``.
    """
    strategy_key = str(params.get("strategy") or strategies.DEFAULT_STRATEGY)
    strategy = strategies.get_strategy(strategy_key)

    gamma = params.get("gamma", 1.0)
    try:
        gamma = float(gamma)
    except (TypeError, ValueError):
        raise InvalidCalibrationError("'gamma' must be a number.")
    if not np.isfinite(gamma) or not 0.05 <= gamma <= 10.0:
        raise InvalidCalibrationError(f"gamma must lie in 0.05-10 (got {gamma}).")

    required = _STRATEGY_INPUTS.get(strategy_key, ())
    if all(params.get(name) is not None for name in required):
        estimated = strategy.estimate(params)
    elif params.get("anchors"):
        estimated = {"anchors": strategies.validate_anchors(params["anchors"])}
        for key in ("card", "fit_model", "samples", "patch_names", "fit",
                    "black_level", "white_level", "gains"):
            if params.get(key) is not None:
                estimated[key] = params[key]
    else:
        missing = ", ".join(required)
        raise InvalidCalibrationError(
            f"The '{strategy.label}' strategy needs {missing} (or a pre-computed "
            f"'anchors' set)."
        )

    return {"strategy": strategy_key, "gamma": gamma, **estimated}


def _describe_response(params: dict) -> str:
    """One line summarising the correction, phrased per strategy."""
    gamma = float(params.get("gamma", 1.0))
    tail = "" if abs(gamma - 1.0) < 1e-9 else f", γ {gamma:.2f}"

    if params.get("strategy") == "gray_wedge":
        try:
            card_label = cards.get_card(params.get("card", "")).label
        except InvalidCalibrationError:
            card_label = params.get("card", "unknown card")
        count = len(params.get("samples") or ())
        fit = "line" if params.get("fit_model", "linear") == "linear" else "curve"
        return f"{count} patches, {card_label}, {fit} fit{tail}"

    gains = params.get("gains") or [1.0, 1.0, 1.0]
    balanced = any(abs(gain - 1.0) > 1e-6 for gain in gains)
    tone = (f"black {params.get('black_level', 0):.1f} → "
            f"white {params.get('white_level', 255):.1f}")
    if balanced:
        tone += " · gains " + " / ".join(f"{gain:.3f}" for gain in gains)
    return tone + tail


def _apply_response(rgb: np.ndarray, params: dict) -> np.ndarray:
    """Interpolate each channel through its anchors, then apply gamma.

    ``np.interp`` clamps to the end anchors rather than extrapolating, which is
    what keeps a specular highlight beyond the brightest reference from being
    driven to an arbitrary value. The original MATLAB returned NaN there, which
    became black.

    Gamma runs last and defaults to 1.0. It is a coarse nonlinearity knob for the
    two-patch strategy, which has only two references and so cannot see the shape
    of the response; a card-based calibration with a measured fit has no use for
    it, because the anchors already carry the real curve.
    """
    anchors = params["anchors"]
    out = np.empty_like(rgb, dtype=np.float32)
    for index, channel in enumerate("rgb"):
        points = anchors[channel]
        observed = [point[0] for point in points]
        target = [point[1] for point in points]
        out[..., index] = np.interp(rgb[..., index], observed, target)

    gamma = float(params.get("gamma", 1.0))
    if abs(gamma - 1.0) > 1e-9:
        np.clip(out, 0.0, 255.0, out=out)
        out = 255.0 * (out / 255.0) ** gamma
    return np.clip(out, 0.0, 255.0)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_KINDS: dict[str, CalibrationKind] = {}


def register(kind: CalibrationKind) -> CalibrationKind:
    """Add a kind to the registry. Called once per kind at import time."""
    if kind.key in _KINDS:
        raise ValueError(f"Calibration kind '{kind.key}' is already registered.")
    _KINDS[kind.key] = kind
    return kind


register(CalibrationKind(
    key="scale",
    label="Scale",
    summary="Physical size of one pixel. Turns pixel geometry into real-world "
            "lengths and areas.",
    order=0,
    permission=Permission.PIXEL_SCALE_SET,
    stale_metric_keys=SCALE_STALE_METRIC_KEYS,
    normalize=_normalize_scale,
    describe=_describe_scale,
    persist=_persist_scale,
    read=_read_scale,
    apply=None,
    fields=(
        ParamField("scale_x", "Scale X", "number", "Physical size of one pixel along x."),
        ParamField("scale_y", "Scale Y", "number", "Physical size of one pixel along y."),
        ParamField("unit", "Unit", "text", "Length unit, e.g. mm or µm."),
    ),
))

register(CalibrationKind(
    key="response",
    label="Color & intensity",
    summary="How this camera, under this light, turned reference values into pixel "
            "values. Corrects brightness and colour together so measurements are "
            "comparable across images.",
    order=10,
    permission=Permission.CALIBRATION_SET,
    stale_metric_keys=APPEARANCE_STALE_METRIC_KEYS,
    normalize=_normalize_response,
    describe=_describe_response,
    apply=_apply_response,
    strategy_keys=("gray_wedge", "two_patch"),
    fields=(
        ParamField("strategy", "Strategy", "enum",
                   "How the response is measured.",
                   choices=("gray_wedge", "two_patch")),
        ParamField("card", "Reference card", "text",
                   "Which card's target values to measure against.", optional=True),
        ParamField("fit_model", "Fit", "enum",
                   "Straight line through the patches, or the measured values directly.",
                   choices=("linear", "measured"), optional=True),
        ParamField("gamma", "Gamma", "number",
                   "Exponent applied after the correction. 1.0 leaves the tone curve "
                   "alone; only useful for the two-patch strategy.", optional=True),
    ),
))


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def get_kind(key: str) -> CalibrationKind:
    """Return the descriptor for ``key``, or raise UnknownCalibrationKindError."""
    try:
        return _KINDS[key]
    except KeyError:
        raise UnknownCalibrationKindError(
            f"Unknown calibration kind '{key}'. Known kinds: {', '.join(sorted(_KINDS))}."
        )


def all_kinds() -> list[CalibrationKind]:
    """Every registered kind, in pipeline order (which is also display order)."""
    return sorted(_KINDS.values(), key=lambda kind: (kind.order, kind.key))


def pixel_stages() -> list[CalibrationKind]:
    """The kinds that transform pixels, in the order they must be applied."""
    return [kind for kind in all_kinds() if kind.apply is not None]

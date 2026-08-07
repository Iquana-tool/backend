"""Calibration strategies — different ways to measure the *same* transform.

The insight this module exists to encode: a two-patch black/white/neutral
calibration and a twenty-patch gray-wedge calibration are not two different
corrections. They are two estimators of one thing — a per-channel transfer curve
from observed pixel value to reference value. Representing them as one kind with
selectable strategies (rather than as separate kinds) means they can never be
applied on top of each other, which is a class of bug that no amount of
`conflicts_with` bookkeeping would have reliably prevented.

The shared representation is a list of ``(observed, target)`` anchors per channel,
interpolated linearly and clamped at the ends. It covers both:

* **two_patch** — two anchors per channel, from a black and a white reference,
  with per-channel gains from an optional neutral patch folded into the upper one.
* **gray_wedge** — one anchor per card patch, from a least-squares line (or the
  measured values directly) through an N-step reference card.

Clamping at the ends replaces the original MATLAB behaviour of returning NaN for
out-of-range values, which silently mapped anything past the reference range to
black.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.exceptions import InvalidCalibrationError
from app.services.calibration import cards

#: Anchors closer together than this in observed value are treated as duplicates.
#: Two patches that read the same cannot both constrain the curve, and keeping
#: both makes the interpolation ill-conditioned.
_MIN_ANCHOR_GAP = 0.5


@dataclass(frozen=True)
class SampleRole:
    """One reference a strategy asks the user to click."""

    role: str
    label: str
    help: str

    def as_dict(self) -> dict:
        return {"role": self.role, "label": self.label, "help": self.help}


@dataclass(frozen=True)
class CalibrationStrategy:
    """How one kind's parameters are estimated from what the user measured."""

    key: str
    label: str
    summary: str
    estimate: Callable[[dict], dict]
    #: Named references the user clicks one at a time (two_patch). Empty for a
    #: strategy that instead samples once per patch of a reference card.
    sample_roles: tuple[SampleRole, ...] = ()
    #: Whether the strategy needs a `card` profile, and therefore samples one
    #: patch per card entry rather than a fixed set of named roles.
    requires_card: bool = False
    #: Fit models the strategy accepts, first being the default.
    fit_models: tuple[dict, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "strategy": self.key,
            "label": self.label,
            "summary": self.summary,
            "requires_card": self.requires_card,
            "sample_roles": [role.as_dict() for role in self.sample_roles],
            "fit_models": [dict(model) for model in self.fit_models],
        }


# ---------------------------------------------------------------------------
# Anchor helpers, shared by every strategy
# ---------------------------------------------------------------------------

def _as_triplet(value, name: str) -> list[float]:
    try:
        triplet = [float(v) for v in value]
    except (TypeError, ValueError):
        raise InvalidCalibrationError(f"'{name}' must be three numbers.")
    if len(triplet) != 3:
        raise InvalidCalibrationError(
            f"'{name}' must have exactly three components (got {len(triplet)})."
        )
    if not all(np.isfinite(v) for v in triplet):
        raise InvalidCalibrationError(f"'{name}' must be finite.")
    return triplet


def build_anchors(observed: list[float], target: list[float], channel: str) -> list[list[float]]:
    """Turn paired measurements into a validated, interpolatable anchor list.

    Sorts by observed value, drops references that read too close together to
    distinguish, and adds the ``(0, 0)`` / ``(255, 255)`` end anchors so values
    beyond the reference range fall back to identity rather than being clipped
    flat or extrapolated wildly.

    Raises:
        InvalidCalibrationError: If fewer than two usable anchors survive, or the
            targets do not increase with the observed values — which means the
            references were clicked out of order, or on the wrong card.
    """
    pairs = sorted(zip(observed, target), key=lambda pair: pair[0])

    kept: list[tuple[float, float]] = []
    for obs, tgt in pairs:
        if kept and obs - kept[-1][0] < _MIN_ANCHOR_GAP:
            continue
        kept.append((float(obs), float(tgt)))

    if len(kept) < 2:
        raise InvalidCalibrationError(
            f"Channel {channel} has fewer than two distinguishable references "
            f"({len(kept)}). Sample patches that actually differ in brightness."
        )

    targets = [tgt for _, tgt in kept]
    if any(b < a for a, b in zip(targets, targets[1:])):
        raise InvalidCalibrationError(
            f"Channel {channel}: brighter references map to darker targets. The "
            f"patches were most likely sampled in the wrong order, or against the "
            f"wrong reference card."
        )

    # End anchors, only where they do not collide with a real measurement.
    if kept[0][0] > _MIN_ANCHOR_GAP:
        kept.insert(0, (0.0, 0.0))
    if kept[-1][0] < 255.0 - _MIN_ANCHOR_GAP:
        kept.append((255.0, 255.0))

    return [[obs, tgt] for obs, tgt in kept]


def validate_anchors(anchors: dict) -> dict:
    """Check an anchor set that arrived pre-computed (a dataset-wide apply)."""
    if not isinstance(anchors, dict):
        raise InvalidCalibrationError("'anchors' must be an object keyed by channel.")
    validated: dict[str, list[list[float]]] = {}
    for channel in ("r", "g", "b"):
        raw = anchors.get(channel)
        if not raw:
            raise InvalidCalibrationError(f"'anchors' is missing channel '{channel}'.")
        points: list[list[float]] = []
        for pair in raw:
            try:
                obs, tgt = (float(pair[0]), float(pair[1]))
            except (TypeError, ValueError, IndexError):
                raise InvalidCalibrationError(
                    f"Channel {channel}: each anchor must be an [observed, target] pair."
                )
            if not (np.isfinite(obs) and np.isfinite(tgt)):
                raise InvalidCalibrationError(f"Channel {channel}: anchors must be finite.")
            points.append([obs, tgt])
        if len(points) < 2:
            raise InvalidCalibrationError(f"Channel {channel} needs at least two anchors.")
        if any(b[0] <= a[0] for a, b in zip(points, points[1:])):
            raise InvalidCalibrationError(
                f"Channel {channel}: anchors must be strictly increasing in observed value."
            )
        validated[channel] = points
    return validated


# ---------------------------------------------------------------------------
# two_patch
# ---------------------------------------------------------------------------

def _estimate_two_patch(params: dict) -> dict:
    """Black point, white point, and per-channel gains from a neutral patch.

    The quick path, for images with no reference card in frame. Two anchors per
    channel: the black reference maps to 0, the white reference to 255 scaled by
    that channel's gain.

    The neutral patch is sampled from the raw image and its gains are derived
    *after* the black/white normalisation is applied to it, so the correction is
    consistent even though the user only clicked once. Deriving gains from the raw
    values instead would leave them wrong by whatever the tone normalisation does.

    Without an explicit ``target_rgb`` the neutral patch is assumed neutral and the
    gains normalise the channels onto their common mean — removing the cast while
    leaving overall brightness to the black/white points.
    """
    black = _required_float(params, "black_level")
    white = _required_float(params, "white_level")

    if not 0.0 <= black <= 255.0 or not 0.0 <= white <= 255.0:
        raise InvalidCalibrationError(
            f"black_level and white_level must lie in 0-255 (got {black}, {white})."
        )
    if white - black < 1.0:
        raise InvalidCalibrationError(
            f"white_level must exceed black_level by at least 1 level (got black="
            f"{black}, white={white}). Sample two patches that differ in brightness."
        )

    gains = _resolve_two_patch_gains(params, black, white)

    anchors = {}
    for index, channel in enumerate("rgb"):
        anchors[channel] = build_anchors(
            observed=[black, white],
            target=[0.0, 255.0 * gains[index]],
            channel=channel,
        )

    estimated = {
        "black_level": black,
        "white_level": white,
        "gains": gains,
        "anchors": anchors,
    }
    for key in ("black_observed_rgb", "white_observed_rgb", "neutral_rgb", "target_rgb"):
        if params.get(key) is not None:
            estimated[key] = _as_triplet(params[key], key)
    return estimated


def _resolve_two_patch_gains(params: dict, black: float, white: float) -> list[float]:
    """Explicit gains if given, otherwise derived from the neutral patch."""
    if params.get("gains") is not None:
        gains = _as_triplet(params["gains"], "gains")
    elif params.get("neutral_rgb") is not None:
        neutral = _as_triplet(params["neutral_rgb"], "neutral_rgb")
        # Normalise the patch the way the black/white points will, then balance.
        normalised = [
            255.0 * min(max((value - black) / (white - black), 0.0), 1.0)
            for value in neutral
        ]
        if any(value <= 1e-6 for value in normalised):
            raise InvalidCalibrationError(
                "The neutral patch has a channel at (or below) the black point, so "
                "no gain can correct it. Pick a patch that is neutral and not "
                "under-exposed."
            )
        target = (_as_triplet(params["target_rgb"], "target_rgb")
                  if params.get("target_rgb") is not None
                  else [float(np.mean(normalised))] * 3)
        gains = [target[c] / normalised[c] for c in range(3)]
    else:
        # A tone-only calibration is legitimate: no neutral reference in frame.
        gains = [1.0, 1.0, 1.0]

    if any(gain <= 0 for gain in gains):
        raise InvalidCalibrationError(f"Channel gains must be positive (got {gains}).")
    if any(gain > 20.0 for gain in gains):
        raise InvalidCalibrationError(
            f"Channel gains above 20x are almost certainly a mis-sampled patch (got {gains})."
        )
    return gains


# ---------------------------------------------------------------------------
# gray_wedge
# ---------------------------------------------------------------------------

def _estimate_gray_wedge(params: dict) -> dict:
    """Per-channel response fitted through every neutral patch of a reference card.

    This is the users' MATLAB method, generalised. Samples arrive one per card
    patch, in card order; the card supplies the target each one should have read.

    Two fit models:

    ``linear``
        A least-squares straight line through the samples per channel, which is
        what the original does. Robust — one mis-clicked patch barely moves it —
        at the cost of assuming an affine camera response.
    ``measured``
        The sampled values used directly as anchors. Uses what twenty references
        actually buy you (the real shape of the response curve), but a bad sample
        becomes a kink in the transfer function rather than being averaged away.

    Note what is *not* done here: the targets are used as the card states them,
    never regressed. The original regressed its reference too, which was harmless
    only because its assumed reference was already a straight line. Fitting a line
    through a physically correct (curved) target would throw away exactly the
    accuracy that target exists to provide.
    """
    card = cards.get_card(params.get("card") or cards.DEFAULT_CARD)
    fit_model = str(params.get("fit_model") or "linear")
    if fit_model not in ("linear", "measured"):
        raise InvalidCalibrationError(
            f"Unknown fit model '{fit_model}' (expected 'linear' or 'measured')."
        )

    patches = card.neutral_patches
    raw_samples = params.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise InvalidCalibrationError(
            f"The {card.label} needs one sample per patch; none were provided."
        )
    if len(raw_samples) != len(patches):
        raise InvalidCalibrationError(
            f"The {card.label} has {len(patches)} usable patches but "
            f"{len(raw_samples)} samples were provided. Every patch must be sampled."
        )

    observed = np.array(
        [_as_triplet(sample, f"samples[{i}]") for i, sample in enumerate(raw_samples)],
        dtype=np.float64,
    )
    targets = np.array([patch.target_rgb for patch in patches], dtype=np.float64)

    anchors: dict[str, list[list[float]]] = {}
    fit: dict[str, dict] = {}
    for index, channel in enumerate("rgb"):
        column = observed[:, index]
        if fit_model == "linear":
            fitted, slope, intercept = _fit_line(column, channel)
            fit[channel] = {"slope": slope, "intercept": intercept}
        else:
            fitted = _monotonise(column, targets[:, index])
        anchors[channel] = build_anchors(
            observed=list(fitted), target=list(targets[:, index]), channel=channel,
        )

    return {
        "card": card.key,
        "fit_model": fit_model,
        "samples": [[float(v) for v in sample] for sample in observed],
        "patch_names": [patch.name for patch in patches],
        "fit": fit,
        "anchors": anchors,
    }


def _fit_line(values: np.ndarray, channel: str) -> tuple[np.ndarray, float, float]:
    """Least-squares line through the samples, evaluated at each patch index.

    Returns the fitted values plus the coefficients, which are kept as provenance
    so the response curve can be drawn against the raw samples.
    """
    indices = np.arange(1, len(values) + 1, dtype=np.float64)
    design = np.stack([np.ones_like(indices), indices], axis=-1)
    (intercept, slope), *_ = np.linalg.lstsq(design, values, rcond=None)
    if abs(slope) < 1e-9:
        raise InvalidCalibrationError(
            f"Channel {channel}: the sampled patches show no brightness gradient. "
            f"They were most likely all taken from the same patch."
        )
    return intercept + slope * indices, float(slope), float(intercept)


def _monotonise(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Force the samples to increase with their targets, keeping them usable as anchors.

    A patch that reads out of order (a glint, a shadow across the card) would
    otherwise make the transfer function non-invertible. Clamping each value to at
    least its predecessor's is the least destructive fix that keeps the rest of the
    curve intact.

    Wholesale disagreement is a different problem and gets a different answer:
    if the samples trend *against* their targets, the card was read from the wrong
    end, and flattening that into a constant would report it as "these patches are
    all the same brightness" — true of the repaired data, and useless as a
    diagnosis of what the user actually did.
    """
    if np.std(values) > 1e-9 and np.corrcoef(values, targets)[0, 1] < 0:
        raise InvalidCalibrationError(
            "The sampled patches get darker where the reference card gets brighter. "
            "They were most likely sampled in the wrong order — start from the "
            "lightest patch."
        )

    order = np.argsort(targets)
    adjusted = np.maximum.accumulate(values[order])
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    return restored


def _required_float(params: dict, name: str) -> float:
    if params.get(name) is None:
        raise InvalidCalibrationError(f"Missing required parameter '{name}'.")
    try:
        value = float(params[name])
    except (TypeError, ValueError):
        raise InvalidCalibrationError(f"Parameter '{name}' must be a number.")
    if not np.isfinite(value):
        raise InvalidCalibrationError(f"Parameter '{name}' must be finite.")
    return value


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_STRATEGIES: dict[str, CalibrationStrategy] = {}


def register_strategy(strategy: CalibrationStrategy) -> CalibrationStrategy:
    if strategy.key in _STRATEGIES:
        raise ValueError(f"Calibration strategy '{strategy.key}' is already registered.")
    _STRATEGIES[strategy.key] = strategy
    return strategy


register_strategy(CalibrationStrategy(
    key="gray_wedge",
    label="Reference card",
    summary="Sample every patch of a gray scale in the frame and fit the camera's "
            "response to it. More references, and it corrects each channel "
            "independently.",
    estimate=_estimate_gray_wedge,
    requires_card=True,
    fit_models=(
        {"key": "linear", "label": "Straight line",
         "help": "Least-squares line through the patches. Robust to a mis-clicked "
                 "patch; assumes the response is affine."},
        {"key": "measured", "label": "Measured curve",
         "help": "Use the sampled values directly. Captures the real response "
                 "shape, but a bad sample becomes a kink."},
    ),
))

register_strategy(CalibrationStrategy(
    key="two_patch",
    label="Black / white / neutral",
    summary="Sample a dark and a bright reference, and optionally a neutral one. "
            "For images with no reference card in frame.",
    estimate=_estimate_two_patch,
    sample_roles=(
        SampleRole("black", "Black patch",
                   "Click the darkest reference in the frame."),
        SampleRole("white", "White patch",
                   "Click the brightest reference that is not blown out."),
        SampleRole("neutral", "Neutral patch (optional)",
                   "Click a gray card to correct the colour cast. Without one, only "
                   "brightness is corrected."),
    ),
))

#: The strategy new calibrations start from. The card-based one: the users'
#: images carry a gray scale by design, and it is the better estimator whenever
#: one is present.
DEFAULT_STRATEGY = "gray_wedge"


def get_strategy(key: str) -> CalibrationStrategy:
    try:
        return _STRATEGIES[key]
    except KeyError:
        raise InvalidCalibrationError(
            f"Unknown calibration strategy '{key}'. Known strategies: "
            f"{', '.join(sorted(_STRATEGIES))}."
        )


def all_strategies() -> list[CalibrationStrategy]:
    """Registered strategies, card-based first (the recommended path)."""
    return sorted(_STRATEGIES.values(), key=lambda s: (not s.requires_card, s.key))

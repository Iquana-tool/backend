"""Reference card profiles — what a physical calibration target *reads as*.

A calibration strategy that works from a reference card needs to know two things:
how many patches the card has, and what each one is supposed to measure. Both are
properties of the card, not of the code, so they live here as data. Supporting a
different card — a ColorChecker, a step wedge with a different patch count, a
densitometry strip — is a profile added to :data:`_CARDS`, with no change to the
estimator, the service or the UI.

Why two Kodak profiles
----------------------
The Kodak Q-13/Q-14 gray scale is linear in optical *density*, not in pixel value:
its steps advance in equal density increments, and density maps to reflectance as
``10**-D``, which an sRGB-encoded camera then compresses further. The original
MATLAB utility assumed a straight line in pixel value instead (``269 - 13i``,
i.e. 256 down to 9), which is off by roughly 48 levels in the mid-tone — its
middle-gray patch is assumed to read 165 when the physical card should read ~117.

That does not make the MATLAB results useless: every image is mapped onto the same
(non-physical) target, so images stay comparable *to each other*. It does mean the
per-channel straight-line fit is being asked to model a curve, so its residuals are
structure rather than noise.

Both targets therefore ship. ``kodak_q13_legacy`` reproduces the MATLAB numbers and
is the default, so existing results stay reproducible; ``kodak_q13`` is the
physically correct target. Switching between them is a dropdown, and the response
curve makes the difference visible: under the legacy profile the measured points
bow away from the fitted line in a way they should not.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.exceptions import InvalidCalibrationError

#: Patch roles. Only ``neutral`` patches drive the gray-wedge estimator today;
#: ``chromatic`` exists so a card's colour patches (the C/Y/M spots on the Kodak
#: strip, a ColorChecker's 18) can be declared now and consumed by a future
#: matrix-fitting strategy without redefining the card.
PATCH_NEUTRAL = "neutral"
PATCH_CHROMATIC = "chromatic"


@dataclass(frozen=True)
class ReferencePatch:
    """One patch on a reference card, and what it should measure."""

    name: str
    #: Expected 8-bit RGB. ``None`` for a patch whose target is not established,
    #: which the estimators skip rather than guess at.
    target_rgb: tuple[float, float, float] | None = None
    role: str = PATCH_NEUTRAL
    #: Optical density, for neutral patches derived from a published density scale.
    #: Kept as provenance: it is what ``target_rgb`` was computed from.
    density: float | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "target_rgb": list(self.target_rgb) if self.target_rgb else None,
            "role": self.role,
            "density": self.density,
        }


@dataclass(frozen=True)
class ReferenceCard:
    """A physical calibration target, described well enough to measure against."""

    key: str
    label: str
    summary: str
    patches: tuple[ReferencePatch, ...]
    #: Longest physical edge and its unit, when known. Not used yet; it is what
    #: would let the same card that calibrates colour also calibrate scale.
    physical_length: float | None = None
    physical_unit: str | None = None
    #: Where the target values came from, shown in the UI so a user can judge them.
    provenance: str = ""

    @property
    def neutral_patches(self) -> tuple[ReferencePatch, ...]:
        """The patches a tone/colour response can be fitted through."""
        return tuple(
            patch for patch in self.patches
            if patch.role == PATCH_NEUTRAL and patch.target_rgb is not None
        )

    def as_dict(self) -> dict:
        return {
            "card": self.key,
            "label": self.label,
            "summary": self.summary,
            "provenance": self.provenance,
            "physical_length": self.physical_length,
            "physical_unit": self.physical_unit,
            "patch_count": len(self.patches),
            "neutral_patch_count": len(self.neutral_patches),
            "patches": [patch.as_dict() for patch in self.patches],
        }


# ---------------------------------------------------------------------------
# Building the Kodak profiles
# ---------------------------------------------------------------------------

#: Step names as printed on the Kodak Q-13 / Q-14 gray scale, in order from the
#: lightest patch. A, M and B are the printed markers for the highlight, middle
#: and shadow references; the rest are numbered, with 7 and 16 taken by M and B.
_KODAK_STEP_NAMES: tuple[str, ...] = (
    "A", "1", "2", "3", "4", "5", "6", "M", "8", "9",
    "10", "11", "12", "13", "14", "15", "B", "17", "18", "19",
)

#: Published density of the lightest step, and the increment between steps.
#: Twenty steps at 0.10 spans 0.05 to 1.95. Worth confirming against a physical
#: card: if the edition in use differs, this is the only thing to change.
_KODAK_BASE_DENSITY = 0.05
_KODAK_DENSITY_STEP = 0.10


def _srgb_encode(linear: float) -> float:
    """Linear reflectance (0-1) to sRGB-encoded value (0-1)."""
    if linear <= 0.0031308:
        return 12.92 * linear
    return 1.055 * (linear ** (1 / 2.4)) - 0.055


def _density_to_srgb8(density: float) -> float:
    """Optical density to the 8-bit value a correctly exposed sRGB camera records."""
    reflectance = 10.0 ** -density
    return round(255.0 * _srgb_encode(reflectance), 1)


def _kodak_physical_patches() -> tuple[ReferencePatch, ...]:
    """The Kodak steps with targets derived from their published densities."""
    patches = []
    for index, name in enumerate(_KODAK_STEP_NAMES):
        density = _KODAK_BASE_DENSITY + _KODAK_DENSITY_STEP * index
        value = _density_to_srgb8(density)
        patches.append(ReferencePatch(
            name=name,
            target_rgb=(value, value, value),
            role=PATCH_NEUTRAL,
            density=round(density, 2),
        ))
    return tuple(patches)


def _kodak_legacy_patches() -> tuple[ReferencePatch, ...]:
    """The Kodak steps with the MATLAB utility's assumed straight-line targets.

    ``Res1_r = [256:-13:0]`` in the original, i.e. ``256 - 13i`` for i = 0..19.
    The leading 256 is out of range for 8-bit output and saturates to 255 on the
    way out of MATLAB, so it is clamped here rather than carried as a value no
    pixel can take.
    """
    patches = []
    for index, name in enumerate(_KODAK_STEP_NAMES):
        value = float(min(255, 256 - 13 * index))
        patches.append(ReferencePatch(
            name=name,
            target_rgb=(value, value, value),
            role=PATCH_NEUTRAL,
        ))
    return tuple(patches)


#: The colour spots printed above the gray steps. Declared so the card is an
#: honest description of the physical object, with no targets: Kodak's nominal
#: values for them are not established here, and a guessed target would be worse
#: than none. The gray-wedge estimator skips them.
_KODAK_COLOR_PATCHES: tuple[ReferencePatch, ...] = (
    ReferencePatch(name="C", role=PATCH_CHROMATIC),
    ReferencePatch(name="Y", role=PATCH_CHROMATIC),
    ReferencePatch(name="M", role=PATCH_CHROMATIC),
)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_CARDS: dict[str, ReferenceCard] = {}


def register_card(card: ReferenceCard) -> ReferenceCard:
    """Add a card profile. Called once per card at import time."""
    if card.key in _CARDS:
        raise ValueError(f"Reference card '{card.key}' is already registered.")
    if not card.neutral_patches:
        raise ValueError(f"Reference card '{card.key}' has no usable neutral patches.")
    _CARDS[card.key] = card
    return card


register_card(ReferenceCard(
    key="kodak_q13_legacy",
    label="Kodak Gray Scale (legacy targets)",
    summary="Twenty gray steps, targeted at the straight ramp the original MATLAB "
            "utility assumed. Reproduces existing results.",
    patches=_kodak_legacy_patches() + _KODAK_COLOR_PATCHES,
    provenance="Res1 = 256 - 13i from CalibrateImage.m. Not physically accurate, but "
               "every image maps onto the same target, so images stay comparable.",
))

register_card(ReferenceCard(
    key="kodak_q13",
    label="Kodak Gray Scale (measured densities)",
    summary="Twenty gray steps, targeted at what the published densities should "
            "actually record through an sRGB camera.",
    patches=_kodak_physical_patches() + _KODAK_COLOR_PATCHES,
    provenance=f"Densities {_KODAK_BASE_DENSITY} to "
               f"{_KODAK_BASE_DENSITY + _KODAK_DENSITY_STEP * 19:.2f} in "
               f"{_KODAK_DENSITY_STEP} steps, converted by 10**-D then sRGB-encoded.",
))


def get_card(key: str) -> ReferenceCard:
    """Return a card profile, or raise if it is not registered."""
    try:
        return _CARDS[key]
    except KeyError:
        raise InvalidCalibrationError(
            f"Unknown reference card '{key}'. Known cards: {', '.join(sorted(_CARDS))}."
        )


def all_cards() -> list[ReferenceCard]:
    """Every registered card profile, by key."""
    return [_CARDS[key] for key in sorted(_CARDS)]


#: The card new calibrations start from. The legacy profile, deliberately: it is
#: what the users' existing numbers were produced with, so adopting this system
#: does not silently move their results.
DEFAULT_CARD = "kodak_q13_legacy"

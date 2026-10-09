"""What a stack reader returns: arrays and metadata, independent of any format."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameData:
    #: 2D uint8 (grayscale) or HxWx3 uint8 (RGB) pixels, ready to save as PNG.
    pixels: np.ndarray
    #: Position along the stack axis, in the stack's ``frame_spacing_unit``.
    position: float | None = None
    #: Where the frame lies on the overview, in overview pixels, e.g.
    #: ``{"type": "line", "start": [x, y], "end": [x, y]}``.
    overview_geometry: dict | None = None
    #: Metadata that belongs to this frame only (e.g. an OCT B-scan's quality).
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class StackData:
    name: str
    #: ``oct_volume`` | ``video`` | ``z_stack``.
    kind: str
    #: ``z`` or ``t``.
    axis: str
    #: Format tag, e.g. ``heidelberg_vol``.
    source_format: str
    frames: list[FrameData]

    #: Pixel size of every frame. ``unit`` is ``px`` when the file gives no
    #: trustworthy physical scale.
    scale_x: float = 1.0
    scale_y: float = 1.0
    unit: str = "px"

    frame_spacing: float | None = None
    frame_spacing_unit: str | None = None

    overview: np.ndarray | None = None
    overview_scale_x: float | None = None
    overview_scale_y: float | None = None
    overview_unit: str | None = None

    #: Metadata true of the whole stack; every frame inherits it.
    metadata: dict[str, str] = field(default_factory=dict)
    #: ``{key: value_type}`` for metadata keys that should not default to
    #: categorical when they are first created (dates, numbers).
    key_types: dict[str, str] = field(default_factory=dict)

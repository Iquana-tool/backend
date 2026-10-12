"""Heidelberg Engineering OCT exports (``.vol``, ``.e2e``), read with eyepy.

A ``.vol`` file holds one volume. An ``.e2e`` export can hold several series --
both eyes, follow-up visits -- and each OCT volume among them becomes its own stack.
Single circle scans and radial (star) scans are skipped: eyepy does not read them
as volumes, and they are not what the volume workflow is for.

What is kept: the B-scans and the IR-SLO overview as displayed by Heidelberg's own
software (eyepy applies the vendor's intensity transform), pixel sizes, where each
B-scan lies on the overview, and a short whitelist of metadata. Patient fields in
the file header are never copied.

Physical scale: ``.vol`` headers carry exact pixel sizes in mm. eyepy does not yet
know where an ``.e2e`` stores them (its volume metadata reports ``px``), so E2E
stacks are stored in pixels until that is resolved, rather than with a guessed
scale that measurements would silently inherit.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.services.stack_readers.base import FrameData, StackData

logger = logging.getLogger(__name__)

#: Heidelberg writes laterality as OD/OS or R/L depending on the file version.
_LATERALITY = {"OD": "OD", "R": "OD", "OS": "OS", "L": "OS"}

#: Metadata keys whose type is not the default categorical.
_KEY_TYPES = {"Visit date": "date", "B-scan quality": "number"}


def read_vol(path: Path, original_name: str) -> list[StackData]:
    from eyepy.io.he.vol_reader import HeVolReader

    volume = HeVolReader(path).volume
    return [_stack_from_volume(volume, name=Path(original_name).stem, source_format="heidelberg_vol")]


def read_e2e(path: Path, original_name: str) -> list[StackData]:
    from eyepy.io.he.e2e_reader import HeE2eReader

    stem = Path(original_name).stem
    stacks: list[StackData] = []
    with HeE2eReader(path) as reader:
        series_list = [series for series in reader.series if series.n_bscans > 1]
        for series in series_list:
            try:
                volume = series.get_volume()
            except ValueError as exc:
                # Circle and radial scans, or a series without an overview image.
                logger.info("Skipping E2E series %s of '%s': %s", series.id, original_name, exc)
                continue
            laterality = _LATERALITY.get(str(series.laterality()).strip().upper())
            name = stem if len(series_list) == 1 else " ".join(
                part for part in (stem, laterality, f"#{series.id}") if part)
            stack = _stack_from_volume(volume, name=name, source_format="heidelberg_e2e")
            container = (volume.meta.get("e2e_metadata") or {}).get("heyex", {}).get("Container", {})
            series_date = container.get("Series Date")
            if series_date and "Visit date" not in stack.metadata:
                stack.metadata["Visit date"] = series_date
            if container.get("Scan Pattern"):
                stack.metadata["Scan pattern"] = str(container["Scan Pattern"])
            stacks.append(stack)
    return stacks


def _stack_from_volume(volume, name: str, source_format: str) -> StackData:
    meta = volume.meta
    bscan_meta = meta["bscan_meta"]
    data = _to_uint8(volume.data)

    physical = meta.get("scale_unit") == "mm"
    spacing = float(meta["scale_z"]) if physical else None

    localizer = volume.localizer
    localizer_meta = localizer.meta if localizer is not None else {}

    frames = []
    for index in range(data.shape[0]):
        frame_meta = bscan_meta[index] if index < len(bscan_meta) else {}
        quality = frame_meta.get("quality")
        frames.append(FrameData(
            pixels=data[index],
            position=index * spacing if spacing is not None else None,
            overview_geometry=_overview_line(frame_meta, localizer_meta),
            metadata={"B-scan quality": f"{float(quality):.2f}"} if quality is not None else {},
        ))

    metadata: dict[str, str] = {}
    laterality = _LATERALITY.get(str(meta.get("laterality") or "").strip().upper())
    if laterality:
        metadata["Eye"] = laterality
    if meta.get("visit_date"):
        metadata["Visit date"] = str(meta["visit_date"])
    if localizer_meta.get("modality"):
        metadata["Overview modality"] = str(localizer_meta["modality"])

    return StackData(
        name=name,
        kind="oct_volume",
        axis="z",
        source_format=source_format,
        frames=frames,
        scale_x=float(meta["scale_x"]) if physical else 1.0,
        scale_y=float(meta["scale_y"]) if physical else 1.0,
        unit="mm" if physical else "px",
        frame_spacing=spacing,
        frame_spacing_unit="mm" if spacing is not None else None,
        overview=_to_uint8(localizer.data) if localizer is not None else None,
        overview_scale_x=_float_or_none(localizer_meta.get("scale_x")),
        overview_scale_y=_float_or_none(localizer_meta.get("scale_y")),
        overview_unit=localizer_meta.get("scale_unit"),
        metadata=metadata,
        key_types=dict(_KEY_TYPES),
    )


def _overview_line(frame_meta, localizer_meta) -> dict | None:
    """The B-scan's line on the overview image, in overview pixels.

    B-scan start/end positions are in mm (``.vol``) or in degrees from the centre
    of the field of view (``.e2e``); the overview's scale is in the same unit.
    """
    start, end = frame_meta.get("start_pos"), frame_meta.get("end_pos")
    scale_x, scale_y = localizer_meta.get("scale_x"), localizer_meta.get("scale_y")
    pos_unit, overview_unit = frame_meta.get("pos_unit"), localizer_meta.get("scale_unit")
    if start is None or end is None or not scale_x or not scale_y or pos_unit != overview_unit:
        return None
    offset = float(localizer_meta.get("field_size") or 30) / 2 if pos_unit == "°" else 0.0
    scale = np.array([scale_x, scale_y], dtype=float)

    def to_pixels(point) -> list[float]:
        return [round(float(v), 2) for v in (np.asarray(point, dtype=float) + offset) / scale]

    return {"type": "line", "start": to_pixels(start), "end": to_pixels(end)}


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Display intensities as uint8; eyepy returns uint8 or floats in [0, 1]."""
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    return (np.clip(np.nan_to_num(array.astype(np.float32)), 0.0, 1.0) * 255).round().astype(np.uint8)


def _float_or_none(value) -> float | None:
    return float(value) if value is not None else None

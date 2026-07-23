"""Regression tests for the COCO export's use of native image dimensions.

Background: contours are stored with normalized [0, 1] coordinates, and the COCO
export materializes pixel-space geometry by scaling them by the image size. A bug at
ingest stored *thumbnail* dimensions (<=200px) in ``Images.width/height`` because
``PIL.Image.thumbnail`` mutates in place, so exports came out ~20x too small.

These tests pin two guarantees:
  1. ``save_image_to_disk`` reports the native (full-resolution) size, not the
     thumbnail size.
  2. ``build_coco_payload`` always emits native ``images[].width/height`` (read from
     the real file on disk, even when the DB columns are stale) and every annotation's
     bbox fits within those native bounds, with area recomputed in native space.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image as PILImage

from app.services.database_access.datasets import build_coco_payload
from app.services.database_access.images import save_image_to_disk, native_image_size


# A couple of native sizes including a non-integer downscale factor (3024/200 = 15.12)
# so the test also covers the "not exactly 20x, and different per axis" case.
NATIVE_SIZES = {
    "landscape.png": (4000, 3000),  # 20.00x on both axes -> thumbnail 200x150
    "portrait.png": (3024, 4032),   # 15.12x -> thumbnail 150x200 (per-axis scale)
}


def _write_image(path, size):
    """Write a real image file of the given (width, height) to disk."""
    width, height = size
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    PILImage.fromarray(arr).save(path)
    return path


def _make_image_row(image_id, file_path, native_size, stale=True):
    """Build a duck-typed stand-in for the ``Images`` ORM row.

    When ``stale`` is True the stored width/height are the *thumbnail* dimensions,
    reproducing the historical bug — the export must ignore them in favour of the
    real file on disk.
    """
    if stale:
        thumb = PILImage.new("RGB", native_size)
        thumb.thumbnail((200, 200))
        stored_w, stored_h = thumb.width, thumb.height
    else:
        stored_w, stored_h = native_size
    return SimpleNamespace(
        id=image_id,
        file_name=file_path.name,
        file_path=str(file_path),
        width=stored_w,
        height=stored_h,
    )


def _make_contour(contour_id, label_id, xs, ys, area):
    return SimpleNamespace(id=contour_id, label_id=label_id, x=xs, y=ys, area=area)


def test_save_image_to_disk_returns_native_dimensions(tmp_path):
    """The ingest helper must report native size, not the mutated thumbnail size."""
    native_w, native_h = 4000, 3000
    file_path = tmp_path / "stored.png"
    thumb_path = tmp_path / "thumb.png"

    # numpy arrays are (height, width, channels); the helper also accepts np.ndarray.
    arr = np.zeros((native_h, native_w, 3), dtype=np.uint8)

    width, height, mode = asyncio.run(save_image_to_disk(arr, file_path, thumb_path))

    assert (width, height) == (native_w, native_h), "native dimensions leaked the thumbnail size"
    assert mode == "RGB"
    # The full-res file keeps native dims; the thumbnail is capped at 200px.
    assert native_image_size(file_path) == (native_w, native_h)
    tw, th = native_image_size(thumb_path)
    assert max(tw, th) == 200


def test_coco_export_uses_native_dimensions_and_bboxes_fit(tmp_path):
    """images[].width/height equal the real file dims and every bbox fits inside."""
    # Materialize real image files at native resolution.
    paths = {name: _write_image(tmp_path / name, size) for name, size in NATIVE_SIZES.items()}

    images = {
        1: _make_image_row(1, paths["landscape.png"], NATIVE_SIZES["landscape.png"], stale=True),
        2: _make_image_row(2, paths["portrait.png"], NATIVE_SIZES["portrait.png"], stale=True),
    }
    label = SimpleNamespace(id=7, name="coral")

    # A square covering the middle half of each image, in normalized coordinates.
    xs = [0.25, 0.75, 0.75, 0.25]
    ys = [0.25, 0.25, 0.75, 0.75]
    normalized_area = 0.5 * 0.5  # fraction of the image area

    rows = [
        (_make_contour(101, 7, xs, ys, normalized_area), images[1], label),
        (_make_contour(102, 7, xs, ys, normalized_area), images[2], label),
    ]
    dataset = SimpleNamespace(name="Corals", description=None)

    payload, image_ids = build_coco_payload(dataset, rows)

    assert image_ids == {1, 2}
    images_by_id = {img["id"]: img for img in payload["images"]}

    # (a) images[].width/height must equal the real file's native dimensions.
    for image_id, name in [(1, "landscape.png"), (2, "portrait.png")]:
        real_w, real_h = native_image_size(paths[name])
        assert (images_by_id[image_id]["width"], images_by_id[image_id]["height"]) == (real_w, real_h)

    # (b) every annotation's bbox must fit within its image's declared bounds, and
    #     area must be recomputed in native space (fraction * W * H).
    for ann in payload["annotations"]:
        img = images_by_id[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        assert x >= 0 and y >= 0
        assert x + w <= img["width"] + 1e-6
        assert y + h <= img["height"] + 1e-6
        # segmentation points are within bounds too
        seg = ann["segmentation"][0]
        assert max(seg[0::2]) <= img["width"] + 1e-6
        assert max(seg[1::2]) <= img["height"] + 1e-6
        # native area == normalized fraction * native width * native height
        expected_area = normalized_area * img["width"] * img["height"]
        assert ann["area"] == pytest.approx(expected_area)
        # sanity: area is in full-res pixel^2, far larger than any thumbnail area
        assert ann["area"] > 200 * 200

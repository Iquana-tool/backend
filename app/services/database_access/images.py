import base64
import io
import os
from logging import getLogger
from pathlib import Path
from typing import Union

import numpy as np
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from iquana_toolbox.schemas.database.image import Image as ImageModel
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from app.database.images import Images
from app.database.masks import Masks
from app.services.database_access.masks import create_new_mask
from config import THUMBNAILS_DIR

logger = getLogger(__name__)


class TIFFPreviewError(HTTPException):
    """A TIFF cannot be represented as a safe, single-image browser preview."""

    def __init__(self, detail: str):
        super().__init__(status_code=422, detail=detail)


def _tiff_png_bytes(image: Image.Image) -> bytes:
    """Encode one TIFF page as PNG without clipping high-bit-depth grayscale data."""
    if getattr(image, "n_frames", 1) != 1:
        raise TIFFPreviewError("Multi-page TIFF images are not supported; upload one page per image.")

    image.load()
    mode = image.mode
    raw_bits = getattr(image, "tag_v2", {}).get(258, (8,))
    bits_per_sample = (raw_bits,) if isinstance(raw_bits, int) else tuple(raw_bits)
    if max(bits_per_sample, default=8) > 8 and mode not in {"I;16", "I;16L", "I;16B", "I", "F"}:
        raise TIFFPreviewError("Multi-channel high-bit-depth TIFF previews are not supported.")
    if mode in {"I;16", "I;16L", "I;16B"}:
        # Rebuild with a native-endian mode so Pillow writes the samples as 16-bit PNG.
        preview = Image.fromarray(np.asarray(image).astype(np.uint16), mode="I;16")
    elif mode == "I":
        pixels = np.asarray(image)
        if pixels.min() < 0 or pixels.max() > 65535:
            raise TIFFPreviewError("TIFF integer pixels must be in the unsigned 16-bit range for preview.")
        preview = Image.fromarray(pixels.astype(np.uint16), mode="I;16")
    elif mode == "F":
        pixels = np.asarray(image, dtype=np.float64)
        if not np.isfinite(pixels).all():
            raise TIFFPreviewError("TIFF float pixels must all be finite for preview.")
        minimum, maximum = float(pixels.min()), float(pixels.max())
        if minimum == maximum:
            raise TIFFPreviewError("A constant float TIFF has no display range for preview.")
        value_range = maximum - minimum
        if not np.isfinite(value_range):
            raise TIFFPreviewError("The float TIFF range is too wide for a reliable preview.")
        scaled = np.rint((pixels - minimum) * (65535.0 / value_range)).astype(np.uint16)
        preview = Image.fromarray(scaled, mode="I;16")
    elif mode in {"1", "L", "LA", "P", "RGB", "RGBA"}:
        preview = image
    elif mode in {"CMYK", "YCbCr", "LAB", "HSV"}:
        preview = image.convert("RGB")
    else:
        raise TIFFPreviewError(f"TIFF pixel mode {mode!r} is not supported for browser preview.")

    output = io.BytesIO()
    try:
        preview.save(output, format="PNG")
    except (OSError, ValueError) as exc:
        raise TIFFPreviewError("Could not encode this TIFF as a browser preview.") from exc
    return output.getvalue()


def tiff_preview_bytes(file_path: Union[str, Path]) -> bytes | None:
    """Return a PNG display copy for TIFF files, or ``None`` for other formats."""
    path = Path(file_path)
    try:
        image = Image.open(path)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        if path.suffix.lower() in {".tif", ".tiff"}:
            raise TIFFPreviewError("Could not read this TIFF image for browser preview.") from exc
        return None

    with image:
        if image.format != "TIFF":
            return None
        try:
            return _tiff_png_bytes(image)
        except TIFFPreviewError:
            raise
        except (OSError, ValueError, SyntaxError) as exc:
            raise TIFFPreviewError("Could not decode this TIFF image for browser preview.") from exc


def load_image_for_display(image, as_thumbnail: bool, as_base64: bool):
    """Use PNG display bytes for TIFFs and preserve the existing loader otherwise."""
    model = ImageModel.from_db(image)
    if not as_base64:
        if as_thumbnail:
            return model.load_thumbnail(as_base64=False)
        return model.load_image(as_base64=False)

    file_path = image.thumbnail_file_path if as_thumbnail else image.file_path
    preview = tiff_preview_bytes(file_path)
    if preview is None:
        if as_thumbnail:
            return model.load_thumbnail(as_base64=as_base64)
        return model.load_image(as_base64=as_base64)
    return base64.b64encode(preview).decode("ascii")


def load_images_for_display(images, as_thumbnail: bool, as_base64: bool):
    return {
        image.id: load_image_for_display(image, as_thumbnail, as_base64)
        for image in images
    }


async def save_image_to_disk(
        image: Union[UploadFile, np.ndarray],
        file_path: Path,
        thumbnail_path: Path
) -> tuple[int, int, str]:
    """
    Save a full-resolution image to ``file_path`` and a downscaled preview (<=500px
    on the longest side, aspect ratio preserved) to ``thumbnail_path``.

    Returns the *native* ``(width, height, color_mode)`` of the full-resolution
    image.

    NOTE: ``PIL.Image.thumbnail`` resizes in place. The thumbnail must therefore be
    built from a copy and the native dimensions captured *before* it runs — otherwise
    the returned object carries thumbnail dimensions, which previously leaked into
    ``Images.width/height`` and made the COCO export ~20x too small.

    Uploaded TIFF bytes are kept verbatim after a single-page PNG-preview preflight.
    """
    # Read and validate TIFFs before creating either output file. Their uploaded bytes
    # are kept verbatim; Pillow is used only for decoding and the thumbnail.
    is_tiff = False
    display_preview = None
    if isinstance(image, UploadFile):
        # Ensure we are at the start of the file stream
        await image.seek(0)
        content = await image.read()
        try:
            img = Image.open(io.BytesIO(content))
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            if Path(image.filename).suffix.lower() in {".tif", ".tiff"}:
                raise TIFFPreviewError("Could not read this TIFF image.") from exc
            raise
        is_tiff = img.format == "TIFF"
        if is_tiff:
            try:
                display_preview = _tiff_png_bytes(img)
            except TIFFPreviewError:
                img.close()
                raise
            except (OSError, ValueError, SyntaxError) as exc:
                img.close()
                raise TIFFPreviewError("Could not decode this TIFF image for browser preview.") from exc
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}.")

    # Save the full-resolution image and capture its native dimensions before the
    # thumbnail (which mutates `img` in place) runs.
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if is_tiff:
        file_path.write_bytes(content)
    else:
        img.save(file_path)
    native_width, native_height, color_mode = img.width, img.height, img.mode

    # Build the preview on a copy so `img`'s native dimensions are preserved.
    thumbnail = Image.open(io.BytesIO(display_preview)) if is_tiff else img.copy()
    thumbnail.thumbnail((500, 500))
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail.save(thumbnail_path)

    logger.info(f"Saved image to disk at {file_path} and thumbnail at {thumbnail_path}.")
    return native_width, native_height, color_mode


async def process_and_save_image(
        file: UploadFile,
        dataset_id: int,
        dataset_folder: str,
        db: Session
) -> int:
    """Internal logic to save one image and its thumbnail."""
    image_folder = Path(dataset_folder) / "images"
    file_path = image_folder / file.filename
    thumbnail_name = file.filename
    if Path(file.filename).suffix.lower() in {".tif", ".tiff"}:
        thumbnail_name = f"{file.filename}.png"
    thumbnail_path = Path(THUMBNAILS_DIR) / thumbnail_name

    native_width, native_height, color_mode = await save_image_to_disk(file, file_path, thumbnail_path)

    new_entry = Images(
        file_name=file.filename,
        file_path=str(file_path),
        thumbnail_file_path=str(thumbnail_path),
        dataset_id=dataset_id,
        width=native_width,
        height=native_height,
        color_mode=color_mode,
    )

    # Add to session but DON'T commit yet
    db.add(new_entry)
    db.flush()  # This populates new_entry.id without ending the transaction

    # Mask logic
    await create_new_mask(new_entry.id, dataset_folder, db)

    db.commit()
    return new_entry.id


def native_image_size(file_path: Union[str, Path]) -> tuple[int, int]:
    """Return the native ``(width, height)`` of an image file.

    ``PIL.Image.open`` is lazy and only the header is parsed to read ``.size``, so
    this does not decode the full image and is cheap to call per file.
    """
    with Image.open(file_path) as img:
        return img.width, img.height


async def backfill_image_dimensions(
        db: Session,
        dataset_id: int | None = None,
) -> dict:
    """Repair ``Images.width/height`` from the full-resolution file on disk.

    Rows ingested before the thumbnail-mutation bug was fixed stored thumbnail
    dimensions instead of the native ones. This recomputes the dimensions from the
    original file at ``file_path`` (the thumbnail lives separately at
    ``thumbnail_file_path``, so the original is untouched) and updates any mismatch.

    Pass ``dataset_id`` to limit the repair to a single dataset; otherwise every
    image is checked. Returns the ids that were corrected and any whose original
    file is missing.

    Geometry metrics (area, perimeter, ...) of the affected contours are
    recomputed as well: they were derived in pixel space from the wrong
    dimensions, so they are off by the same factor squared. The normalized
    contour coordinates themselves are dimension-free and stay untouched.
    """
    query = db.query(Images)
    if dataset_id is not None:
        query = query.filter_by(dataset_id=dataset_id)

    corrected: list[int] = []
    missing: list[int] = []
    for image in query.all():
        if not os.path.exists(image.file_path):
            missing.append(image.id)
            continue
        width, height = native_image_size(image.file_path)
        if (image.width, image.height) != (width, height):
            logger.info(
                "Correcting image %s dimensions %sx%s -> %sx%s",
                image.id, image.width, image.height, width, height,
            )
            image.width = width
            image.height = height
            corrected.append(image.id)

    recomputed_contours = 0
    if corrected:
        db.commit()
        recomputed_contours = _recompute_geometry_for_images(db, corrected)
        db.commit()
    return {"corrected": corrected, "missing": missing,
            "recomputed_contours": recomputed_contours}


def _recompute_geometry_for_images(db: Session, image_ids: list[int]) -> int:
    """Re-derive the geometry metrics of every contour on the given images.

    Reuses the write path's dual-write (legacy columns + tall ``contour_metrics``
    rows), so the repaired values land in both stores consistently.
    """
    from app.database.contours import Contours, dual_write_geometry_metrics

    recomputed = 0
    masks = db.query(Masks).filter(Masks.image_id.in_(image_ids)).all()
    for mask in masks:
        contours = db.query(Contours).filter_by(mask_id=mask.id).all()
        if not contours:
            continue
        dual_write_geometry_metrics(db, mask.id, contours)
        recomputed += len(contours)
    return recomputed


async def delete_image(
        image_id: int,
        db: Session
):
    """Delete an image."""
    image = db.query(Images).filter_by(id=image_id).first()
    if not image:
        raise KeyError(f"Image with id {image_id} was not found.")
    if os.path.exists(image.file_path):
        os.remove(image.file_path)  # Remove the original image file
    if os.path.exists(image.thumbnail_file_path):
        os.remove(image.thumbnail_file_path)  # Remove the thumbnail
    db.delete(image)
    db.commit()


async def get_image_data(
        image_id: int,
        as_thumbnail: bool,
        as_base64: bool,
        db: Session
) -> Images:
    image_query = db.query(Images).filter_by(id=image_id).first()
    return await run_in_threadpool(
        load_image_for_display, image_query, as_thumbnail, as_base64
    )


async def get_images_data(
        image_ids: list[int],
        as_thumbnail: bool,
        as_base64: bool,
        db: Session
):
    images_query = db.query(Images).filter(Images.id.in_(image_ids)).all()
    return await run_in_threadpool(
        load_images_for_display, images_query, as_thumbnail, as_base64
    )


async def get_masks_of_image(
        image_id: int,
        db: Session
):
    return db.query(Masks).filter_by(image_id=image_id).all()

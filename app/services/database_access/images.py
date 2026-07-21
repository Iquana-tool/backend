import io
import os
from logging import getLogger
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from iquana_toolbox.schemas.database.image import Image as ImageModel
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile

from app.database.images import Images
from app.database.masks import Masks
from app.services.database_access.masks import create_new_mask
from config import THUMBNAILS_DIR

logger = getLogger(__name__)


async def save_image_to_disk(
        image: Union[UploadFile, np.ndarray],
        file_path: Path,
        thumbnail_path: Path
) -> tuple[int, int, str]:
    """
    Save a full-resolution image to ``file_path`` and a downscaled preview (<=200px
    on the longest side, aspect ratio preserved) to ``thumbnail_path``.

    Returns the *native* ``(width, height, color_mode)`` of the full-resolution
    image.

    NOTE: ``PIL.Image.thumbnail`` resizes in place. The thumbnail must therefore be
    built from a copy and the native dimensions captured *before* it runs — otherwise
    the returned object carries thumbnail dimensions, which previously leaked into
    ``Images.width/height`` and made the COCO export ~20x too small.
    """
    # Read the image
    if isinstance(image, UploadFile):
        # Ensure we are at the start of the file stream
        await image.seek(0)
        content = await image.read()
        img = Image.open(io.BytesIO(content))
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}.")

    # Save the full-resolution image and capture its native dimensions before the
    # thumbnail (which mutates `img` in place) runs.
    img.save(file_path)
    native_width, native_height, color_mode = img.width, img.height, img.mode

    # Build the preview on a copy so `img`'s native dimensions are preserved.
    thumbnail = img.copy()
    thumbnail.thumbnail((200, 200))
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
    os.makedirs(image_folder, exist_ok=True)
    file_path = image_folder / file.filename
    thumbnail_path = Path(THUMBNAILS_DIR) / file.filename

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

    if corrected:
        db.commit()
    return {"corrected": corrected, "missing": missing}


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
    image = ImageModel.from_db(image_query)
    if as_thumbnail:
        return image.load_thumbnail(as_base64=as_base64)
    return image.load_image(as_base64=as_base64)


async def get_images_data(
        image_ids: list[int],
        as_thumbnail: bool,
        as_base64: bool,
        db: Session
):
    images_query = db.query(Images).filter(Images.id.in_(image_ids)).all()
    if as_thumbnail:
        images = {
            image_query.id:
                ImageModel.from_db(image_query).load_thumbnail(as_base64)
            for image_query in images_query
        }
    else:
        images = {
            image_query.id:
                ImageModel.from_db(image_query).load_image(as_base64)
            for image_query in images_query
        }
    return images


async def get_masks_of_image(
        image_id: int,
        db: Session
):
    return db.query(Masks).filter_by(image_id=image_id).all()

import io
import os
from logging import getLogger
from os.path import join
from pathlib import Path
from typing import Union

import cv2 as cv
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile

from app.database import get_context_session
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.masks import Masks
from app.database.scans import Scans
from config import THUMBNAILS_DIR

logger = getLogger(__name__)


def get_height_width_of_image(image_id: int) -> tuple[int, int]:
    """Get the height and width of an image from the database by its ID."""
    with get_context_session() as session:
        image = session.query(Images).filter_by(id=image_id).first()
    if image:
        return image.height, image.width
    else:
        raise ValueError(f"Image with ID {image_id} not found in database.")


def save_array_to_disk(array: np.ndarray, dataset_id: int, scan_id: int = None,
                       new_filename: str = None, is_mask: bool = False) -> str:
    """Save a numpy array as an image file to disk."""
    with get_context_session() as session:
        dataset = session.query(Datasets).filter_by(id=dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found.")
        if dataset.dataset_type == "scan":
            if scan_id is None:
                raise ValueError("Scan ID must be provided for scan datasets.")
            scan = session.query(Scans).filter_by(id=scan_id).first()
            path = join(scan.folder_path, "masks" if is_mask else "slices")
        else:
            path = join(dataset.folder_path, "masks" if is_mask else "images")
    os.makedirs(path, exist_ok=True)
    file_path = join(path, new_filename if new_filename else "image.png")
    if is_mask:
        # Masks must be saved as PNG. Other file formats might be lossy and change the values!
        file_path = file_path.rsplit(".", 1)[0] + ".png"
    cv.imwrite(file_path, array)
    logger.info(f"Image saved to disk at {file_path}")
    return str(file_path)


async def save_image_to_disk(image: Union[UploadFile, np.ndarray],
                             file_path: Path,
                             thumbnail_path: Path):
    """
    Save an image file to disk. If as_thumbnail is True, the image is downsized to 50 x 50 resolution before saving.
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

    # Save the processed image to the file path
    img.save(file_path)

    # Resize using thumbnail (maintains aspect ratio) or resize (forces 50x50)
    img.thumbnail((100, 100))
    img.save(thumbnail_path)

    logger.info(f"Saved image to disk at {file_path} and thumbnail at {thumbnail_path}.")
    return img


async def process_and_save_image(
        file: UploadFile,
        dataset_id: int,
        dataset_folder: str,
        db: Session
) -> int:
    """Internal logic to save one image and its thumbnail."""
    file_path = Path(dataset_folder) / file.filename
    thumbnail_path = Path(THUMBNAILS_DIR) / file.filename

    # We pass the same UploadFile to save_image_to_disk twice.
    # IMPORTANT: The fix we discussed earlier (await file.seek(0)) is critical here!
    img = await save_image_to_disk(file, file_path, thumbnail_path)

    new_entry = Images(
        file_name=file.filename,
        file_path=str(file_path),
        thumbnail_file_path=str(thumbnail_path),
        dataset_id=dataset_id,
        width=img.width,
        height=img.height,
        color_mode=img.mode,
    )

    # Add to session but DON'T commit yet
    db.add(new_entry)
    db.flush()  # This populates new_entry.id without ending the transaction

    # Mask logic
    await create_new_mask(new_entry.id, db)

    return new_entry.id


async def create_new_mask(
        image_id: int,
        db: Session,
):
    new_mask = Masks(image_id=image_id)
    db.add(new_mask)
    db.flush()
    return new_mask

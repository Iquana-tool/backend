import io
import os
from logging import getLogger
from os.path import join
from pathlib import Path
from typing import Union

import cv2 as cv
import numpy as np
from PIL import Image
from fastapi import UploadFile

from app.database import get_context_session
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.scans import Scans

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


async def save_image_to_disk(image: Union[UploadFile, np.ndarray], file_path: Path, as_thumbnail: bool = False):
    """
    Save an image file to disk. If as_thumbnail is True, the image is downsized to 50 x 50 resolution before saving.
    """
    # Read the image
    if isinstance(image, UploadFile):
        content = await image.read()
        img = Image.open(io.BytesIO(content))
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}.")

    # Save the image
    if as_thumbnail:
        # Resize using thumbnail (maintains aspect ratio) or resize (forces 50x50)
        img.thumbnail((50, 50))
    # Save the processed image to the file path
    img.save(file_path)
    return img

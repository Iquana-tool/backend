import io
import os
from logging import getLogger
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from iquana_toolbox.schemas.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.labels import LabelHierarchy
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile

from app.database import get_context_session
from app.database.contours import Contours
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from config import THUMBNAILS_DIR

logger = getLogger(__name__)


async def save_semantic_mask(
        semantic_mask: np.ndarray,
        file_path: Path,
):
    # Convert to PIL image & save
    semantic_mask = Image.fromarray(semantic_mask, mode="L")  # <- Saves as a greyscale image, tiny file size
    semantic_mask.save(file_path)


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
    img.thumbnail((200, 200))
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
    image_folder = Path(dataset_folder) / "images"
    os.makedirs(image_folder, exist_ok=True)
    file_path = image_folder / file.filename
    thumbnail_path = Path(THUMBNAILS_DIR) / file.filename

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
    await create_new_mask(new_entry.id, dataset_folder, db)

    return new_entry.id


async def create_new_mask(
        image_id: int,
        dataset_folder: str,
        db: Session,
):
    mask_folder = Path(dataset_folder) / "masks"
    os.makedirs(mask_folder, exist_ok=True)
    mask_path = mask_folder / f"{image_id}.png"
    new_mask = Masks(
        image_id=image_id,
        file_path=str(mask_path),
    )
    db.add(new_mask)
    db.flush()
    return new_mask

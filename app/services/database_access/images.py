import io
import os
from logging import getLogger
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from iquana_toolbox.schemas.image import Image as ImageModel
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
):
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

    db.commit()
    return new_entry.id


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
            image_query:
                ImageModel.from_db(image_query).load_thumbnail(as_base64)
            for image_query in images_query
        }
    else:
        images = {
            image_query:
                ImageModel.from_db(image_query).load_image(as_base64)
            for image_query in images_query
        }
    return images


async def get_masks_of_image(
        image_id: int,
        db: Session
):
    return db.query(Masks).filter_by(image_id=image_id).all()

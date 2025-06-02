import base64
import hashlib
import logging
import os
import cv2 as cv
from logging import getLogger
from os import remove
from os.path import join, exists
from typing import Union, AnyStr

import numpy as np
from fastapi import UploadFile
from PIL import Image

import config
from app.database import get_context_session
from app.database.images import Images, ImageEmbeddings
from app.schemas.segmentation_and_masks import SegmentationRequest

logger = getLogger(__name__)


# Function to generate hash of a file
def generate_hash_for_image(image: UploadFile):
    """Generate a hash for the given image file."""
    hasher = hashlib.sha256()
    image.file.seek(0)  # Reset file pointer to the beginning
    while True:
        data = image.file.read(65536)  # Read in 64k chunks
        if not data:
            break
        hasher.update(data)
    image.file.seek(0)  # Reset file pointer to the beginning again for further use
    return hasher.hexdigest()


def delete_image_from_disk_and_db(image_id: int):
    """Deletes the image files and the embeddings"""
    with get_context_session() as session:
        image = session.query(Images).filter_by(id=image_id).first()
        cutouts_to_delete = session.query(Images).filter_by(parent_image_id=image_id).all()
        for cutout in cutouts_to_delete:
            cutout_path = join(config.Paths.images_dir, cutout.filename)
            if exists(cutout_path):
                remove(cutout_path)
            session.delete(cutout)
        embeddings = session.query(ImageEmbeddings).filter_by(id=image_id).all()
        for embedding in embeddings:
            embedding_path = join(config.Paths.images_dir, embedding.filename)
            if exists(embedding_path):
                remove(embedding_path)
            session.delete(embedding)
        image_path = join(config.Paths.images_dir, image.filename)
        if exists(image_path):
            remove(image_path)
        session.delete(image)
        session.commit()


def load_image_as_base64_from_disk(image_id):
    """Load an image from the database by its ID and return it as a base64 string."""
    with get_context_session() as session:
        image = session.query(Images).filter_by(id=image_id).first()
    if image:
        path = join(config.Paths.images_dir, image.filename)
        with open(path, "rb") as image_file:
            image = image_file.read()
        # Encode the image to base64
        return base64.b64encode(image)
    else:
        raise ValueError(f"Image with ID {image_id} not found in database.")


def load_image_as_array_from_disk(image_id):
    """Load an image from the database by its ID."""
    with get_context_session() as session:
        image_query_result = session.query(Images).filter_by(id=image_id).first()
    if image_query_result:
        image = np.array(cv.imread(join(config.Paths.images_dir, image_query_result.filename)))
        if image.shape[0] == image_query_result.width and image.shape[1] == image_query_result.height:
            logger.warning(f"Image {image_id} has different dimensions than expected.")
            image = np.moveaxis(image, 1, 0)
        if image.shape[-1] != 3:
            logger.warning("Converting RGBA image to RGB.")
            image = cv.cvtColor(image, cv.COLOR_RGBA2RGB)
        return np.array(image)
    else:
        return None


def get_height_width_of_image(image_id: int) -> tuple[int, int]:
    """Get the height and width of an image from the database by its ID."""
    with get_context_session() as session:
        image = session.query(Images).filter_by(id=image_id).first()
    if image:
        return image.height, image.width
    else:
        raise ValueError(f"Image with ID {image_id} not found in database.")


def save_embedding(request: SegmentationRequest, embedding: dict[str, Union[np.ndarray, list[np.ndarray]]]):
    with get_context_session() as db:
        new_embedding = ImageEmbeddings(
            image_id=request.image_id,
            model=request.model,
            embed_dimensions=str(embedding["image_embed"].shape),
        )
        db.add(new_embedding)
        db.commit()
        save_embeddings_to_disk(embedding, new_embedding.image_id, new_embedding.model)


def save_embeddings_to_disk(embedding: dict[str, Union[np.ndarray, list[np.ndarray]]], image_id: int,
                            model_name: str) -> None:
    """ Save an image embedding to disk.
        Args:
            embedding (dict[str, Union[np.ndarray, list[np.ndarray]]]): The embedding to save.
            image_id (int): The ID of the image embedding.
            model_name (str): The name of the model used to generate the embedding.
    """
    base_path = join(config.Paths.embedding_dir, str(image_id))
    os.makedirs(base_path, exist_ok=True)
    path = join(base_path, model_name + ".npz")
    new_dict = {"image_embed": embedding["image_embed"]}
    for i, mask in enumerate(embedding["high_res_feats"]):
        new_dict[f"high_res_feats_{i}"] = mask
    np.savez_compressed(str(path), **new_dict)


async def save_image_to_disk_and_db(image: AnyStr, dataset_id: int, scan_id=None, index_in_scan=None) -> int:
    """Save an image to disk and to the database and return the new image ID."""
    image_data = image.file.read()

    # Generate hash for the image
    hash_code = generate_hash_for_image(image)

    # Check if image already exists in the database
    with get_context_session() as session:
        images_with_hash = session.query(Images).filter_by(hash_code=hash_code).all()
        if images_with_hash and dataset_id in [image.dataset_id for image in images_with_hash]:
            return session.query(Images).filter_by(dataset_id=dataset_id, hash_code=hash_code).first().id
        else:
            next_id = session.query(Images).count() + 1
            if images_with_hash:
                file_name = images_with_hash[0].filename
            else:
                # Save the new image to disk
                original_extension = image.filename.split(".")[-1]
                file_name = f"{next_id}.{original_extension}"
                path = join(config.Paths.images_dir, file_name)
                with open(path, "wb") as file:
                    file.write(image_data)
                image_array = np.array(Image.open(path))
            try:
                # Save the new image to the database
                # Image comes in WHC format because of PIL
                new_entry = Images(filename=file_name,
                                   dataset_id=dataset_id,
                                   width=image_array.shape[1],
                                   height=image_array.shape[0],
                                   scan_id=scan_id,
                                   index_in_scan=index_in_scan,
                                   hash_code=hash_code)
                session.add(new_entry)
                session.commit()
                logger.info("New image saved to disk and database.")
                return new_entry.id
            except Exception as e:
                logger.error(f"Error saving image to database: {str(e)}")
                logger.error(f"Deleting image '{file_name}' from disk to ensure consistency.")
                os.remove(path)
                return None


def parse_log_file(log_file: AnyStr):
    """Parse the log file and return the log entries."""
    meta_data = {}
    with open(log_file, "r") as file:
        for line in file:
            line = line.strip()  # Ignore empty lines
            if not (line.startswith("[") and line.endswith("]")):
                # Extract the key and value from the line
                key, value = line[1:-1].split("=", 1)
                key = key.strip()
                value = value.strip()
                # Add the key-value pair to the meta_data dictionary
                meta_data[key] = value
    return meta_data

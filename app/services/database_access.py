from typing import Union

import numpy as np
from PIL import Image
from fastapi import UploadFile
import config
from os.path import join, exists
from os import remove
from app.database.images import Images, ImageEmbeddings
from app.database import get_session, get_context_session
from logging import getLogger
import hashlib

logger = getLogger(__name__)


# Function to generate hash of a file
def generate_hash_for_image(image: UploadFile):
    """Generate a hash for the given image file."""
    hasher = hashlib.sha256()
    while True:
        data = image.file.read(65536)  # Read in 64k chunks
        if not data:
            break
        hasher.update(data)
    return hasher.hexdigest()


def delete_image_from_disk_and_db(image_id: int):
    """Deletes the image files and the embeddings"""
    with get_context_session() as session:
        image = session.query(Images).filter_by(id=image_id).first()
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
        with open(join(config.Paths.images_dir, image.filename), "rb") as file:
            return str(file.read())
    else:
        raise ValueError(f"Image with ID {image_id} not found in database.")


def load_image_as_array_from_disk(image_id):
    """Load an image from the database by its ID."""
    with get_context_session() as session:
        image = session.query(Images).filter_by(id=image_id).first()
    if image:
        return np.array(Image.open(join(config.Paths.images_dir, image.filename)))
    else:
        return None


def load_embedding(embedding_id: int):
    """Load an image embedding from the database by its image ID."""
    with get_context_session() as session:
        embedding = session.query(ImageEmbeddings).filter_by(id=embedding_id).first()
    if embedding:
        try:
            loaded_data = np.load(join(config.Paths.embedding_dir, str(embedding.id) + ".npz"))
            files = set(loaded_data.files)
            new_dict = {"image_embed": loaded_data["image_embed"]}
            files.remove("image_embed")
            new_dict["high_res_feats"] = [loaded_data[high_res_feat] for high_res_feat in files]
            return new_dict
        except FileNotFoundError:
            logger.warning(f"File not found for embedding ID {embedding_id}.")
            return None
    else:
        return None


def save_embeddings_to_disk(embedding: dict[str, Union[np.ndarray, list[np.ndarray]]], embedding_id: int) -> None:
    """ Save an image embedding to disk.
        Args:
            embedding (dict[str, Union[np.ndarray, list[np.ndarray]]]): The embedding to save.
            embedding_id (int): The ID of the image embedding.
    """
    path = join(config.Paths.embedding_dir, str(embedding_id) + ".npz")
    new_dict = {"image_embed": embedding["image_embed"]}
    for i, mask in enumerate(embedding["high_res_feats"]):
        new_dict[f"high_res_feats_{i}"] = mask
    np.savez_compressed(str(path), **new_dict)


async def save_image_to_disk_and_db(image: UploadFile):
    """Save an image to disk and to the database and return the new image ID."""
    image_data = image.file.read()

    # Generate hash for the image
    hash_code = generate_hash_for_image(image)

    # Check if image already exists in the database
    with get_context_session() as session:
        if session.query(Images).filter_by(hash_code=hash_code).first():
            logger.info("Image already exists in the database.")
            return session.query(Images).filter_by(hash_code=hash_code).first().id
        else:
            next_id = session.query(Images).count() + 1

    # Save the new image to disk
    original_extension = image.filename.split(".")[-1]
    new_file_name = f"{next_id}.{original_extension}"
    path = join(config.Paths.images_dir, new_file_name)
    with open(path, "wb") as file:
        file.write(image_data)
    image_array = np.array(Image.open(path))

    # Save the new image to the database
    with get_context_session() as session:
        session.add(Images(filename=new_file_name,
                           width=image_array.shape[1],
                           height=image_array.shape[0],
                           hash_code=hash_code))
        session.commit()
    logger.info("New image saved to disk and database.")
    return session.query(Images).order_by(Images.id.desc()).first().id


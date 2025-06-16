import base64
import hashlib
import os
import cv2 as cv
from logging import getLogger
from os import remove
from os.path import join, exists
from typing import Union, AnyStr

import numpy as np
from fastapi import UploadFile

import paths
from app.database import get_context_session
from app.database.datasets import Datasets
from app.database.images import Images, Scans

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
        if exists(image.file_path):
            remove(image.file_path)
        session.delete(image)
        session.commit()


def load_image_as_base64_from_disk(image_id):
    """Load an image from the database by its ID and return it as a base64 string."""
    with get_context_session() as session:
        image = session.query(Images).filter_by(id=image_id).first()
    if image:
        with open(image.file_path, "rb") as image_file:
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
        image = np.array(cv.imread(image_query_result.file_path, cv.IMREAD_COLOR_RGB))
        if image.shape[0] == image_query_result.width and image.shape[1] == image_query_result.height:
            logger.warning(f"Image {image_id} has different dimensions than expected.")
            image = np.moveaxis(image, 1, 0)
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


def get_height_width_of_scan(scan_id: int) -> tuple[int, int]:
    """Get the height and width of a scan from the database by its ID."""
    with get_context_session() as session:
        scan = session.query(Scans).filter_by(id=scan_id).first()
        if scan:
            image = session.query(Images).filter_by(id=scan.id).first()
            return image.height, image.width
        else:
            raise ValueError(f"Scan with ID {scan_id} not found in database.")


def get_image_id_via_scan_index(scan_id: int, index_in_scan: int, reset_index: bool = False) -> int:
    with get_context_session() as session:
        scan = session.query(Scans).filter_by(id=scan_id).first()
        if not scan:
            raise ValueError(f"Scan with ID {scan_id} not found.")
        if not reset_index:
            return session.query(Images).filter_by(id=scan.id, index_in_scan=index_in_scan).first().id
        else:
            # The index given does not match the index in the database, so we need to reset it.
            min_index = session.query(Images).filter_by(scan_id=scan_id).order_by(Images.index_in_scan).first().index_in_scan
            return session.query(Images).filter_by(scan_id=scan_id, index_in_scan=index_in_scan + min_index).first().id


def save_embeddings_to_disk(embedding: dict[str, Union[np.ndarray, list[np.ndarray]]], image_id: int,
                            model_name: str) -> None:
    """ Save an image embedding to disk.
        Args:
            embedding (dict[str, Union[np.ndarray, list[np.ndarray]]]): The embedding to save.
            image_id (int): The ID of the image embedding.
            model_name (str): The name of the model used to generate the embedding.
    """
    base_path = join(paths.Paths.embedding_dir, str(image_id))
    os.makedirs(base_path, exist_ok=True)
    path = join(base_path, model_name + ".npz")
    new_dict = {"image_embed": embedding["image_embed"]}
    for i, mask in enumerate(embedding["high_res_feats"]):
        new_dict[f"high_res_feats_{i}"] = mask
    np.savez_compressed(str(path), **new_dict)


def save_image_to_disk(image: UploadFile, dataset_id: int, scan_id: int = None) -> str:
    """Save an image file to disk."""
    with get_context_session() as session:
        dataset = session.query(Datasets).filter_by(id=dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found.")
        if dataset.dataset_type == "scan":
            if scan_id is None:
                raise ValueError("Scan ID must be provided for scan datasets.")
            scan = session.query(Scans).filter_by(id=scan_id).first()
            path = join(scan.folder_path, "slices")
        else:
            path = join(dataset.folder_path, "images")
    os.makedirs(path, exist_ok=True)
    file_path = join(path, image.filename)
    with open(file_path, "wb") as file:
        file.write(image.file.read())
    logger.info(f"Image saved to disk at {file_path}")
    return str(file_path)


async def save_image_to_disk_and_db(image: AnyStr, dataset_id: int, scan_id=None, index_in_scan=None, convert_to: str = None) -> int:
    """Save an image to disk and to the database and return the new image ID."""
    # Generate hash for the image
    hash_code = generate_hash_for_image(image)

    # Check if image already exists in the database
    with get_context_session() as session:
        images_with_hash = session.query(Images).filter_by(hash_code=hash_code, dataset_id=dataset_id).all()
        if images_with_hash:
            return session.query(Images).filter_by(dataset_id=dataset_id, hash_code=hash_code).first().id
        else:
            file_path = save_image_to_disk(image, dataset_id, scan_id)
            image_array = np.array(cv.imread(file_path))
            if convert_to:
                os.remove(file_path)
                file_path = file_path.split(".")[0] + f".{convert_to}"
                cv.imwrite(file_path, image_array)
            try:
                # Save the new image to the database
                # Image comes in HWC format
                new_entry = Images(file_name=image.filename,
                                   file_path=file_path,
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
                logger.error(f"Deleting image '{image.file_name}' from disk to ensure consistency.")
                os.remove(file_path)
                return None


def get_scan_image_folder_path(scan_id: int) -> str:
    """Get the folder path where the scan images are stored."""
    with get_context_session() as session:
        scan = session.query(Scans).filter_by(id=scan_id).first()
        if not scan:
            raise ValueError(f"Scan with ID {scan_id} not found.")
        return scan.folder_path + "/images"


def get_image_query(image_id: int):
    """Get the image query from the database by its ID."""
    with get_context_session() as session:
        image = session.query(Images).filter_by(id=image_id).first()
        if not image:
            raise ValueError(f"Image with ID {image_id} not found.")
        return image


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

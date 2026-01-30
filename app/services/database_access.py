import hashlib
import os
from logging import getLogger
from os.path import join
from typing import AnyStr

import cv2 as cv
import numpy as np
from fastapi import UploadFile

from app.database import get_context_session
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.scans import Scans
from config import THUMBNAILS_DIR

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


def save_image_to_disk(image: UploadFile, dataset_id: int, scan_id: int = None, new_filename: str = None) -> str:
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
    file_path = join(path, image.filename if not new_filename else new_filename + "." + image.filename.split(".")[-1])
    with open(file_path, "wb") as file:
        file.write(image.file.read())
    logger.info(f"Image saved to disk at {file_path}")
    return str(file_path)


async def save_image_to_disk_and_db(image: AnyStr,
                                    dataset_id: int,
                                    scan_id=None,
                                    index_in_scan=None,
                                    convert_to: str = None) -> int:
    """Save an image to disk and to the database and return the new image ID.

    Args:
        image (AnyStr): The image file to save.
        dataset_id (int): The ID of the dataset to which the image belongs.
        scan_id (int, optional): The ID of the scan if applicable.
        index_in_scan (int, optional): The index of the image in the scan.
        convert_to (str, optional): Format to convert the image to (e.g., "png", "jpg").

    Returns:
        int: The ID of the newly saved image in the database, or None if an error occurs.
    """
    # Check if image already exists in the database
    with get_context_session() as session:
        file_path = save_image_to_disk(image, dataset_id, scan_id,
                                       new_filename=str(index_in_scan) if index_in_scan is not None else None)
        thumbnail_path = save_as_low_res_image_to_disk(np.array(cv.imread(file_path)),
                                                       filename=str(index_in_scan) if index_in_scan is not None else
                                                       image.filename)
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
                               thumbnail_file_path=thumbnail_path,
                               dataset_id=dataset_id,
                               width=image_array.shape[1],
                               height=image_array.shape[0],
                               channels=image_array.shape[2] if len(image_array.shape) > 2 else 1,
                               scan_id=scan_id,
                               index_in_scan=index_in_scan,
                            )
            session.add(new_entry)
            session.commit()
            logger.info("New image saved to disk and database.")
            return new_entry.id
        except Exception as e:
            logger.error(f"Error saving image to database: {str(e)}")
            logger.error(f"Deleting image '{image.file_name}' from disk to ensure consistency.")
            os.remove(file_path)
            return None


def save_as_low_res_image_to_disk(image: np.ndarray, filename: str) -> str:
    """Save a low resolution version of the image to disk."""
    # Scale down the image to low resolution of max 256 pixels on the longest side
    max_size = 256
    height, width = image.shape[:2]
    if height > width:
        scale_factor = max_size / height
    else:
        scale_factor = max_size / width
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    low_res_path = join(THUMBNAILS_DIR, f"{filename}.png")
    cv.imwrite(low_res_path, image)
    logger.info(f"Low resolution image saved to disk at {low_res_path}")
    return low_res_path


def get_scan_image_folder_path(scan_id: int) -> str:
    """Get the folder path where the scan images are stored."""
    with get_context_session() as session:
        scan = session.query(Scans).filter_by(id=scan_id).first()
        if not scan:
            raise ValueError(f"Scan with ID {scan_id} not found.")
        return scan.folder_path + "/slices"


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

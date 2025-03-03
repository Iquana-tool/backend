import base64
from typing import Union

import cv2
import numpy as np


def rle_encode(mask: np.ndarray) -> str:
    """Run-length encode a binary mask."""
    pixels = mask.flatten()
    rle = []
    prev_pixel = 0
    count = 0

    for pixel in pixels:
        if pixel == prev_pixel:
            count += 1
        else:
            if prev_pixel == 1:
                rle.append(count)
            count = 1
            prev_pixel = pixel

    if prev_pixel == 1:
        rle.append(count)

    return ' '.join(map(str, rle))


def base64_encode_image(image: Union[np.ndarray, str]) -> str:
    """Encode an image to base64.
    Args:
        image: Image as a numpy array or file path.
    Returns:
        Base64 encoded string of the image.
    """
    if isinstance(image, np.ndarray):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    elif isinstance(image, str):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        raise ValueError("Input must be a numpy array or a file path.")

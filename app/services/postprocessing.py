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


def rle_decode(rle_str: str, height: int, width: int) -> np.ndarray:
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(max(ends), dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    return mask.reshape((height, width))  # Replace height and width with actual dimensions


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


def base64_decode_string(image: str) -> np.ndarray:
    """Decode a base64 string to a numpy array.
    Args:
        image: Base64 encoded image.
    Returns:
        Numpy array of the image.
    """
    return cv2.imdecode(np.frombuffer(base64.b64decode(image), np.uint8), cv2.IMREAD_COLOR)

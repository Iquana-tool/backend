import os
import re

import numpy as np


def extract_numbers(text):
    # This pattern matches positive integers
    pattern = r'\d+'
    numbers = re.findall(pattern, text)
    # Convert the extracted strings to integers
    return [int(num) for num in numbers]


def get_mask_path_from_image_path(path: str):
    """ Given an image path, returns the corresponding mask path. """
    parts = path.split(os.path.sep)
    parts[-2] = "masks"  # Replace the parent directory
    full_path = os.path.sep.join(parts)
    return full_path.rsplit(".", 1)[0] + ".png"


def extract_mask_from_response(response):
    # Extract metadata from headers
    shape = tuple(map(int, response.headers["X-Mask-Shape"].split(',')))
    dtype = np.dtype(response.headers["X-Mask-Dtype"])
    score = float(response.headers.get("X-Score", 0.0))

    # Load the mask from raw bytes
    mask_bytes = response.content
    mask = np.frombuffer(mask_bytes, dtype=dtype).reshape(shape)
    return mask, shape, dtype, score

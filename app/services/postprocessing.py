import cv2 as cv
import numpy as np


def postprocess_binary_mask(mask: np.ndarray):
    """Post-process a binary mask by removing small objects and filling holes."""
    # Ensure mask is binary
    if np.unique(mask).size > 2:
        raise ValueError("Input mask is not binary. Postprocessing requires a binary mask.")

    # Fill holes via Closing
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Add more methods here

    return mask

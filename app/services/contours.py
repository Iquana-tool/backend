from logging import getLogger

import cv2
import numpy as np

logger = getLogger(__name__)


def get_contours_from_binary_mask(mask: np.ndarray, only_return_biggest=False, limit=200) -> np.ndarray:
    """ Get the contours of the mask.
        Args:
            mask (np.ndarray): The mask to get the contours of.
            only_return_biggest (bool): If True, returns only the biggest contour. If False, returns all contours.
            limit (int): Specifies how many contours to return at most. If the detected contours surpass this limit,
                only the largest returns up to the limit will be returned.

        Returns:
            np.ndarray: The contours of the mask.
    """
    logger.debug("Computing contours for mask.")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:  # check if any contours found
        if only_return_biggest:
            biggest = max(contours, key=cv2.contourArea)
            return np.array([biggest])
        else:
            contours = sorted(contours, key=cv2.contourArea)
            if len(contours) > limit:
                logger.warning(f"Detected over {limit} objects. Only returning the biggest 500 objects.")
                contours = contours[:limit]
            return contours
    else:
        return np.array([])

import cv2
import numpy as np


def get_contours(mask: np.ndarray) -> np.ndarray:
    """ Get the contours of the mask.
        Args:
            mask (np.ndarray): The mask to get the contours of.

        Returns:
            np.ndarray: The contours of the mask.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours




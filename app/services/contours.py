import cv2
import numpy as np
from app.schemas.segmentation.contours_and_quantifications import ContourModel


def get_contours(mask: np.ndarray) -> np.ndarray:
    """ Get the contours of the mask.
        Args:
            mask (np.ndarray): The mask to get the contours of.

        Returns:
            np.ndarray: The contours of the mask.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def get_contour_from_coordinates(x_coords: list[float], y_coords: list[float]) -> np.array:
    return np.expand_dims(np.array(list(zip(x_coords, y_coords)), dtype=np.int32), 1)


def create_binary_mask_from_contours(width, height, contours: list[ContourModel]):
    """ Create a mask from the contours.
        Args:
            width (int): The width of the mask.
            height (int): The height of the mask.
            contours (list[ContourModel]): The contours to create the mask from.

        Returns:
            np.ndarray: The mask created from the contours.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for contour in contours:
        contour = get_contour_from_coordinates(contour.x, contour.y)
        cv2.fillPoly(mask, [contour], 1)
    return mask

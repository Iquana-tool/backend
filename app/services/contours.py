import cv2
import numpy as np
from app.schemas.segmentation.contours_and_quantifications import ContourModel


def get_contours(mask: np.ndarray, only_return_one=False) -> np.ndarray:
    """ Get the contours of the mask.
        Args:
            mask (np.ndarray): The mask to get the contours of.

        Returns:
            np.ndarray: The contours of the mask.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:  # check if any contours found
        if only_return_one:
            biggest = max(contours, key=cv2.contourArea)
            return np.array([biggest])
        else:
            return contours
    else:
        return np.array([])


def get_contour_from_coordinates(x_coords: list[float], y_coords: list[float], height=None, width=None) -> np.array:
    if height is not None and width is not None:
        # Ensure coordinates are within bounds
        x_coords = [max(0, min(width - 1, int(x * width))) for x in x_coords]
        y_coords = [max(0, min(height - 1, int(y * height))) for y in y_coords]
    return np.expand_dims(np.array(list(zip(x_coords, y_coords)), dtype=np.int32), 1)


def create_binary_mask_from_contours(width, height, contours: list[np.ndarray]):
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
        try:
            cv2.fillPoly(mask, [contour], 1)
        except cv2.error as e:
            print(contour.shape)
            print(contour)
            raise e
    return mask

import json

import cv2
import cv2 as cv
import numpy as np
from sqlalchemy.orm import Session
from logging import getLogger
from app.database.contours import Contours
from app.schemas.segmentation.contours_and_quantifications import ContourModel


logger = getLogger(__name__)


def get_contours(mask: np.ndarray, only_return_biggest=False) -> np.ndarray:
    """ Get the contours of the mask.
        Args:
            mask (np.ndarray): The mask to get the contours of.
            only_return_biggest (bool): If True, returns only the biggest contour. If False, returns all contours.

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
            if len(contours) > 100:
                logger.warning("Detected over 100 objects. Only returning the biggest 500 objects.")
                contours = contours[:100]
            return contours
    else:
        return np.array([])


def get_contour_from_coordinates(x_coords: list[float], y_coords: list[float], height, width) -> np.array:
    """ Convert lists of x and y coordinates to an opencv contour.

    Args:
        x_coords (list[float]): List of x coordinates.
        y_coords (list[float]): List of y coordinates.
        height (int): The height of the mask/ images.
        width (int): The width of the mask/ image.

    Returns:
        np.ndarray: The contour as a numpy array of shape (N, 1, 2), where N is the number of points.
    """
    if height is not None and width is not None:
        # Ensure coordinates are within bounds
        x_coords = [max(0, min(width - 1, int(x * width))) for x in x_coords]
        y_coords = [max(0, min(height - 1, int(y * height))) for y in y_coords]
    return np.expand_dims(np.array(list(zip(x_coords, y_coords)), dtype=np.int32), 1)


def create_binary_mask_from_contours(width, height, contours: list[np.ndarray]):
    """
        Create a binary mask from the contours.

        Args:
            width (int): The width of the mask.
            height (int): The height of the mask.
            contours (list[ContourModel]): The contours to create the mask from.

        Returns:
            np.ndarray: The mask created from the contours.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for contour in contours:
        cv2.fillPoly(mask, [contour], 1)
    return mask


def build_depth_first_contour_list(contours, parent=None):
    """
    Recursively build a list of contours in depth-first order starting from the given parent contour ID.

    Args:
        contours (list[Contours]): List of Contours objects.
        parent (int): The parent contour ID to start from. If None, starts from the root.

    Returns:
        list[Contours]: A list of Contours objects in depth-first order.
    """
    contour_list = []
    for contour in contours:
        if contour.parent_id == parent:
            contour_list.append(contour)
            contour_list.extend(build_depth_first_contour_list(contours, contour.id))
    return contour_list


def contour_is_contained(contained_contour, containing_contour):
    """
    Check if a contour is contained by another contour.

    Args:
        contained_contour (np.ndarray): The contour that is expected to be contained.
        containing_contour (np.ndarray): The contour that is expected to contain the other contour.

    Returns:
        bool: True if the contained contour is fully contained within the containing contour, False otherwise.
    """
    return np.all([cv.pointPolygonTest(containing_contour, (p[0, 1], p[0,0]), False) >= 0 for p in contained_contour])


def find_parent_contour(parent_label_id, mask_id, image_shape, contour_to_add, db: Session):
    """ Find the parent contour ID for a contour to be added. This is used to ensure that the contour is added to the
    correct parent contour in the database. """
    parent_contours = db.query(Contours).filter_by(label=parent_label_id, mask_id=mask_id).all()
    overlap = []
    for parent_contour in parent_contours:
        coords = json.loads(parent_contour.coords)
        parent_contour_cv = coords_to_cv_contour(coords['x'], coords['y'])
        overlap.append(calculate_contour_overlap(contour_to_add, parent_contour_cv, image_shape))
    if not np.any(overlap):
        return None
    return parent_contours[np.argmax(overlap)]


def contour_overlaps_with_existing(contour, existing_contours):
    """ Check if a contour overlaps with any existing contours. """
    for existing_contour in existing_contours:
        if cv.contourArea(cv.intersectConvexConvex(contour, existing_contour)) > 0:
            return True
    return False


def calculate_contour_overlap(contour_1, contour_2, shape):
    """ Calculate the amount of pixels that two contours overlap. """
    canvas_1 = np.zeros(shape, dtype=np.uint8)
    cv.fillPoly(canvas_1, [contour_1.astype(np.int32)], color=1)
    canvas_2 = np.zeros(shape, dtype=np.uint8)
    cv.fillPoly(canvas_2, [contour_2.astype(np.int32)], color=1)
    return np.sum(np.logical_and(canvas_1, canvas_2))


def coords_to_cv_contour(x_coords, y_coords):
    """ Convert coordinates to a cv2 contour. """
    return np.expand_dims(np.array(list(zip(x_coords, y_coords)), dtype=np.int32), 1)

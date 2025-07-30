import json

import cv2
import cv2 as cv
import numpy as np
from sqlalchemy.orm import Session

from app.database.contours import Contours
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


def build_contour_hierarchy(contours, parent=None):
    """ Recursively build a list of contours, where the contours go from root to leaf. """
    contour_list = []
    for contour in contours:
        if contour.parent_id == parent:
            contour_list.append(contour)
            contour_list.extend(build_contour_hierarchy(contours, contour.id))
    return contour_list


def contour_is_enclosed_by_parent(contour, parent_contour):
    """ Check if a contour is enclosed by its parent contour. """
    if parent_contour is None:
        return True
    return np.all([cv.pointPolygonTest(parent_contour, (p[0, 1], p[0,0]), False) >= 0 for p in contour])


def combine_contours(contour_1, contour_2, canvas_shape):
    """ Combine two contours into one. """
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    cv.fillPoly(canvas, [contour_1.astype(np.int32)], color=1)
    cv.fillPoly(canvas, [contour_2.astype(np.int32)], color=1)
    contours, _ = cv.findContours(canvas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours[0] if contours else None


def find_parent_contour(parent_label_id, mask_id, image_shape, contour_to_add, db: Session):
    """ Find the parent contour ID for a contour to be added. This is used to ensure that the contour is added to the
    correct parent contour in the database. """
    parent_contours = db.query(Contours).filter_by(label=parent_label_id, mask_id=mask_id).all()
    overlap = []
    for parent_contour in parent_contours:
        coords = json.loads(parent_contour.coords)
        parent_contour_cv = coords_to_cv_contour(coords['x'], coords['y'])
        overlap.append(contours_overlap(contour_to_add, parent_contour_cv, image_shape))
    if not np.any(overlap):
        return None
    return parent_contours[np.argmax(overlap)]


def contour_overlaps_with_existing_on_parent_level(contour, contours_on_same_level):
    """ Check if a contour overlaps with any existing contours on the same level. """
    for existing_contour in contours_on_same_level:
        if cv.contourArea(cv.intersectConvexConvex(contour, existing_contour)) > 0:
            return True
    return False


def contours_overlap(contour_1, contour_2, shape):
    """ Check if two contours overlap. """
    canvas_1 = np.zeros(shape, dtype=np.uint8)
    cv.fillPoly(canvas_1, [contour_1.astype(np.int32)], color=1)
    canvas_2 = np.zeros(shape, dtype=np.uint8)
    cv.fillPoly(canvas_2, [contour_2.astype(np.int32)], color=1)
    return np.sum(np.logical_and(canvas_1, canvas_2))


def coords_to_cv_contour(x_coords, y_coords):
    """ Convert coordinates to a cv2 contour. """
    return np.expand_dims(np.array(list(zip(x_coords, y_coords)), dtype=np.int32), 1)


def is_contour_addable(contour, label, mask_id):
    """ Check if a contour can be added to the mask. This means it does not overlap with any existing contours. and it
     is enclosed by its parent. """

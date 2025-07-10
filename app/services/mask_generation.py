import json

from app.database import get_context_session
from app.database.images import Images
from app.database.mask_generation import Contours, Masks
from app.database.datasets import Labels
import numpy as np
import cv2 as cv


def build_contour_hierarchy(contours, parent=None):
    """ Recursively build a list of contours, where the contours go from root to leaf. """
    contour_list = []
    for contour in contours:
        if contour.parent_id == parent:
            contour_list.append(contour)
            contour_list.extend(build_contour_hierarchy(contours, contour.id))
    return contour_list


def generate_mask(mask_id):
    """Generate a mask from the saved contours of that mask."""
    with get_context_session() as session:
        contours = session.query(Contours).filter_by(mask_id=mask_id).all()
        image_id = session.query(Masks).filter_by(id=mask_id).first().image_id
        image = session.query(Images).filter_by(id=image_id).first()
        labels = session.query(Labels).filter_by(dataset_id=image.dataset_id).all()
        label_id_to_value = {label.id: i + 1 for i, label in enumerate(labels)}
        print("Label map", label_id_to_value)
        canvas = np.zeros((image.height, image.width), dtype=np.uint8)
        contour_hierarchy = build_contour_hierarchy(contours)
        for contour in contour_hierarchy:
            print(f"Drawing contour {contour.id} with label {contour.label}")
            coords_dict = json.loads(contour.coords)
            x = coords_dict['x']
            y = coords_dict['y']
            cv_contour = np.expand_dims(np.array(list(zip(x, y))), 1)
            cv_contour[..., 0] *= image.width
            cv_contour[..., 1] *= image.height
            cv.fillPoly(canvas, [cv_contour.astype(np.int32)], color=[label_id_to_value[contour.label]])
        return canvas


def contour_is_enclosed_by_parent(contour, parent_contour):
    """ Check if a contour is enclosed by its parent contour. """
    if parent_contour is None:
        return True
    return np.all([cv.pointPolygonTest(parent_contour, (p[0, 1], p[0,0]), False) >= 0 for p in contour])


def contour_overlaps_with_existing_on_parent_level(contour, contours_on_same_level):
    """ Check if a contour overlaps with any existing contours on the same level. """
    for existing_contour in contours_on_same_level:
        if cv.contourArea(cv.intersectConvexConvex(contour, existing_contour)) > 0:
            return True
    return False


def coords_to_cv_contour(x_coords, y_coords):
    """ Convert coordinates to a cv2 contour. """
    return np.expand_dims(np.array(list(zip(x_coords, y_coords)), dtype=np.int8), 1)


def is_contour_addable(contour, label, mask_id):
    """ Check if a contour can be added to the mask. This means it does not overlap with any existing contours. and it
     is enclosed by its parent. """

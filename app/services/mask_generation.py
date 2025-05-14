from app.database import get_session
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
    with get_session() as session:
        contours = session.query(Contours).filter_by(mask_id=mask_id).all()
        image_id = session.query(Masks).filter_by(id=mask_id).first().image_id
        image = session.query(Images).filter_by(id=image_id).first()
        labels = session.query(Labels).filter_by(dataset_id=image.dataset_id).all()
        label_id_to_value = {label.id: label.value for label in labels}
        canvas = np.zeros((image.height, image.width), dtype=np.uint8)
        for contour in build_contour_hierarchy(contours):
            coords_dict = contour.coords
            x = coords_dict['x']
            y = coords_dict['y']
            cv_contour = np.expand_dims(np.array(list(zip(x, y)), dtype=np.int8), 1)
            cv.fillPoly(canvas, cv_contour, color=[label_id_to_value[contour.label]])
        return canvas

import json

from app.database import get_context_session
from app.database.images import Images
from app.database.masks import Masks
from app.database.contours import Contours
from app.database.labels import Labels
import numpy as np
import cv2 as cv

from app.services.contours import build_depth_first_contour_list


def generate_mask(mask_id):
    """ Generate a mask from the saved contours of that mask and return an array of the mask."""
    with get_context_session() as session:
        contours = session.query(Contours).filter_by(mask_id=mask_id).all()
        image_id = session.query(Masks).filter_by(id=mask_id).first().image_id
        image = session.query(Images).filter_by(id=image_id).first()
        labels = session.query(Labels).filter_by(dataset_id=image.dataset_id).all()
        label_id_to_value = {label.id: i + 1 for i, label in enumerate(labels)}
        print("Label map", label_id_to_value)
        canvas = np.zeros((image.height, image.width), dtype=np.uint8)
        contour_hierarchy = build_depth_first_contour_list(contours)
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

from app.database import get_session
from database.images import Images
from database.mask_generation import Contours, Masks
import numpy as np
import cv2 as cv


def generate_mask(mask_id):
    """Generate a mask from the saved contours of that mask."""
    with get_session() as session:
        contours = session.query(Contours).filter_by(mask_id=mask_id).all()
        image_id = session.query(Masks).filter_by(id=mask_id).first().image_id
        image = session.query(Images).filter_by(id=image_id).first()
        canvas = np.zeros((image.height, image.width), dtype=np.uint8)
        for contour in contours:
            coords_dict = contour.coords
            x = coords_dict['x']
            y = coords_dict['y']
            coords = np.array([x, y]).T
            cv.fillPoly(canvas, [coords], color=contour.label)

from logging import getLogger

import cv2
import numpy as np
from fastapi import HTTPException, status

from app.database.contours import Contours
from app.database.images import Images
from app.schemas.contours import Contour

logger = getLogger(__name__)


def get_contours_from_binary_mask(mask: np.ndarray,
                                  only_return_biggest=False,
                                  limit=None,
                                  added_by: str = "system",
                                  label_id: int = None,
                                  temporary: bool = True) -> list[Contour]:
    """ Get contour models from a binary mask
    :param mask: A binary mask in the form of a numpy array
    :param only_return_biggest: If true, only return the biggest contour.
    :param limit: Number of contours to return. If None, return all contours.
    :param added_by: Author of this contour, by default "system".
    :param label_id: Contour label id. If None, no label is given to the contour.
    :param temporary: Sets the contour to temporary.
    :return: List of contour models
    """
    logger.debug("Computing contours for mask.")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:  # check if any contours found
        if only_return_biggest:
            contours = [max(contours, key=cv2.contourArea)]
        else:
            contours = sorted(contours, key=cv2.contourArea)
            if limit is not None and len(contours) > limit:
                logger.warning(f"Detected over {limit} objects. Only returning the biggest 500 objects.")
                contours = contours[:limit]
        models = []
        for contour in contours:
            # First dim of contour is x, but first dim of mask is height, so it needs to be switched!
            contour[..., 0] /= mask.shape[1]
            contour[..., 1] /= mask.shape[0]
            models.append(Contour.from_normalized_cv_contour(contour,
                                                             label=label_id,
                                                             added_by=added_by,
                                                             temporary=temporary)
                          )
        return models
    else:
        return np.array([])


def contour_ids_to_indices(image_id, contour_ids, db):
    height, width = db.query(Images.height, Images.width).filter_by(id=image_id).first()
    seeds = []
    label = None
    for contour_id in contour_ids:
        contour_db = db.query(Contours).filter_by(id=contour_id).first()
        if contour_db is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Contour with id {contour_id} not found! "
                                       f"Completion request failed!")
        contour_model = Contour.from_db(contour_db)
        if label is None:
            label = contour_model.label_id
        else:
            if not label == contour_model.label_id:
                raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                                    detail="You are trying to run completion on contours of different labels. This is "
                                           "not allowed!")
        seeds.append(np.argwhere(contour_model.to_binary_mask(height, width)).flatten())

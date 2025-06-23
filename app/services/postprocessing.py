import json

import cv2 as cv
import numpy as np
from app.database import get_context_session
from app.database.mask_generation import Contours, Masks
from app.services.contours import get_contour_from_coordinates, create_binary_mask_from_contours
from app.services.database_access import get_height_width_of_image
from logging import getLogger


logger = getLogger(__name__)


def postprocess_binary_mask(mask: np.ndarray):
    """Post-process a binary mask by removing small objects and filling holes."""
    # Ensure mask is binary
    if np.unique(mask).size > 2:
        raise ValueError("Input mask is not binary. Postprocessing requires a binary mask.")

    # Fill holes via Closing
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Add more methods here

    return mask


def fit_mask_to_already_created_masks(mask_id: int, mask: np.ndarray,
                                      parent_contour_id : int = None) -> np.ndarray:
    """Ensure that the contours of a label are within the bounds of the mask.
       Args:
           mask_id (int): The mask ID to check.
           mask (np.ndarray): The binary mask to check against.
           parent_contour_id (int, optional): The parent contour ID to check against. Defaults to None.
       Returns:
           list[np.ndarray]: The filtered list of contours that are within the bounds of the mask.
    """
    if not np.any(mask):
        logger.warning("Input mask is empty! Returning empty mask.")
        return np.zeros_like(mask, dtype=np.uint8)

    with get_context_session() as session:
        mask_db = session.query(Masks).filter_by(id=mask_id).first()
        height, width = get_height_width_of_image(mask_db.image_id)
        contours_on_same_level = session.query(Contours).filter_by(mask_id=mask_id, parent_id=parent_contour_id).all()
        parent_contour = session.query(Contours).filter_by(id=parent_contour_id).first() if parent_contour_id else None
        if parent_contour:
            coords = json.loads(parent_contour.coords)
            parent_contour = get_contour_from_coordinates(coords["x"], coords["y"], height, width)
        contours = []
        for contour in contours_on_same_level:
            coords = json.loads(contour.coords)
            contours.append(get_contour_from_coordinates(coords["x"], coords["y"], height, width))

        if parent_contour:
            positive_mask = create_binary_mask_from_contours(width, height, [parent_contour])
        else:
            positive_mask = np.ones((height, width), dtype=np.uint8)
        negative_mask = create_binary_mask_from_contours(width, height, contours)

        if not np.any(positive_mask):
            logger.warning("No positive mask found! Returning empty mask.")
            return np.zeros_like(mask, dtype=np.uint8)

        # Fit the entire mask to the parent masks. Pixels outside the parent are not allowed.
        on_parent_mask = np.logical_and(positive_mask, mask).astype(np.uint8)

        if not np.any(on_parent_mask):
            logger.warning("Predicted mask does not overlap with parent mask! Returning empty mask.")
            return np.zeros_like(mask, dtype=np.uint8)

        # Remove pixels that are already in the negative mask. This means the new mask overlaps with already existing
        # masks, which is not allowed.
        final_mask = np.logical_and(np.logical_not(negative_mask), on_parent_mask).astype(np.uint8)

        if not np.any(final_mask):
            logger.warning("Predicted mask overlaps completely with existing masks! Returning empty mask.")
        return final_mask

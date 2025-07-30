import json

import cv2 as cv
import numpy as np
from app.database import get_context_session
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.contours import Contours
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


def fit_mask_to_already_created_masks(mask_id: int,
                                      mask: np.ndarray,
                                      label_id: int,
                                      parent_contour_id : int = None):
    """
    Ensure that the contours of a label are within the bounds of the mask.
       Args:
           mask_id (int): The mask ID to check.
           mask (np.ndarray): The binary mask to check against.
           label_id (int): The label ID to check against.
           parent_contour_id (int, optional): The parent contour ID to check against. Defaults to None.
       Returns:
             dict: A dictionary containing the success status, message, and the final mask.
    """
    if not np.any(mask):
        logger.warning("Input mask is empty! Returning empty mask.")
        return {
            "success": False,
            "message": "Input mask is empty, cannot fit mask to existing contours."
        }

    with get_context_session() as session:
        mask_db = session.query(Masks).filter_by(id=mask_id).first()
        height, width = get_height_width_of_image(mask_db.image_id)

        # Get the parent label of the current label
        parent_label_id = session.query(Labels.parent_id).filter_by(id=label_id).first()
        # Get all labels that have the same parent label, our new mask cannot overlap with any of them.
        labels_on_same_level = session.query(Labels.id).filter_by(parent_id=parent_label_id[0]).all()
        labels_on_same_level = [label[0] for label in labels_on_same_level]  # Extract IDs from tuples
        contours_on_same_level = session.query(Contours).filter(Contours.mask_id == mask_id,
                                                                Contours.label.in_(labels_on_same_level)).all()
        parent_contour = session.query(Contours).filter_by(id=parent_contour_id).first() if parent_contour_id else None
        logger.debug(f"Fitting mask to parent contour {parent_contour_id} and "
                     f"{len(contours_on_same_level)} contours on the same level.")
    if parent_contour is not None:
        coords = json.loads(parent_contour.coords)
        parent_contour = get_contour_from_coordinates(coords["x"], coords["y"], height, width)
        positive_mask = create_binary_mask_from_contours(width, height, [parent_contour])
    else:
        positive_mask = np.ones((height, width), dtype=np.uint8)

    if not np.any(positive_mask):
        logger.warning("No positive mask found! Returning empty mask.")
        return {
            "success": False,
            "message": "Parent contour is empty, cannot fit mask to existing contours."
        }

    # Fit the entire mask to the parent masks. Pixels outside the parent are not allowed.
    on_parent_mask = np.logical_and(positive_mask, mask).astype(np.uint8)
    if not np.any(on_parent_mask):
        logger.warning("Predicted mask does not overlap with parent mask! Returning empty mask.")
        return {
            "success": False,
            "message": "Predicted mask does not overlap with parent mask. Maybe you selected "
                       "the wrong parent contour?"
        }

    contours = []
    for contour in contours_on_same_level:
        coords = json.loads(contour.coords)
        contours.append(get_contour_from_coordinates(coords["x"], coords["y"], height, width))
    negative_mask = create_binary_mask_from_contours(width, height, contours)

    # Remove pixels that are already in the negative mask. This means the new mask overlaps with already existing
    # masks, which is not allowed.
    final_mask = np.logical_and(np.logical_not(negative_mask), on_parent_mask).astype(np.uint8)

    if not np.any(final_mask):
        logger.warning("Predicted mask overlaps completely with existing masks! Returning empty mask.")
        return {
            "success": False,
            "message": "Predicted mask overlaps completely with existing masks on the same level. "
                       "Maybe you already annotated this object?"
        }
    if np.all(mask == final_mask):
        logger.info("Predicted mask is identical to the final mask, no changes made.")
        return {
            "success": True,
            "message": "Added mask completely.",
            "mask": final_mask
        }
    else:
        return {
            "success": True,
            "message": "Mask was fitted to existing contours successfully.",
            "mask": final_mask
        }

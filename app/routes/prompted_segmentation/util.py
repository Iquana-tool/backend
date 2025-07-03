import numpy as np
from app.services.contours import get_contours
from app.schemas.segmentation.contours_and_quantifications import ContourModel
from app.schemas.segmentation.segmentations import SegmentationMaskModel


async def get_masks_responses(masks, qualities):
    masks_response = []
    for mask, quality in zip(masks, qualities):
        # Get contours of the postprocessed mask if postprocessing is enabled
        # Postprocessing might improve performance by removing noise
        contours_response = []
        unique_labels = np.unique(mask)
        for label in unique_labels:
            if label == 0:
                continue
            # Extract the mask for the current label
            mask_label = (mask == label).astype(np.uint8)
            contours = get_contours(mask_label)
            contours_response += get_contour_models(contours, label, mask_label.shape[0], mask_label.shape[1])
        masks_response.append(SegmentationMaskModel(contours=contours_response, predicted_iou=quality))
    return masks_response


def get_contour_models(contours, label, height, width):
    """ Convert contours to ContourModel objects. """
    contour_models = []
    for contour in contours:
        if len(contour) < 3:
            # Skip contours with less than 3 points
            continue
        x_coords = contour[..., 0].flatten() / width  # Normalize x-coordinates
        y_coords = contour[..., 1].flatten() / height  # Normalize y-coordinates
        contour_models.append(ContourModel(
            x=x_coords.tolist(),
            y=y_coords.tolist(),
            label=label
        ))
    return contour_models

import numpy as np
from app.services.contours import get_contours_from_binary_mask
from app.services.postprocessing import postprocess_binary_mask
from app.schemas.contours import Contour
from app.schemas.prompted_segmentation.segmentations import SemanticSegmentationMask


async def convert_numpy_masks_to_segmentation_mask_models(masks, qualities, label_map: dict = None, only_return_one: bool = True) \
        -> list[SemanticSegmentationMask]:
    """ Convert masks to SegmentationMaskModel objects with contours. Internally, it separates masks by labels and
    extracts contours for each label.

    Args:
        masks (list of np.ndarray): List of masks, each mask is a 2D numpy array with integer labels.
        qualities (list of float): List of quality scores for each mask.
        label_map (dict, optional): Mapping from original labels to new labels. If None, uses original labels.
        only_return_one (bool): If True, returns only one contour per label. If False, returns all contours. This is used
                    for prompted prompted_segmentation where we want to return only one contour per label.

    Returns:
        List[SegmentationMaskModel]: List of SegmentationMaskModel objects, each containing contours and predicted IoU."""
    masks_response = []
    for mask, quality in zip(masks, qualities):
        # Get contours of the postprocessed mask if postprocessing is enabled
        # Postprocessing might improve performance by removing noise
        contours_response = []
        unique_labels = np.unique(mask)
        for label in unique_labels:
            if label == 0:
                # Skip the background label (usually 0)
                continue
            # Extract the mask for the current label
            mask_label = postprocess_binary_mask((mask == label).astype(np.uint8))
            contours = get_contours_from_binary_mask(mask_label, only_return_biggest=only_return_one)
            contours_response += get_contour_models(contours,
                                                    label if not label_map else label_map[label],
                                                    mask_label.shape[0],
                                                    mask_label.shape[1])
        masks_response.append(SemanticSegmentationMask(contours=contours_response, confidence=quality))
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
        contour_models.append(Contour(
            x=x_coords.tolist(),
            y=y_coords.tolist(),
            label=label
        ))
    return contour_models

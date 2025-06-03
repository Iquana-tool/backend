import logging

import numpy as np
from fastapi import APIRouter

from app.schemas.segmentation.segmentations import PromptedSegmentationRequest, SegmentationMaskModel, SegmentationResponse, \
    AutomaticSegmentationRequest
from app.schemas.segmentation.contours_and_quantifications import ContourModel
from app.services.contours import get_contours
from app.services.database_access import get_height_width_of_image
from app.services.postprocessing import postprocess_binary_mask

from app.services.segmentation import MockupSegmentationModel, ModelCache
from app.services.segmentation.sam2 import SAM2Prompted, SAM2Automatic

from config import SAM2TinyConfig, SAM2SmallConfig, SAM2LargeConfig, SAM2BasePlusConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/segmentation", tags=["segmentation"])


class PromptedSegmentationModelsConfig:
    """ This class contains the configuration options for the model. """
    selected_model = 'SAM2Tiny'
    available_models = {
        'Mockup': MockupSegmentationModel,
        'SAM2Tiny': SAM2Prompted.set_model_config(SAM2TinyConfig),
        'SAM2Small': SAM2Prompted.set_model_config(SAM2SmallConfig),
        'SAM2Large': SAM2Prompted.set_model_config(SAM2LargeConfig),
        'SAM2BasePlus': SAM2Prompted.set_model_config(SAM2BasePlusConfig)
    }


prompted_model_cache = ModelCache(PromptedSegmentationModelsConfig.available_models)


class AutomaticSegmentationModelsConfig:
    """ This class contains the configuration options for the semantic segmentation model. """
    selected_model = 'SAM2Tiny'
    available_models = {
        'Mockup': MockupSegmentationModel,
        'SAM2Tiny': SAM2Prompted.set_model_config(SAM2TinyConfig),
        'SAM2Small': SAM2Prompted.set_model_config(SAM2SmallConfig),
        'SAM2Large': SAM2Prompted.set_model_config(SAM2LargeConfig),
        'SAM2BasePlus': SAM2Prompted.set_model_config(SAM2BasePlusConfig)
    }


automatic_model_cache = ModelCache(AutomaticSegmentationModelsConfig.available_models)


@router.post('/segment_image')
async def segment_image(request: PromptedSegmentationRequest):
    """Perform segmentation with optional prompts, using data validation.
    This function handles the segmentation of images based on the provided request.
    It validates the request, retrieves the appropriate model, and processes the image.

    Args:
        request (PromptedSegmentationRequest): The request object containing image data and parameters. When using cropping,
        make sure to remap the annotation coordinates to the cropped image.

    Returns:
        SegmentationResponse: The response object containing the segmentation results. When using cropping,
        the contours will be remapped to the original image size.
    """
    # Get the model based on the identifier
    model = prompted_model_cache.set_and_get_model(request.model)
    logger.debug(f"Using model: {model.model_name}")

    # Process the request with the model
    # This method should handle the image preprocessing and segmentation
    # All model specific logic should be encapsulated in the model class
    masks, quality = model.process_prompted_request(request)
    logger.debug(f"Segmentation completed for image_id: {request.image_id} with {len(masks)} masks.")

    # Postprocess the masks and get contours
    height, width = get_height_width_of_image(request.image_id)
    masks_response = []
    for mask, quality in zip(masks, quality):
        # Get contours of the postprocessed mask if postprocessing is enabled
        # Postprocessing might improve performance by removing noise
        contours = get_contours(postprocess_binary_mask(mask) if request.apply_post_processing else mask)
        contours_response = get_contour_models(contours, request.label)
        masks_response.append(SegmentationMaskModel(contours=contours_response, predicted_iou=quality))
    return SegmentationResponse(masks=masks_response, image_id=request.image_id, model=request.model)


@router.post('/generate_mask')
async def generate_mask(request: AutomaticSegmentationRequest):
    """ Generate segmentation masks for an image using automatic semantic segmentation. """
    # Get the model based on the identifier
    model = automatic_model_cache.set_and_get_model(request.model)
    logger.debug(f"Using model: {model.model_name}")
    # Process the request with the model
    masks, quality = model.process_automatic_request(request)
    height, width = get_height_width_of_image(request.image_id)
    masks_response = []
    for mask, quality in zip(masks, quality):
        # Get contours of the postprocessed mask if postprocessing is enabled
        # Postprocessing might improve performance by removing noise
        contours_response = []
        unique_labels = np.unique(mask)
        for label in unique_labels:
            if label == 0:
                continue
            # Extract the mask for the current label
            mask_label = (mask == label).astype(np.uint8)
            contours = get_contours(postprocess_binary_mask(mask) if request.apply_post_processing else mask)
            contours_response += get_contour_models(contours, label)
        masks_response.append(SegmentationMaskModel(contours=contours_response, predicted_iou=quality))
    return SegmentationResponse(
        masks=masks_response, image_id=request.image_id, model=request.model
    )


def get_contour_models(contours, label):
    """ Convert contours to ContourModel objects. """
    contour_models = []
    for contour in contours:
        if len(contour) < 3:
            # Skip contours with less than 3 points
            continue
        x_coords = contour[..., 0].flatten()
        y_coords = contour[..., 1].flatten()
        contour_models.append(ContourModel(
            x=x_coords.tolist(),
            y=y_coords.tolist(),
            label=label
        ))
    return contour_models

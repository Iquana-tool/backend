import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

import config
from app.database import get_session
from app.schemas.segmentation_and_masks import (
    PromptedSegmentationRequest, SegmentationResponse, ContourModel,
    SegmentationMaskModel, QuantificationsModel, AutomaticSegmentationRequest
)
from app.services.segmentation import ModelCache
from app.services.contours import get_contours
from app.services.quantifications import ContourQuantifier
from app.services.database_access import get_height_width_of_image
from app.services.postprocessing import postprocess_binary_mask

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/segmentation", tags=["segmentation"])
prompted_model_cache = ModelCache(config.PromptedSegmentationModelsConfig.available_models)
automatic_model_cache = ModelCache(config.AutomaticSegmentationModelsConfig.available_models)


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
        contours_response = []
        for contour in contours:
            if len(contour) < 3:
                # Skip contours with less than 3 points
                continue
            x_coords = (contour[..., 0].flatten() + int(request.min_x * width)) / width
            y_coords = (contour[..., 1].flatten() + int(request.min_y * height)) / height
            contours_response.append(ContourModel(
                # We have to rescale the images to the original size
                x=x_coords,
                y=y_coords,
                label=request.label
            ))
        masks_response.append(SegmentationMaskModel(contours=contours_response, predicted_iou=quality))
    return SegmentationResponse(masks=masks_response, image_id=request.image_id, model=request.model)


@router.post('/generate_mask')
async def generate_mask(request: AutomaticSegmentationRequest):
    """ Generate segmentation masks for an image using automatic semantic segmentation. """


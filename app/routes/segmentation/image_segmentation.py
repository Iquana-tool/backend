from logging import getLogger
from app.services.logging import log_execution_time

import numpy as np
from fastapi import APIRouter

from app.schemas.segmentation.segmentations import PromptedSegmentationRequest, SegmentationMaskModel, SegmentationResponse, \
    AutomaticSegmentationRequest
from app.services.segmentation import ModelCache
from app.routes.segmentation.util import get_masks_responses


logger = getLogger(__name__)
router = APIRouter(prefix="/segmentation", tags=["segmentation"])


prompted_model_cache = ModelCache()
automatic_model_cache = ModelCache()


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

    # Process the request with the model
    # This method should handle the image preprocessing and segmentation
    # All model specific logic should be encapsulated in the model class
    masks, quality = model.process_prompted_request(request)

    # Postprocess the masks and get contours
    masks_response = get_masks_responses(masks, quality)
    return SegmentationResponse(masks=masks_response, image_id=request.image_id, model=request.model)


@router.post('/generate_mask')
async def generate_mask(request: AutomaticSegmentationRequest):
    """ Generate segmentation masks for an image using automatic semantic segmentation. """
    # Get the model based on the identifier
    model = automatic_model_cache.set_and_get_model(request.model)
    logger.debug(f"Using model: {model.model_name}")
    # Process the request with the model
    masks, qualities = model.process_automatic_request(request)
    masks_response = get_masks_responses(masks, qualities)
    return SegmentationResponse(
        masks=masks_response, image_id=request.image_id, model=request.model
    )

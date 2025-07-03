from logging import getLogger
from app.services.labels import label_value_to_label_id
import numpy as np
from fastapi import APIRouter, Depends

from app.schemas.segmentation.segmentations import PromptedSegmentationRequest, SegmentationMaskModel, SegmentationResponse, \
    AutomaticSegmentationRequest
from app.services.segmentation import ModelCache
from app.services.postprocessing import fit_mask_to_already_created_masks
from app.routes.prompted_segmentation.util import get_masks_responses
from app.database import get_session
from app.database.images import Images
from sqlalchemy.orm import Session


logger = getLogger(__name__)
router = APIRouter(prefix="/prompted_segmentation", tags=["prompted_segmentation"])


prompted_model_cache = ModelCache()
automatic_model_cache = ModelCache()


@router.post('/segment_image')
async def segment_image(request: PromptedSegmentationRequest):
    """Perform prompted_segmentation with optional prompts, using data validation.
    This function handles the prompted_segmentation of images based on the provided request.
    It validates the request, retrieves the appropriate model, and processes the image.

    Args:
        request (PromptedSegmentationRequest): The request object containing image data and parameters. When using cropping,
        make sure to remap the annotation coordinates to the cropped image.

    Returns:
        SegmentationResponse: The response object containing the prompted_segmentation results. When using cropping,
        the contours will be remapped to the original image size.
    """
    # Get the model based on the identifier
    model = prompted_model_cache.set_and_get_model(request.model)

    # Process the request with the model
    # This method should handle the image preprocessing and prompted_segmentation
    # All model specific logic should be encapsulated in the model class
    masks, quality = model.process_prompted_request(request)

    # Postprocess the masks. This fits the mask into the already existing contours.
    if request.mask_id:
        masks = [mask * request.label for mask in masks]
        masks = [fit_mask_to_already_created_masks(request.mask_id, mask, request.parent_contour_id) for mask in masks]
    else:
        # If no mask_id is provided, we cannot correct the masks.
        logger.warning("No mask_id provided, skipping mask fitting to existing contours.")
    masks_response = get_masks_responses([mask * request.label for mask in masks], quality)
    return SegmentationResponse(masks=masks_response, image_id=request.image_id, model=request.model)


@router.post('/generate_mask')
async def generate_mask(request: AutomaticSegmentationRequest, db: Session = Depends(get_session)):
    """ Generate prompted_segmentation masks for an image using automatic semantic prompted_segmentation. """

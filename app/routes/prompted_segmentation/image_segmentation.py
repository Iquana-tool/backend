from logging import getLogger

from app.database.masks import Masks
from app.services.labels import label_value_to_label_id
import numpy as np
from fastapi import APIRouter, Depends

from app.schemas.segmentation.segmentations import PromptedSegmentationRequest, SegmentationMaskModel, \
    SegmentationResponse, \
    AutomaticSegmentationRequest
from app.services.prompted_segmentation import ModelCache
from app.services.postprocessing import fit_mask_to_already_created_masks
from app.routes.prompted_segmentation.util import get_masks_responses
from app.database import get_session, get_context_session
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
    The predicted contour will be fit to existing contours if a mask_id is provided in the request.
    This means that a contour is always contained in its parent contour and has no overlap with other contours on
    the same level.

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
    if len(masks) > 1:
        logger.warning("This should only return one mask, but got multiple masks. Dropping all but the first mask.")
    mask = masks[0]
    quality = quality[0]

    # Postprocess the masks. This fits the mask into the already existing contours.
    with get_context_session() as session:
        mask_id = session.query(Masks.id).filter_by(image_id=request.image_id).first()
    if mask_id:
        response = fit_mask_to_already_created_masks(request.mask_id,
                                                 mask,
                                                 request.label,
                                                 request.parent_contour_id)
        if not response["success"]:
            return response
        else:
            masks = [response["mask"]]
    else:
        # If no mask_id is provided, we cannot correct the masks.
        logger.warning("No mask_id provided, skipping mask fitting to existing contours.")
        response = {"success": True, "message": "No existing contours to fit the mask to."}
    masks_response = await get_masks_responses([mask * request.label], [quality])
    return {
        "success": True,
        "message": "Prompted segmentation completed successfully. " + response["message"],
        "response": SegmentationResponse(masks=masks_response, image_id=request.image_id, model=request.model)
    }


@router.post('/generate_mask')
async def generate_mask(request: AutomaticSegmentationRequest, db: Session = Depends(get_session)):
    """ Generate prompted_segmentation masks for an image using automatic semantic prompted_segmentation. """
    pass

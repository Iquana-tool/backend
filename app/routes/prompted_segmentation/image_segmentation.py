import json
from logging import getLogger

import cv2

from app.database.contours import Contours
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
import app.services.ai_services.prompted_segmentation as prompted_service

logger = getLogger(__name__)
router = APIRouter(prefix="/prompted_segmentation", tags=["prompted_segmentation"])

prompted_model_cache = ModelCache()
automatic_model_cache = ModelCache()


@router.get("/health")
async def health_check():
    """Health check endpoint to verify if the prompted segmentation backend is reachable."""
    if await prompted_service.check_backend():
        return {
            "success": True,
            "message": "Prompted segmentation backend is reachable.",
            "response": None
        }
    else:
        return {
            "success": False,
            "message": "Prompted segmentation backend is not reachable. Please make sure it is running.",
            "response": None
        }


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
    # Check if the backend is running
    if not await prompted_service.check_backend():
        return {
            "success": False,
            "message": "Prompted segmentation backend is not reachable. Please make sure it is running.",
            "response": None
        }
    logger.debug("Prompted segmentation backend is reachable.")

    # First set the image in the model cache
    await prompted_service.upload_image("test", request.image_id)
    logger.debug(f"Image {request.image_id} uploaded to prompted segmentation backend.")

    # Then, if a crop is provided, focus the model on the crop
    use_crop = request.parent_contour_id is not None
    if use_crop:
        with get_context_session() as session:
            contour = session.query(Contours).filter_by(id=request.parent_contour_id).first()
            coords = json.loads(contour.coords)
            min_x = min(coords.x)
            min_y = min(coords.y)
            max_x = max(coords.x)
            max_y = max(coords.y)
            await prompted_service.focus_crop("test",
                                                min_x,
                                                min_y,
                                                max_x,
                                                max_y)
        logger.debug(f"Image cropped to contour {request.parent_contour_id} for prompted segmentation.")
    else:
        await prompted_service.unfocus_crop("test")
        logger.debug("Image uncropped for prompted segmentation.")

    # Now segment the image
    response = await prompted_service.segment_image_with_prompts(
        "test",
        request.model,
        request.prompts,
    )
    logger.debug("Prompted segmentation successful.")

    mask = response["mask"]
    score = response["score"]

    cv2.imwrite("debug_mask.png", mask * 255)
    # Convert to response object
    masks_response = await get_masks_responses([mask * request.label], [score])
    return {
        "success": True,
        "message": "Prompted segmentation completed successfully.",
        "response": SegmentationResponse(masks=masks_response, image_id=request.image_id, model=request.model)
    }

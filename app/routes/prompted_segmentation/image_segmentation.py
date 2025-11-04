from logging import getLogger

import cv2
from fastapi import APIRouter

import app.services.ai_services.prompted_segmentation as prompted_service
from app.database import get_context_session
from app.database.contours import Contours
from app.routes.prompted_segmentation.util import convert_numpy_masks_to_segmentation_mask_models
from app.schemas.contours import Contour
from app.schemas.prompted_segmentation.segmentations import PromptedSegmentationHTTPRequest, SegmentationResponse, \
    PromptedSegmentationWebsocketRequest

logger = getLogger(__name__)
router = APIRouter(prefix="/prompted_segmentation", tags=["prompted_segmentation"])


@router.get("/health")
async def health_check():
    """Health check endpoint to verify if the prompted prompted_segmentation backend is reachable."""
    if await prompted_service.check_backend():
        return {
            "success": True,
            "message": "Prompted prompted_segmentation backend is reachable.",
            "response": None
        }
    else:
        return {
            "success": False,
            "message": "Prompted prompted_segmentation backend is not reachable. Please make sure it is running.",
            "response": None
        }


@router.get("/models")
async def get_available_models():
    """Retrieve the list of available prompted segmentation models from the backend."""
    models = await prompted_service.get_available_models()
    return {
        "success": True,
        "message": "Retrieved available prompted segmentation models.",
        "response": models
    }


@router.post('/segment_image')
async def segment_image(request: PromptedSegmentationHTTPRequest):
    """Perform prompted_segmentation with optional prompts, using data validation.
    This function handles the prompted_segmentation of images based on the provided request.
    It validates the request, retrieves the appropriate model, and processes the image.
    The predicted contour will be fit to existing contours if a mask_id is provided in the request.
    This means that a contour is always contained in its parent contour and has no overlap with other contours on
    the same level.

    Args:
        request (PromptedSegmentationHTTPRequest): The request object containing image data and parameters. When using cropping,
        make sure to remap the annotation coordinates to the cropped image.

    Returns:
        SegmentationResponse: The response object containing the prompted_segmentation results. When using cropping,
        the contours will be remapped to the original image size.
    """
    user_id = "HTTP_Request"
    # Check if the backend is running
    if not await prompted_service.check_backend():
        return {
            "success": False,
            "message": "Prompted prompted_segmentation backend is not reachable. Please make sure it is running.",
            "response": None
        }
    logger.debug("Prompted prompted_segmentation backend is reachable.")

    # First set the image in the model cache
    await prompted_service.upload_image(user_id, request.image_id)
    logger.debug(f"Image {request.image_id} uploaded to prompted segmentation backend.")

    # Then check if we refine a contour
    if request.refine_contour_id is not None:
        # Get the contour to refine
        with get_context_session() as session:
            contour = session.query(Contours).filter_by(id=request.refine_contour_id).first()
            contour_model = Contour.from_db(contour)
        previous_mask = contour_model.to_binary_mask(1000, 1000)
        logger.debug(f"Using contour {request.refine_contour_id} as previous mask for refinement.")
    else:
        previous_mask = None

    # Then, if a crop is provided, focus the model on the crop
    use_crop = request.parent_contour_id is not None
    if use_crop:
        with get_context_session() as session:
            contour = session.query(Contours).filter_by(id=request.parent_contour_id).first()
            contour_model = Contour.from_db(contour)
            min_x = min(contour_model.x)
            min_y = min(contour_model.y)
            max_x = max(contour_model.x)
            max_y = max(contour_model.y)
            await prompted_service.focus_crop(user_id,
                                                min_x,
                                                min_y,
                                                max_x,
                                                max_y)
            if previous_mask is not None:
                # Crop the previous mask as well
                previous_mask = previous_mask[min_y:max_y, min_x:max_x]
        logger.debug(f"Image cropped to contour {request.parent_contour_id} for prompted prompted_segmentation.")
    else:
        await prompted_service.unfocus_crop(user_id)
        logger.debug("Image uncropped for prompted prompted_segmentation.")

    # Now segment the image
    response = await prompted_service.segment_image_with_prompts(
        PromptedSegmentationWebsocketRequest(
            user_id=user_id,
            model_identifier=request.model_identifier,
            prompts=request.prompts,
            previous_mask=previous_mask.tolist() if previous_mask is not None else None
        )
    )
    logger.debug("Prompted prompted_segmentation successful.")

    mask = response["mask"]
    score = response["score"]

    cv2.imwrite("debug_mask.png", mask * 255)
    # Convert to response object
    masks_response = await convert_numpy_masks_to_segmentation_mask_models([mask * request.label], [score])
    return {
        "success": True,
        "message": "Prompted prompted_segmentation completed successfully.",
        "response": SegmentationResponse(masks=masks_response, image_id=request.image_id, model=request.model_registry_key)
    }

from logging import getLogger

from fastapi import APIRouter, Depends

from app.database import get_context_session
from app.database.contours import Contours
from app.schemas.contours import Contour
from app.schemas.prompted_segmentation.segmentations import PromptedSegmentationHTTPRequest, \
    PromptedSegmentationWebsocketRequest, SegmentationResponse
from app.schemas.user import User
from app.services.ai_services.prompted_segmentation import PromptedSegmentationService
from app.services.auth import get_current_user

logger = getLogger(__name__)
router = APIRouter(prefix="/prompted_segmentation", tags=["prompted_segmentation"])


@router.get("/health")
async def health_check(user: User = Depends(get_current_user)):
    """Health check endpoint to verify if the prompted prompted_segmentation backend is reachable."""
    if await PromptedSegmentationService().check_backend():
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
async def get_available_models(user: User = Depends(get_current_user)):
    """Retrieve the list of available prompted segmentation models from the backend."""
    models = await PromptedSegmentationService().get_models()
    return {
        "success": True,
        "message": "Retrieved available prompted segmentation models.",
        "response": models
    }


@router.post('/segment_image', deprecated=True)
async def segment_image(request: PromptedSegmentationHTTPRequest,
                        user: User = Depends(get_current_user)):
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
    raise DeprecationWarning("This endpoint has been deprecated. Please use the image annotation session websocket "
                             "instead.")

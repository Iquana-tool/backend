from logging import getLogger

from fastapi import APIRouter, Depends
from iquana_toolbox.schemas.user import User
from app.services.ai_services.prompted_segmentation import PromptedSegmentationService
from app.services.auth import get_current_user
from app.services.model_registry import list_available_models

logger = getLogger(__name__)
router = APIRouter(prefix="/prompted_segmentation", tags=["prompted_segmentation"])
service = PromptedSegmentationService()


@router.get("/health")
async def health_check(user: User = Depends(get_current_user)):
    """Health check endpoint to verify if the prompted prompted_segmentation backend is reachable."""
    if await service.check_backend():
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
    """Retrieve available prompted segmentation models directly from MLflow."""
    return list_available_models("prompted-segmentation")


# The POST /run endpoint was removed. It had already been deprecated in favour of
# the annotation-session WebSocket, but raised DeprecationWarning from inside the
# handler, so calling it produced a 500 and a stack trace rather than a clean
# refusal. Prompted segmentation runs through the WebSocket, which resolves the
# image path server-side from the image id.

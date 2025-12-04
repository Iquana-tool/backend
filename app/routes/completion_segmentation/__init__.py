from fastapi import APIRouter, Depends
from app.services.ai_services import completion_segmentation as completion_service
from app.schemas.user import User
from app.services.auth import get_current_user

router = APIRouter(prefix="/completion_segmentation", tags=["completion_segmentation"])


@router.get("/health")
async def health_check(user: User = Depends(get_current_user)):
    """Health check endpoint to verify if the prompted prompted_segmentation backend is reachable."""
    if await completion_service.check_backend():
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

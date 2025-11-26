from fastapi import Depends

from app.schemas.user import User
from app.services.ai_services import completion_segmentation as completion_service
from app.routes.completion_segmentation import router
from app.services.auth import get_current_user


@router.get("/models")
async def get_available_models(user: User = Depends(get_current_user)):
    """Retrieve the list of available prompted segmentation models from the backend."""
    models = await completion_service.get_models()
    return {
        "success": True,
        "message": "Retrieved available prompted segmentation models.",
        "response": models
    }

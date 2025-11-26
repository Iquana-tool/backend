from fastapi import Depends

from app.schemas.user import User
from app.services.ai_services import completion_segmentation as completion_service
from app.routes.completion_segmentation import router
from app.services.auth import get_current_user


@router.post("/upload_image")
async def upload_image(image_id: int, user: User = Depends(get_current_user)):
    await completion_service.upload_image(user.username, image_id)
    return {
        "success": True,
        "message": "Image upload successful"
    }
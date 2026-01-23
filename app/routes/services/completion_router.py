from fastapi import APIRouter, Depends
from schemas.service_requests import CompletionRequest
from schemas.contours import Contour
from schemas.user import User
from sqlalchemy.orm import Session

from app.database import get_session
from app.services.ai_services.completion_segmentation import CompletionService
from app.services.auth import get_current_user

completion_service = CompletionService()
router = APIRouter(prefix="/completion", tags=["Completion Segmentation"])


@router.get("/models")
async def get_available_models(user: User = Depends(get_current_user)):
    """Retrieve the list of available prompted segmentation models from the backend."""
    response = await completion_service.get_models()
    return {
        "success": True,
        "message": "Retrieved available prompted segmentation models.",
        "response": response
    }


@router.post("/upload_image")
async def upload_image(image_id: int, user: User = Depends(get_current_user)):
    await completion_service.upload_image(user.username, image_id)
    return {
        "success": True,
        "message": "Image upload successful"
    }


@router.post("/run")
async def infer_completion(
        request: CompletionRequest,
        user: User = Depends(get_current_user),
        db: Session = Depends(get_session),
):
    # Finally add the result to the db
    return await completion_service.inference(request)


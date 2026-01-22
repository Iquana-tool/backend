from fastapi import APIRouter, Depends
from schemas.completion_segmentation.inference import CompletionMainAPIRequest, CompletionServiceRequest
from schemas.contours import Contour
from schemas.user import User
from sqlalchemy.orm import Session

from app.database import get_session
from app.services.ai_services.completion_segmentation import CompletionService
from app.services.auth import get_current_user

completion_service = CompletionService()
router = APIRouter(prefix="/completion_segmentation", tags=["Completion Segmentation"])


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


@router.post("/inference/image_id={image_id}")
async def infer_completion(
        request: CompletionMainAPIRequest,
        user: User = Depends(get_current_user),
        db: Session = Depends(get_session),
):
    # First upload the image
    await completion_service.upload_image(user.username, request.image_id)

    contours = [Contour.from_id(contour_id).model_dump(include=["x", "y"]) for contour_id in request.seed_contour_ids]

    service_request = CompletionServiceRequest(
        model_key=request.model_key,
        user_id=user.username,
        contours=contours,
    )
    # Finally add the result to the db
    response = await completion_service.infer_instances(service_request)
    pass

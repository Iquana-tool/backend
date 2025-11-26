import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.contours import Contours
from app.database.images import Images
from app.schemas.completion_segmentation.inference import CompletionMainAPIRequest, CompletionServiceRequest
from app.schemas.contours import Contour
from app.schemas.user import User
from app.services.auth import get_current_user
from app.database import get_session
from app.services.ai_services import completion_segmentation as completion_service


router = APIRouter(prefix="/completion_segmentation", tags=["Completion Segmentation"])


@router.post("/infer/image_id={image_id}")
async def infer_completion(
        request: CompletionMainAPIRequest,
        user: User = Depends(get_current_user),
        db: Session = Depends(get_session),
):
    # First upload the image
    await completion_service.upload_image(user.username, request.image_id)
    # Finally add the result to the db
    response = await completion_service.infer_instances(request.to_service_request(db))
    pass

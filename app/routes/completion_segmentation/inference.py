from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.schemas.completion_segmentation.inference import CompletionRequest
from app.schemas.user import User
from app.services.auth import get_current_user
from app.database import get_session


router = APIRouter(prefix="/completion_segmentation", tags=["Completion Segmentation"])


@router.post("/infer/image_id={image_id}")
async def infer_completion(
        request: CompletionRequest,
        user: User = Depends(get_current_user),
        db: Session = Depends(get_session),
):
    # First upload the image

    # Second send the request to the completion backend

    # Finally add the result to the db

    pass

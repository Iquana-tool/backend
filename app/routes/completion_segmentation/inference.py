from fastapi import APIRouter, Depends
from iquana_toolbox.schemas.service_requests import CompletionRequest
from iquana_toolbox.schemas.user import User
from sqlalchemy.orm import Session

from app.database import get_session
from app.services.ai_services import completion_segmentation as completion_service
from app.services.auth import get_current_user
from app.services.contours import contour_ids_to_indices

router = APIRouter(prefix="/completion_segmentation", tags=["Completion Segmentation"])


@router.post("/infer/image_id={image_id}")
async def infer_completion(
        request: CompletionRequest,
        user: User = Depends(get_current_user),
        db: Session = Depends(get_session),
):
    raise NotImplementedError
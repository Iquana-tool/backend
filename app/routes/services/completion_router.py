from fastapi import APIRouter, Depends
from iquana_toolbox.schemas.networking.http.services import InstanceDiscoveryRequest
from iquana_toolbox.schemas.user import User
from sqlalchemy.orm import Session

from app.database import get_session
from app.services.ai_services.instance_discovery import CompletionService
from app.services.auth import get_current_user
from app.services.model_registry import list_available_models

completion_service = CompletionService()
router = APIRouter(prefix="/completion_segmentation", tags=["Completion Segmentation"])


@router.get("/models")
async def get_available_models(user: User = Depends(get_current_user)):
    """Retrieve available instance-discovery models directly from MLflow."""
    return list_available_models("instance-discovery")


@router.post("/run")
async def infer_completion(
        request: InstanceDiscoveryRequest,
        user: User = Depends(get_current_user),
        db: Session = Depends(get_session),
):
    # Finally add the result to the db
    return await completion_service.inference(request)


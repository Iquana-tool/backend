from logging import getLogger

from fastapi import Depends, APIRouter
from sqlalchemy.orm import Session

from app.database import get_session
from app.schemas.user import User
from app.services.ai_services.semantic_segmentation import segment_image_with_semantic_model
from app.services.auth import get_current_user

logger = getLogger(__name__)
router = APIRouter(prefix="/semantic_segmentation", tags=["semantic_segmentation"])


@router.post("/model={model_registry_key}&image={image_id}")
async def send_inference_job(model_registry_key: str,
                             image_id: int,
                             user: User = Depends(get_current_user),
                             db: Session = Depends(get_session)):
    """ Sends an inference job to the semantic segmentation service.

    Args:
        user: User dependency.
        model_registry_key (str): The registry key of the model you want to use.
        image_id (int): ID of the image to segment.
        db (Session): Database session dependency.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    try:
        contour_hierarchy = await segment_image_with_semantic_model(model_registry_key, image_id, db)
        return {
            "success": True,
            "message": "Inference done.",
            "contour_hierarchy": contour_hierarchy.model_dump(),
        }
    except Exception as e:
        logger.error(f"Batch prompted_segmentation failed: {e}")
        raise e
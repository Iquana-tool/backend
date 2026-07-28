"""Cross-image concept suggestion: retrieve exemplars, then transfer the concept to a target.

The one HTTP entry point that ties the pipeline together: pick exemplars from the embedding
store (retrieval strategy), resolve them, and ask the ai-service to segment the concept on the
target image using those exemplars. ``GET /strategies`` lists the selectable retrieval
strategies for the picker.
"""
from logging import getLogger

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_session
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.cross_image import (
    CrossImageSuggestResponse,
    CrossImageSuggestRequest,
    ExemplarInfo,
)
from app.schemas.permissions import Permission
from app.services.auth import get_current_user
from app.services.cross_image_orchestration import build_cross_image_request
from app.services.exemplar_retrieval import strategy_options
from app.services.permissions import ensure_permission

logger = getLogger(__name__)
router = APIRouter(prefix="/cross_image_suggestion", tags=["Cross-image Suggestion"])


@router.get("/strategies")
async def get_strategies(user: AuthenticatedUser = Depends(get_current_user)):
    """List the selectable exemplar-retrieval strategies (for the frontend picker)."""
    return strategy_options()


@router.post("/suggest", response_model=CrossImageSuggestResponse)
async def suggest(
    body: CrossImageSuggestRequest,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Suggest instances of a concept on the target image from retrieved cross-image exemplars."""
    # Authorize against the target image's dataset (AI assistance is an annotator capability).
    from app.services.permissions import dataset_id_for_image

    dataset_id = dataset_id_for_image(body.target_image_id, db)
    if dataset_id is None:
        from fastapi import HTTPException, status
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Target image {body.target_image_id} not found.")
    ensure_permission(user, dataset_id, Permission.AI_INTERACTIVE)

    request, matches = build_cross_image_request(
        db,
        target_image_id=body.target_image_id,
        strategy=body.strategy,
        concept_label_id=body.concept_label_id,
        query_contour_id=body.query_contour_id,
        top_k=body.top_k,
        user_id=str(user.username),
    )
    exemplars = [
        ExemplarInfo(contour_id=m.contour_id, image_id=m.image_id, score=m.score) for m in matches
    ]
    if request is None:
        return CrossImageSuggestResponse(
            success=True, message="No exemplars found for retrieval.", exemplars=[], result=[],
        )

    # Local import keeps the new-toolbox-schema dependency off the module import path.
    from app.services.ai_services.cross_image import CrossImageService

    response = await CrossImageService().inference(request)
    result = response.get("result", []) if isinstance(response, dict) else []
    return CrossImageSuggestResponse(
        success=True,
        message=f"Suggested {len(result)} object(s) from {len(exemplars)} exemplar(s).",
        exemplars=exemplars,
        result=result,
    )

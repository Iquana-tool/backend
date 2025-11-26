import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.contours import Contours
from app.database.images import Images
from app.schemas.completion_segmentation.inference import CompletionRequest, CompletionBackendRequest
from app.schemas.contours import Contour
from app.schemas.user import User
from app.services.auth import get_current_user
from app.database import get_session
from app.services.ai_services import completion_segmentation as completion_service


router = APIRouter(prefix="/completion_segmentation", tags=["Completion Segmentation"])


@router.post("/infer/image_id={image_id}")
async def infer_completion(
        request: CompletionRequest,
        user: User = Depends(get_current_user),
        db: Session = Depends(get_session),
):
    # First upload the image
    await completion_service.upload_image(user.username, request.image_id)
    height, width = db.query(Images.height, Images.width).filter_by(id=request.image_id).first()
    seeds = []
    label = None
    for contour_id in request.contour_ids:
        contour_db = db.query(Contours).filter_by(id=contour_id).first()
        if contour_db is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Contour with id {contour_id} not found! "
                                                                              f"Completion request failed!")
        contour_model = Contour.from_db(contour_db)
        if label is None:
            label = contour_model.label_id
        else:
            if not label == contour_model.label_id:
                raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                                    detail="You are trying to run completion on contours of different labels. This is "
                                           "not allowed!")
        seeds.append(np.argwhere(contour_model.to_binary_mask(height, width)).flatten())

    backend_req = CompletionBackendRequest(
        model_key=request.model_registry_key,
        user_id=user.username,
        seeds=seeds,
    )

    # Finally add the result to the db
    response = await completion_service.infer_instances(backend_req)
    pass

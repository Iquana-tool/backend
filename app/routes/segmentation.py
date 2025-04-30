import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_session
from app.schemas.segmentation_and_masks import (
    SegmentationRequest, SegmentationResponse, ContourModel, 
    SegmentationMaskModel, QuantificationsModel
)
from app.services.segmentation import get_model_via_identifier
from app.services.contours import get_contours
from app.services.quantifications import Contour
from app.services.database_access import get_height_width
from app.services.postprocessing import postprocess_binary_mask

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/segmentation", tags=["segmentation"])


@router.post('/segment_image')
async def segment_image(request: SegmentationRequest, db: Session = Depends(get_session)):
    """Perform segmentation with optional prompts, using data validation."""
    model = get_model_via_identifier(request.model)
    logger.debug(f"Using model: {model.model_name}")
    masks, quality = model.process_request(request)
    logger.debug(f"Segmentation completed for image_id: {request.image_id} with {len(masks)} masks.")
    height, width = get_height_width(request.image_id)
    masks_response = []
    for mask, quality in zip(masks, quality):
        if request.apply_post_processing:
            mask = postprocess_binary_mask(mask)
        contours = get_contours(mask)
        contours_response = []
        for contour in contours:
            if len(contour) < 3:
                # Skip contours with less than 3 points
                continue
            contour = Contour(contour)
            contours_response.append(ContourModel(
                x=[list_val[0] / width for list_val in contour.contour[..., 0].tolist()],
                y=[list_val[0] / height for list_val in contour.contour[..., 1].tolist()],
                label=request.label,
                quantifications=QuantificationsModel(
                    area=contour.area,
                    perimeter=contour.perimeter,
                    circularity=contour.circularity,
                    diameters=contour.get_diameters(100)
                )
            ))
        masks_response.append(SegmentationMaskModel(contours=contours_response, predicted_iou=quality))
    return SegmentationResponse(masks=masks_response, image_id=request.image_id, model=request.model)

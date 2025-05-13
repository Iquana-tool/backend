import logging
import cv2
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import config
from app.database import get_session
from app.database.images import ImageEmbeddings, Images
from app.schemas.segmentation_and_masks import (
    SegmentationRequest, SegmentationResponse, ContourModel,
    SegmentationMaskModel, QuantificationsModel
)
from app.schemas.scale import ScaleInput
from app.services.database_access import load_image_as_array_from_disk, load_embedding, save_embeddings_to_disk
from app.services.prompts import Prompts
from app.services.segmentation.sam2 import SAM2, set_current_image_id
from app.services.contours import get_contours
from app.services.quantifications import Contour
from app.services.scale_computation import compute_pixel_scale_from_points

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/segmentation", tags=["segmentation"])


@router.post('/segment_image')
async def segment_image(request: SegmentationRequest):
    """Perform segmentation with optional prompts, using data validation."""
    # Get the model based on the identifier
    model = get_model_via_identifier(request.model)
    logger.debug(f"Using model: {model.model_name}")

    # Process the request with the model
    # This method should handle the image preprocessing and segmentation
    # All model specific logic should be encapsulated in the model class
    masks, quality = model.process_request(request)
    logger.debug(f"Segmentation completed for image_id: {request.image_id} with {len(masks)} masks.")

    # Postprocess the masks and get contours
    height, width = get_height_width_of_image(request.image_id)
    masks_response = []
    for mask, quality in zip(masks, quality):
        # Get contours of the postprocessed mask if postprocessing is enabled
        # Postprocessing might improve performance by removing noise
        contours = get_contours(postprocess_binary_mask(mask) if request.apply_post_processing else mask)
        contours_response = []
        for contour in contours:
            contour = Contour(contour)
            if contour.area <= 0 or contour.perimeter <= 0:
                # We could filter here based on the area or perimeter or other quantifications from the contour
                continue
            contours_response.append(ContourModel(
                x=[x_coord / width for x_coord in contour.x_coords],  # Scale x-coordinates to [0, 1]
                y=[y_coord / height for y_coord in contour.y_coords],  # Scale y-coordinates to [0, 1]
                label=request.label,
                quantifications=QuantificationsModel(
                    area=contour.area,
                    perimeter=contour.perimeter,
                    circularity=contour.circularity,
                    diameters=contour.get_diameters()
                )
            ))
        masks_response.append(SegmentationMaskModel(contours=contours_response, predicted_iou=quality))
    return SegmentationResponse(masks=masks_response, image_id=request.image_id, model=request.model)

@router.post('/set_pixel_scale_via_drawn_line')
def set_pixel_scale_via_drawn_line(scale_input: ScaleInput, db: Session = Depends(get_session)):
    """
    Set the pixel scale based on a known distance between two points drawn on the image.
    """
    image = db.query(Images).filter_by(id=scale_input.image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    if not scale_input.unit:
        raise HTTPException(status_code=400, detail="Unit must be provided (e.g., mm)")

    # Compute the scale in both directions
    scale_x, scale_y = compute_pixel_scale_from_points(
        (scale_input.x1, scale_input.y1),
        (scale_input.x2, scale_input.y2),
        scale_input.known_distance
    )

    # Save scale information in the DB
    image.scale_x = scale_x
    image.scale_y = scale_y
    image.unit = scale_input.unit
    db.commit()

    return {
        "message": "Scale set successfully",
        "scale_x": scale_x,
        "scale_y": scale_y,
        "unit": image.unit
    }

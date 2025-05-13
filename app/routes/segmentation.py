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
from app.services.database_access import get_height_width_of_image
from app.services.postprocessing import postprocess_binary_mask

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/segmentation", tags=["segmentation"])


@router.post('/segment_image')
async def segment_image(request: SegmentationRequest):
    """Perform segmentation with optional prompts, using data validation.
    This function handles the segmentation of images based on the provided request.
    It validates the request, retrieves the appropriate model, and processes the image.

    Args:
        request (SegmentationRequest): The request object containing image data and parameters. When using cropping,
        make sure to remap the annotation coordinates to the cropped image.

    Returns:
        SegmentationResponse: The response object containing the segmentation results. When using cropping,
        the contours will be remapped to the original image size.
    """
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
                # We have to rescale the images to the original size
                x=[(x_coord + int(request.min_x * width)) / width for x_coord in contour.x_coords],
                y=[(y_coord + int(request.min_y * height)) / height for y_coord in contour.y_coords],
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

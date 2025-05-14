from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.scale import ScaleInput
from app.database import get_session
from app.database.images import Images
from app.services.scale_computation import compute_pixel_scale_from_points
from logging import getLogger

# Set up logging
logger = getLogger(__name__)

# Create a router for pixel scale-related routes
router = APIRouter()


@router.post('/set_pixel_scale')
async def set_pixel_scale(scale_x: float, scale_y: float, unit: str, image_id: int, db: Session = Depends(get_session)):
    """
    Set the pixel scale for an image.
    """
    # Fetch the image from the database
    image = db.query(Images).filter_by(id=image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Ensure the unit is provided
    if not unit:
        raise HTTPException(status_code=400, detail="Unit must be provided (e.g., mm)")

    if image.scale_x is not None or image.scale_y is not None:
        logger.warning("Overwriting existing pixel scale values.")

    # Save scale information in the database
    image.scale_x = scale_x
    image.scale_y = scale_y
    image.unit = unit
    db.commit()

    # Return a success response
    return {
        "message": "Scale set successfully",
        "scale_x": scale_x,
        "scale_y": scale_y,
        "unit": unit
    }

@router.post('/set_pixel_scale_via_drawn_line')
def set_pixel_scale_via_drawn_line(scale_input: ScaleInput, db: Session = Depends(get_session)):
    """
    Set the pixel scale based on a known distance between two points drawn on the image.
    """

    # Compute the scale in both directions
    scale_x, scale_y = compute_pixel_scale_from_points(
        (scale_input.x1, scale_input.y1),
        (scale_input.x2, scale_input.y2),
        scale_input.known_distance
    )

    # Return a success response
    return set_pixel_scale(scale_x, scale_y, scale_input.unit, scale_input.image_id, db)

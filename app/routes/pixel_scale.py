from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.scale import ScaleInput
from app.database import get_session
from app.database.images import Images
from app.services.scale_computation import compute_pixel_scale_from_points

# Create a router for pixel scale-related routes
router = APIRouter()

@router.post('/set_pixel_scale_via_drawn_line')
def set_pixel_scale_via_drawn_line(scale_input: ScaleInput, db: Session = Depends(get_session)):
    """
    Set the pixel scale based on a known distance between two points drawn on the image.
    """
    # Fetch the image from the database
    image = db.query(Images).filter_by(id=scale_input.image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Ensure the unit is provided
    if not scale_input.unit:
        raise HTTPException(status_code=400, detail="Unit must be provided (e.g., mm)")

    # Compute the scale in both directions
    scale_x, scale_y = compute_pixel_scale_from_points(
        (scale_input.x1, scale_input.y1),
        (scale_input.x2, scale_input.y2),
        scale_input.known_distance
    )

    # Save scale information in the database
    image.scale_x = scale_x
    image.scale_y = scale_y
    image.unit = scale_input.unit
    db.commit()

    # Return a success response
    return {
        "message": "Scale set successfully",
        "scale_x": scale_x,
        "scale_y": scale_y,
        "unit": image.unit
    }

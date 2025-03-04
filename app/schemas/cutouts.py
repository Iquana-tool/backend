from pydantic import BaseModel, field_validator


class CutoutsRequest(BaseModel):
    """Model for validating the cutouts request."""
    image_id: int
    base64_mask: str
    resize_factor: float
    darken_outside_contours: bool
    darkening_factor: float

    @field_validator('resize_factor', 'darkening_factor')
    def validate_values(cls, value):
        if not 0 < value <= 1.0:
            raise ValueError("Values must be a float in ]0;1] (excluding 0).")
        return value

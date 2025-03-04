from pydantic import BaseModel, field_validator
from typing import List
from app.database.images import Images
import numpy as np
from app.schemas.images import ImageID, Base64Image
from app.schemas.masks import RLEString


class CutoutsRequest(BaseModel):
    """Model for validating the cutouts request."""
    image_id: ImageID
    rle_mask: RLEString
    resize_factor: float
    darken_outside_contours: bool
    darkening_factor: float

    @field_validator('resize_factor', 'darkening_factor')
    def validate_values(cls, value):
        if not 0 < value <= 1.0:
            raise ValueError("Values must be a float in ]0;1] (excluding 0).")
        return value

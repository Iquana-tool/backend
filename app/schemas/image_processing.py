from pydantic import BaseModel, Field, conlist, validator, field_serializer, field_validator
from typing import List, Optional
from app.database.images import Images
import numpy as np


class ImagesResponse(BaseModel):
    """ Model for validating the images response. """
    images: dict[int, np.ndarray]

    class Config:
        arbitrary_types_allowed = True

    @field_validator('images')
    def validate_images(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Images must be a dictionary.")
        for key, image in value.items():
            if not isinstance(key, int):
                raise ValueError("Keys must be integers.")
            elif not isinstance(image, np.ndarray):
                raise ValueError("Image data must be a numpy array.")
            elif image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Image data must be in RGB format.")
            elif np.any(image < 0.0) or np.any(image > 1.0):
                raise ValueError("Image data must be in the range [0, 255].")

class ImagesRequest(BaseModel):
    ids: List[int]

    @field_validator('ids')
    def validate_ids(cls, value):
        if not isinstance(value, list):
            raise ValueError("Ids must be a list.")
        error_ids = []
        for id in value:
            if not isinstance(id, int):
                raise ValueError("Ids must be integers.")
            elif id <= 0:
                raise ValueError("Ids must be positive integers.")
            elif Images.query.filter_by(id=id).first() is None:
                error_ids.append(id)
        if error_ids:
            raise ValueError(f"The following ids do not exist in the database: {error_ids}.")


class CutoutsRequest(BaseModel):
    """ Model for validating the cutouts request. """
    image_id: int
    mask: np.ndarray[bool]
    resize_factor: float
    darken_outside_contours: bool
    darkening_factor: float

    class Config:
        arbitrary_types_allowed = True

    @field_validator('image_id', 'mask')
    def validate_image_id(cls, value):
        if value <= 0:
            raise ValueError("image_id must be a positive integer.")
        elif Images.query.filter_by(id=value).first() is None:
            raise ValueError("image_id does not exist in the database.")

    @field_validator('resize_factor', 'darkening_factor')
    def validate_values(cls, value):
        if not 0 < value <= 1.0:
            raise ValueError("Values must be a float in ]0;1] (excluding 0).")

    @field_validator('mask')
    def validate_mask(cls, value):
        if np.all(value == 0) or np.all(value == 1):
            raise ValueError("Mask must contain both 0 and 1 values.")

class CutoutsResponse(BaseModel):
    """ Model for validating the cutouts response. """
    cutouts: List[np.ndarray]

    class Config:
        arbitrary_types_allowed = True

    @field_validator('cutouts')
    def validate_cutouts(cls, value):
        if not isinstance(value, list):
            raise ValueError("Cutouts must be a list.")
        for cutout in value:
            if not isinstance(cutout, np.ndarray):
                raise ValueError("Cutout data must be a numpy array.")
            elif cutout.ndim != 3 or cutout.shape[2] != 3:
                raise ValueError("Cutout data must be in RGB format.")
            elif np.any(cutout < 0.0) or np.any(cutout > 1.0):
                raise ValueError("Cutout data must be in the range [0, 255].")

from pydantic import BaseModel, Field, conlist, validator, field_serializer, field_validator, root_validator, \
    model_validator
from typing import List, Optional
from app.database.images import Images
from app.database import get_context_session
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
    """Model for validating the cutouts request."""
    image_id: int
    rle_mask: str
    resize_factor: float
    darken_outside_contours: bool
    darkening_factor: float

    class Config:
        arbitrary_types_allowed = True

    @field_validator('image_id')
    def validate_image_id(cls, value):
        if value <= 0:
            raise ValueError("image_id must be a positive integer.")
        elif Images.query.filter_by(id=value).first() is None:
            raise ValueError("image_id does not exist in the database.")
        return value

    @field_validator('resize_factor', 'darkening_factor')
    def validate_values(cls, value):
        if not 0 < value <= 1.0:
            raise ValueError("Values must be a float in ]0;1] (excluding 0).")
        return value

    @field_validator('rle_mask')
    def validate_rle_mask(cls, value):
        # Decode the RLE mask to check its validity
        try:
            mask = cls.rle_decode(value)
        except Exception as e:
            raise ValueError(f"Invalid RLE mask: {e}")

        if np.all(mask == 0) or np.all(mask == 1):
            raise ValueError("Mask must contain both 0 and 1 values.")
        return value

    @model_validator(mode="after")
    def decode_rle_mask(cls, values):
        # Decode the RLE mask and store it as a numpy array
        rle_mask = values.get('rle_mask')
        if rle_mask:
            values['mask'] = cls.rle_decode(rle_mask)
        return values

    @staticmethod
    def rle_decode(rle_str):
        """Decodes an RLE encoded mask."""
        s = rle_str.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
        starts -= 1
        ends = starts + lengths
        mask = np.zeros(max(ends), dtype=np.uint8)
        for start, end in zip(starts, ends):
            mask[start:end] = 1
        with get_context_session() as session:
            image = session.query(Images).filter_by(id=1).first()
        return mask.reshape((image.height, image.width))  # Replace height and width with actual dimensions



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

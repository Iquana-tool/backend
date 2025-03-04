from pydantic import BaseModel, Field, conlist, validator, field_serializer, field_validator, root_validator, \
    model_validator
from typing import List, Optional
from app.database.images import Images
from app.database import get_context_session
import numpy as np


class ImageID(BaseModel):
    """Model for validating an image ID."""
    image_id: int

    @field_validator('image_id')
    def validate_image_id(cls, value):
        if not isinstance(value, int):
            raise ValueError("Image ID must be an integer.")
        if value <= 0:
            raise ValueError("Image ID must be a positive integer.")
        with get_context_session() as session:
            if session.query(Images).filter_by(id=value).first() is None:
                raise ValueError("Image ID does not exist in the database.")
        return value


class Base64Image(BaseModel):
    """Model for validating a base64 image."""
    image_id: ImageID
    image: str

    @field_validator('image')
    def validate_image(cls, value):
        if not isinstance(value, str):
            raise ValueError("Image must be a string.")
        if len(value) == 0:
            raise ValueError("Image must not be empty.")
        return value

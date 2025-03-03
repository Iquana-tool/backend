from pydantic import BaseModel, Field, conlist, validator, field_serializer, field_validator
from typing import List, Optional
from app.database.images import Images


class PointPrompt(BaseModel):
    """ Model for validating a point annotation. """
    x: float
    y: float
    label: int

    @field_validator('label')
    def validate_label(cls, value):
        if value not in [0, 1]:
            raise ValueError("Label must be 0 (background) or 1 (foreground).")

    @field_validator('x', 'y')
    def validate_coordinates(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("Coordinates must be between 0 and 1.")


class BoxPrompt(BaseModel):
    """ Model for validating a bounding box annotation. """
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @field_validator('min_x', 'min_y', 'max_x', 'max_y')
    def validate_coordinates(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("Box coordinates must be between 0 and 1.")


class SegmentationRequest(BaseModel):
    """ Model for validating the segmentation request. """
    use_prompts: bool
    image_id: int
    point_prompts: Optional[List[PointPrompt]] = Field(default_factory=list)
    box_prompts: Optional[List[BoxPrompt]] = Field(default_factory=list)

    @field_validator('image_id')
    def validate_image_id(cls, value):
        if value <= 0:
            raise ValueError("image_id must be a positive integer.")
        elif Images.query.filter_by(id=value).first() is None:
            raise ValueError("image_id does not exist in the database.")


class SegmentationResponse(BaseModel):
    """ Model for validating the segmentation response. """
    masks: List[conlist(bool, min_length=1)]  # Nested list for masks
    quality: List[float]

from typing import List, Annotated

from pydantic import BaseModel, Field, field_validator

import config
from app.database import get_context_session
from app.database.images import Images


class PointPrompt(BaseModel):
    """ Model for validating a point annotation. """
    x: Annotated[float, "Coordinates must be between 0 and 1."]
    y: Annotated[float, "Coordinates must be between 0 and 1."]
    label: Annotated[bool, "Label must be 0 (background) or 1 (foreground)."]

    @field_validator('label')
    def validate_label(cls, value):
        if value not in [True, False]:
            raise ValueError("Label must be 0 (background) or 1 (foreground).")
        return value

    @field_validator('x', 'y')
    def validate_coordinates(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("Coordinates must be between 0 and 1.")
        return value


class BoxPrompt(BaseModel):
    """ Model for validating a bounding box annotation. """
    min_x: Annotated[float, "Coordinates must be between 0 and 1."]
    min_y: Annotated[float, "Coordinates must be between 0 and 1."]
    max_x: Annotated[float, "Coordinates must be between 0 and 1."]
    max_y: Annotated[float, "Coordinates must be between 0 and 1."]

    @field_validator('min_x', 'min_y', 'max_x', 'max_y')
    def validate_coordinates(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("Box coordinates must be between 0 and 1.")
        return value


class PolygonPrompt(BaseModel):
    """ Model for validating a polygon annotation. """
    vertices: Annotated[List[List[float]], "List of vertices of the polygon."] = Field(default_factory=list)

    @field_validator('vertices')
    def validate_vertices(cls, value):
        if len(value) < 3:
            raise ValueError("Polygon must have at least 3 vertices.")
        for vertex in value:
            if len(vertex) != 2:
                raise ValueError("Each vertex must have exactly 2 coordinates.")
            if not all(0 <= coord <= 1 for coord in vertex):
                raise ValueError("Coordinates must be between 0 and 1.")
        return value


class CirclePrompt(BaseModel):
    """ Model for validating a circle annotation. """
    center_x: Annotated[float, "Coordinates must be between 0 and 1."]
    center_y: Annotated[float, "Coordinates must be between 0 and 1."]
    radius: Annotated[float, "Radius must be a positive float."]

    @field_validator('center_x', 'center_y')
    def validate_coordinates(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("Coordinates must be between 0 and 1.")
        return value

    @field_validator('radius')
    def validate_radius(cls, value):
        if value <= 0:
            raise ValueError("Radius must be a positive float.")
        return value


class SegmentationRequest(BaseModel):
    """ Model for validating the segmentation request. """
    use_prompts: Annotated[bool, ("Use prompts for segmentation (=true) or use automatic segmentation "
                                  "without prompts (=false).")] = True
    image_id: Annotated[int, "ID of the image to segment."] = 0
    model: Annotated[str, "Model to use for segmentation."] = "SAM2Tiny"
    point_prompts: Annotated[List[PointPrompt], "List of point prompts supplied by the user"] = (
        Field(default_factory=list))
    box_prompts: Annotated[List[BoxPrompt], "List of box prompts supplied by the user"] = Field(default_factory=list)
    polygon_prompts: Annotated[List[PolygonPrompt], "List of polygon prompts supplied by the user"] = (
        Field(default_factory=list)
    )
    circle_prompts: Annotated[List[CirclePrompt], "List of circle prompts supplied by the user"] = (
        Field(default_factory=list)
    )

    @field_validator('image_id')
    def validate_image_id(cls, value):
        with get_context_session() as session:
            if value <= 0:
                raise ValueError("image_id must be a positive integer.")
            elif session.query(Images).filter_by(id=value).first() is None:
                raise ValueError("image_id does not exist in the database.")
            return value

    @field_validator("model")
    def validate_model(cls, value):
        if not value in config.ModelConfig.available_models.keys():
            raise ValueError("Model must be one of {}.".format(config.ModelConfig.available_models.keys()))
        return value

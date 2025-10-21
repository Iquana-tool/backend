from typing import Annotated, List

from pydantic import BaseModel, field_validator, Field


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

    @field_validator('center_x', 'center_y', 'radius')
    def validate_coordinates(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("Coordinates must be between 0 and 1.")
        return value


class Prompts(BaseModel):
    """ Model for validating a prompted prompted_segmentation request. One request represents one object to be segmented."""
    point_prompts: list[PointPrompt] | None = Field(default=None,
                                             description="A list of point prompts. Each point prompt must have x, y, and label.")
    box_prompt: BoxPrompt | None = Field(default=None,
                                         description="A bounding box prompt. Must have min_x, min_y, max_x, and max_y.")
    circle_prompt: CirclePrompt | None = Field(default=None,
                                               description="A circle prompt. Must have center_x, center_y, and radius.")
    polygon_prompt: PolygonPrompt | None = Field(default=None,
                                                 description="A polygon prompt. Must have a list of vertices.")

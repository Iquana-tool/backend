from typing import List, Annotated

from pydantic import BaseModel, field_validator, Field


class QuantificationsModel(BaseModel):
    """ Model for the quantifications. """
    area: float
    perimeter: float
    circularity: float
    diameters: List[float]

    @field_validator('area', 'perimeter', 'circularity')
    def validate_positive(cls, value):
        if value <= 0:
            raise ValueError("Area, perimeter, and circularity must be positive values.")
        return value

    @field_validator('diameters')
    def validate_diameters(cls, value):
        if not all(isinstance(diameter, (int, float)) and diameter > 0 for diameter in value):
            raise ValueError("Diameters must be a list of positive values.")
        return value


class ContourModel(BaseModel):
    """ Model for the contour. """
    x: List[float] = Field(default_factory=list, description="X-coordinates of the contour.")
    y: List[float] = Field(default_factory=list, description="Y-coordinates of the contour.")
    label: Annotated[int, "Label of the mask."] = Field(default=0, description="Label of the mask.")
    # quantifications: QuantificationsModel

    @field_validator('x', 'y')
    def validate_coordinates(cls, value):
        return [min(max(coord, 0), 1) for coord in value]

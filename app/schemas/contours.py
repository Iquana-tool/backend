from typing import List, Annotated

from pydantic import BaseModel, field_validator, Field


class ContourModel(BaseModel):
    """ Model for the contour. """
    x: List[float] = Field(default_factory=list, description="X-coordinates of the contour.")
    y: List[float] = Field(default_factory=list, description="Y-coordinates of the contour.")
    label: Annotated[int, "Label of the mask."] = Field(default=0, description="Label of the mask.")

    @field_validator('x', 'y')
    def validate_coordinates(cls, value):
        return [min(max(coord, 0), 1) for coord in value]

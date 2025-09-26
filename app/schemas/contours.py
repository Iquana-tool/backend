from typing import List, Annotated
from app.database import get_context_session
from app.database.contours import Contours
from app.database.masks import Masks
from pydantic import BaseModel, field_validator, Field, ValidationError


class ContourModel(BaseModel):
    """ Model for a contour to be added. """
    x: List[float] = Field(default_factory=list, description="X-coordinates of the contour.")
    y: List[float] = Field(default_factory=list, description="Y-coordinates of the contour.")
    label: int | None = Field(default=None, description="ID of the label of the mask. None for unlabelled contour.")
    parent_contour_id: int | None = Field(default=None, description="ID of the parent contour. None if the contour has "
                                                                    "no parent")
    added_by: str = Field(default_factory=str, description="ID of the user or model who added this contour.")
    confidence: float = Field(default=1., description="Confidence score of the contour.")
    temporary: bool = Field(default=False, description="Whether or not this contour is temporarily added (eg. if a model added it).")
    area: float | None = Field(default=None, description="Area of the contour.")
    perimeter: float | None = Field(default=None, description="Perimeter of the contour.")
    circularity: float | None = Field(default=None, description="Circularity of the contour.")
    diameters: List[float] | None = Field(default=None, description="List of diameters of the contour measured from "
                                                                    "different angles.")

    @field_validator('x', 'y')
    def validate_coordinates(cls, value):
        return [min(max(coord, 0), 1) for coord in value]

from typing import List, Annotated, Any, Self

import cv2
import numpy as np

from app.database import get_context_session
from app.database.contours import Contours
from app.database.masks import Masks
from pydantic import BaseModel, field_validator, Field, ValidationError, model_validator

from app.schemas.quantification import QuantificationModel


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
    quantification: QuantificationModel | None = Field(default=None, description="Quantification of the contour. Does "
                                                                                 "not need to be provided.")

    @model_validator(mode="after")
    def validate_after(self):
        """ Validate after initialization. Check if quantifications are computed. """
        if self.quantification is None:
            self.quantification = QuantificationModel.from_cv_contour(self.contour)
        elif self.quantification.is_empty:
            self.quantification.parse_cv_contour(self.contour)



    @field_validator('x', 'y')
    def validate_coordinates(cls, value):
        return [min(max(coord, 0), 1) for coord in value]

    @property
    def contour(self) -> np.ndarray:
        """ As a opencv contour. """
        return np.array([[self.x, self.y]])

    @property
    def points(self) -> np.ndarray[tuple[float, float]]:
        return np.array(zip(self.x, self.y))

    def to_orm(self, id, mask_id):
        return Contours(
            id=id,
            mask_id=mask_id,
            coords={"x": self.x, "y": self.y},
            label=self.label,
            parent_id=self.parent_contour_id,
            temporary=self.temporary,
            added_by=self.added_by,
            confidence_score=self.confidence,
            area=self.quantification.area,
            perimeter=self.quantification.perimeter,
            circularity=self.quantification.circularity,
            diameters=self.quantification.diameters,
        )

    @classmethod
    def from_cv_contour(cls, cv_contour, label, height, width):
        x_coords = cv_contour[..., 0].flatten() / width  # Normalize x-coordinates
        y_coords = cv_contour[..., 1].flatten() / height  # Normalize y-coordinates
        return ContourModel(
            x=x_coords.tolist(),
            y=y_coords.tolist(),
            label=label
        )

    def __in__(self, other):
        """
        Is this contour contained in another contour. Checks whether all points in this contour are inside another.
        """
        return all(cv2.pointPolygonTest(
            (other.contour * 10_000).astype(np.uint32),
            pt=(pt * 10_000).astype(np.uint32),
            measureDist=False) >= 0 for pt in self.points)


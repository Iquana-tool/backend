import numpy as np
from pydantic import BaseModel, Field
import cv2 as cv
from scipy.spatial.distance import pdist, squareform


class QuantificationModel(BaseModel):
    """ A model to keep track of quantifications and enable easy extensions to the quantifications."""
    area: float | None = Field(default=None, description="Area of the contour.")
    perimeter: float | None = Field(default=None, description="Perimeter of the contour.")
    circularity: float | None = Field(default=None, description="Circularity of the contour.")
    max_diameter:float | None = Field(default=None, description="Maximum distance of any two points in "
                                                                       "the contour.")

    @property
    def is_empty(self) -> bool:
        return self.area is None

    def parse_cv_contour(self, cv_contour):
        # Compute quantification values with opencv functions
        self.area = cv.contourArea(cv_contour)
        self.perimeter = cv.arcLength(cv_contour, True)
        if self.area == 0:
            self.circularity = 0
        else:
            self.circularity = (4 * np.pi * self.area) / (self.perimeter ** 2)

        # Compute the max diameter separately, because no function exists.
        # Squeeze one dimensional dimension. Idk why opencv has this.
        points = cv_contour.squeeze()
        # Compute all pairwise distances
        distances = squareform(pdist(points, 'euclidean'))
        # The diameter is the maximum distance
        self.max_diameter = np.max(distances)

    @classmethod
    def from_cv_contour(cls, cv_contour: np.ndarray):
        cv_contour = cv_contour.astype(np.float32)
        # Compute quantification values with opencv functions
        area_px = cv.contourArea(cv_contour)
        perimeter_px = cv.arcLength(cv_contour, True)
        if area_px == 0:
            circularity = 0
        else:
            circularity = (4 * np.pi * area_px) / (perimeter_px ** 2)

        # Compute the max diameter separately, because no function exists.
        # Squeeze one dimensional dimension. Idk why opencv has this.
        points = cv_contour.squeeze()
        # Compute all pairwise distances
        distances = squareform(pdist(points, 'euclidean'))
        # The diameter is the maximum distance
        diameter = np.max(distances)
        return QuantificationModel(
            area=area_px,
            perimeter=perimeter_px,
            circularity=circularity,
            max_diameter=diameter,
        )


import cv2 as cv
import numpy as np
from logging import getLogger

# Set up logging
logger = getLogger(__name__)


class Contour:
    area = None
    perimeter = None
    circularity = None

    def __init__(self, contour):
        """ Initialize the quantifier.

            Args:
                contour (np.array): The contour of the image.
        """
        self.contour = contour
        self.parse_contour()

    @property
    def x_coords(self):
        """ Get the x-coordinates of the contour.
        """
        return self.contour[..., 0]

    @property
    def y_coords(self):
        """ Get the y-coordinates of the contour.
        """
        return self.contour[..., 1]

    def parse_contour(self):
        """ Parse the contour to get the area, perimeter, circularity and diameters.
        """
        self.area = cv.contourArea(self.contour)
        self.perimeter = cv.arcLength(self.contour, True)
        if self.area == 0:
            self.circularity = 0
        else:
            self.circularity = (4 * np.pi * self.area) / (self.perimeter ** 2)

    def get_diameters(self, step_size=25):
        """ Get the diameters of the objects in the mask.
        Args:
            step_size (int): The number of times to measure the diameter. This will measure the diameter each
                180/step_size degrees. E.g. step_size=100 will measure the diameter at 0, 1.8, 3.6, ...,
                178.2, and 180 degrees.
        Returns:
            list: The diameters of the objects in the mask.
        """
        measuring_degrees = np.linspace(0, 180, step_size)
        contour_mask = np.zeros((np.max(self.contour[..., 1] + 1), np.max(self.contour[..., 0] + 1)), dtype=np.uint8)
        contour_mask = cv.drawContours(contour_mask, [self.contour], 0, 1, 1)
        Cx, Cy = np.mean(self.contour[..., 0]), np.mean(self.contour[..., 1])
        diameters = []
        for degree in measuring_degrees:
            radian = np.deg2rad(degree)
            x1 = int(Cx + np.cos(radian) * 1000)
            y1 = int(Cy + np.sin(radian) * 1000)
            x2 = int(Cx - np.cos(radian) * 1000)
            y2 = int(Cy - np.sin(radian) * 1000)
            # Get the line between the center and the point
            line = cv.line(np.zeros_like(
                contour_mask, dtype=np.uint8),
                (x1, y1), (x2, y2),
                   color=1, thickness=1
            )
            # Get the intersection of the line and the contour
            intersection = cv.bitwise_and(line, contour_mask)
            if np.count_nonzero(intersection) == 2:
                # Get the distance between the two intersecting points
                # It can happen that there are more or less than 2 points, but we should ignore these cases
                distance = np.linalg.norm(np.argwhere(intersection))
                diameters.append(distance)
        return diameters

import cv2 as cv
import numpy as np
from logging import getLogger

# Set up logging
logger = getLogger(__name__)


class ContourQuantifier:
    """ Quantifier to compute area, perimeter, and circularity of a contour. """
    def __init__(self, scale_x=1.0, scale_y=1.0, unit="px"):
        """
        Initialize the quantifier.

        Args:
            contour (np.array): The contour of the image.
            scale_x (float): Scale in x direction (e.g., mm per pixel).
            scale_y (float): Scale in y direction (e.g., mm per pixel).
            unit (str): Unit of measurement (e.g., "mm").
        """
        self.contour = None
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.unit = unit

        self.area = None
        self.perimeter = None
        self.circularity = None

    def from_contour(self, contour):
        """
        Set the contour and reparse.
        """
        self.contour = contour
        self.parse_contour()

    def from_coordinates(self, x_coords, y_coords):
        """
        Set the coordinates and reparse.
        """
        # Bring the coordinates into opencv contour format
        self.contour = np.expand_dims(np.array(list(zip(x_coords, y_coords)), dtype=np.int32), 1)
        self.parse_contour()

    @property
    def x_coords(self):
        """ Get the x-coordinates of the contour.
        """
        return self.contour[..., 1]

    @property
    def y_coords(self):
        """ Get the y-coordinates of the contour.
        """
        return self.contour[..., 0]

    def parse_contour(self):
        """
        Parse the contour to get area (in unit²), perimeter (in unit), and circularity.
        """
        area_px = cv.contourArea(self.contour)
        self.area = area_px * self.scale_x * self.scale_y

        perimeter_px = cv.arcLength(self.contour, True)
        avg_scale = (self.scale_x + self.scale_y) / 2
        self.perimeter = perimeter_px * avg_scale

        if self.area == 0:
            self.circularity = 0
        else:
            self.circularity = (4 * np.pi * area_px) / (perimeter_px ** 2)

    def get_diameters(self, step_size=100):
        """
        Measure diameters from multiple angles around the centroid and rescale.

        Args:
            step_size (int): Number of angles to measure between 0 and 180 degrees.

        Returns:
            list: List of diameters in the same unit as the scale (e.g., mm).
        """
        measuring_degrees = np.linspace(0, 180, step_size)
        max_y = np.max(self.contour[..., 1]) + 1
        max_x = np.max(self.contour[..., 0]) + 1
        contour_mask = np.zeros((max_y, max_x), dtype=np.uint8)
        cv.drawContours(contour_mask, [self.contour], 0, 1, 1)

        Cx, Cy = np.mean(self.contour[..., 0]), np.mean(self.contour[..., 1])
        diameters = []

        for degree in measuring_degrees:
            rad = np.deg2rad(degree)
            x1 = int(Cx + np.cos(rad) * 1000)
            y1 = int(Cy + np.sin(rad) * 1000)
            x2 = int(Cx - np.cos(rad) * 1000)
            y2 = int(Cy - np.sin(rad) * 1000)

            line = cv.line(np.zeros_like(contour_mask), (x1, y1), (x2, y2), color=1, thickness=1)
            intersection = cv.bitwise_and(line, contour_mask)
            points = np.argwhere(intersection)

            if len(points) >= 2:
                pt1, pt2 = points[0], points[-1]
                dx = (pt2[1] - pt1[1]) * self.scale_x
                dy = (pt2[0] - pt1[0]) * self.scale_y
                distance = np.hypot(dx, dy)
                diameters.append(distance)

        return diameters

import cv2 as cv
import numpy as np
from app.services.cutouts import get_contours
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

    def parse_contour(self):
        """ Parse the contour to get the area, perimeter, circularity and diameters.
        """
        self.area = cv.contourArea(self.contour)
        self.perimeter = cv.arcLength(self.contour, True)
        if self.area == 0:
            self.circularity = 0
        else:
            self.circularity = (4 * np.pi * self.area) / (self.perimeter ** 2)

    def get_diameters(self, step_size=100):
        """ Get the diameters of the objects in the mask.
        Args:
            step_size (int): The number of times to measure the diameter. This will measure the diameter each
                180/step_size degrees. E.g. step_size=100 will measure the diameter at 0, 1.8, 3.6, ...,
                178.2, and 180 degrees.
        Returns:
            list: The diameters of the objects in the mask.
        """
        measuring_degrees = np.linspace(0, 180, step_size)
        contour_mask = np.zeros_like(self.mask)
        contour_mask = cv.drawContours(contour_mask, [self.contour], -1, color=1)
        Cx, Cy = np.mean(self.contour, axis=0)
        diameters = []
        for degree in measuring_degrees:
            radian = np.deg2rad(degree)
            x = int(Cx + np.cos(radian) * 1000)
            y = int(Cy + np.sin(radian) * 1000)
            # Get the line between the center and the point
            line = cv.line(np.zeros_like(self.mask), (int(Cx), int(Cy)), (x, y), color=1, thickness=1)
            # Get the intersection of the line and the contour
            intersection = cv.bitwise_and(line, contour_mask)
            if np.count_nonzero(intersection) > 2:
                logger.warning("More than 2 points in the intersection. This indicates non spherical shape.")
            elif np.count_nonzero(intersection) == 2:
                # Get the distance between the two intersecting points
                distance = np.linalg.norm(np.argwhere(intersection))
                diameters.append(distance)
            else:
                logger.error("Less than 2 points in the intersection. This indicates one dimensional shape or "
                             "empty contour.")
        return diameters



class Quantifier2D:
    """
    Class to handle 2D quantifications.
    """
    def __init__(self, mask, image, unit="px", pixel_scale=1.):
        """ Initialize the quantifier.

            Args:
                mask (np.array): The mask of the image.
                image (np.array): The image to be quantified.
                unit (str): The unit of quantifier. E.g. px (pixel scale), mm (millimeter).
                pixel_scale (float): The pixel scale of the image. One pixel represents this many of the unit.
        """
        self.mask = mask
        self.objects = [Contour(contour) for contour in get_contours(mask)]
        self.image = image
        self.unit = unit
        self.scale = pixel_scale

    def get_area(self):
        return [contour.area for contour in self.objects]

    def get_diameters(self, step_size=100):
        """ Get the diameters of the objects in the mask.

        Args:
            step_size (int): The number of times to measure the diameter. This will measure the diameter each
                180/step_size degrees. E.g. step_size=100 will measure the diameter at 0, 1.8, 3.6, ...,
                178.2, and 180 degrees.
        Returns:
            list: The diameters of the objects in the mask.
        """
        return [contour.get_diameters(step_size) for contour in self.objects]

    def get_perimeters(self):
        """ Get the perimeters of the objects in the mask.

        Returns:
            list: The perimeters of the objects in the mask.
        """
        return [contour.perimeter for contour in self.objects]

    def get_circularities(self):
        """ Get the circularities of the objects in the mask.

        Returns:
            list: The circularities of the objects in the mask.
        """
        return [contour.circularity for contour in self.objects]



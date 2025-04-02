import cv2 as cv
import numpy as np
from app.services.cutouts import get_contours


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
        self.objects = get_contours(mask)
        self.image = image
        self.unit = unit
        self.scale = pixel_scale

    def get_area(self):
        results = []
        for contour in self.objects:
            canvas = np.zeros_like(self.mask)
            canvas = cv.fillPoly(canvas, [contour], color=1)
            # Count the nonzero pixels and multiply by the scale. Squared because it is an area measure.
            results.append((np.count_nonzero(canvas) * self.scale ** 2))
        return results

    def get_diameters(self):
        results = []
        for contour in self.objects:
            diameter = []
            center = contour.centroid
            # For each point on the contour compute the distance to the next point through the center


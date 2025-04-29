import numpy as np
from app.services.segmentation import Prompts
from app.services.segmentation import SegmentationBaseModel
from typing import Union


def create_random_mask(height: int, width: int) -> np.ndarray:
    """ Create a mask of the given height and width with random circles.
        Args:
            height (int): The height of the mask.
            width (int): The width of the mask.

        Returns:
            np.ndarray: A random mask of shape (height, width).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    num_circles = np.random.randint(5, 15)  # Random number of circles

    for _ in range(num_circles):
        radius = np.random.randint(5, min(height, width) // 4)  # Random radius
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        y, x = np.ogrid[:height, :width]
        circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        mask[circle] = 1

    return mask


class MockupSegmentationModel(SegmentationBaseModel):
    def prepare_input(self):
        """ Does not do anything and returns None.
        """
        return None

    def segment(self):
        """ Returns random masks and scores.
        """
        masks = [create_random_mask(original_height_width[0], original_height_width[1]) for _ in range(3)]
        scores = np.random.rand(len(masks))
        return np.array(masks), scores

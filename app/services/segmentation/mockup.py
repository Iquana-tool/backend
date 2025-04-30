import numpy as np
from app.services.segmentation import SegmentationBaseModel
from app.services.database_access import get_height_width_of_image
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
    num_circles = np.random.randint(1, 5)  # Random number of circles

    for _ in range(num_circles):
        radius = np.random.randint(5, min(height, width) // 4)  # Random radius
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        y, x = np.ogrid[:height, :width]
        circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        mask[circle] = 1
    return mask


class MockupSegmentationModel(SegmentationBaseModel):
    model_name = "mockup_model"

    def prepare_input(self, **kwargs):
        """ Does not do anything and returns None.
        """

        shape = get_height_width_of_image(kwargs["image_id"])
        if kwargs["min_x"] > 0 or kwargs["min_y"] > 0 or kwargs["max_x"] < 1 or kwargs["max_y"] < 1:
            scale_x = kwargs["max_x"] - kwargs["min_x"]
            scale_y = kwargs["max_y"] - kwargs["min_y"]
        else:
            scale_x = 1
            scale_y = 1
        return {"height": int(shape[0] * scale_y), "width": int(shape[1] * scale_x)}

    def segment(self, height, width, **kwargs):
        """ Returns random masks and scores.
        """
        masks = [create_random_mask(height, width) for _ in range(3)]
        scores = np.random.rand(len(masks))
        return np.array(masks), scores

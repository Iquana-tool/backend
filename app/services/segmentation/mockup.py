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
    def embed_image(self, image: np.ndarray) -> dict[str, Union[np.ndarray, list[np.ndarray]]]:
        """ Compute embeddings for image.
            Args:
                image (Union[np.ndarray, Image.Image]): The image to embed.

            Returns:
                dict[str, torch.Tensor]: A dictionary containing the embeddings. The dict has two entries: 'image_embed'
                 and 'high_res_feats'.
        """
        return {}

    def segment_with_prompts(self,
                             embedding: dict,
                             original_height_width: tuple[int, int],
                             input_prompts: Prompts):
        """ Segment an image using prompts.
            Args:
                embedding: (dict[str, np.ndarray]): The embedding of the image to segment.
                original_height_width: (tuple[int, int]): The original height and width of the image.
                input_prompts (Prompts): The prompts to use for segmentation.

            Returns:
                A tuple containing a CxHxW array, where C is the number of masks, and an array of length C,
                 where each entry is the quality of the corresponding mask.
        """
        masks = [create_random_mask(original_height_width[0], original_height_width[1]) for _ in range(3)]
        scores = np.random.rand(len(masks))
        return np.array(masks), scores

    def segment_without_prompts(self, image: np.ndarray):
        """ Segment an image without prompts.
            Args:
                image (Union[np.ndarray, Image.Image]): The image to segment.

            Returns:
                An array containing the segmentation masks, and an array of the predicted iou scores.
        """
        masks = [create_random_mask(image.shape[0], image.shape[1]) for _ in range(3)]
        scores = np.random.rand(len(masks))
        return np.array(masks), scores

    def segment_stack(self, stack: np.ndarray, input_prompts: Prompts):
        raise NotImplementedError("This method is not implemented yet!")
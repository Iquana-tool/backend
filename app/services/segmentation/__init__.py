from app.services.database_access import get_meso_path
from app.services.segmentation.sam2 import SAM2
from app.services.prompts import Prompts
import numpy as np
from typing import Union
from PIL import Image
from torch import Tensor


_model = SAM2()  # Initialize the model on import.

def embed_image(image: Union[np.ndarray, Image.Image]) -> dict[str, Union[np.ndarray, list[np.ndarray]]]:
    """ Embed an image using the SAM2 model.
        Args:
            image (Union[np.ndarray, Image.Image]): The image to embed.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the embeddings. The dict has two entries: 'image_embed'
             and 'high_res_feats'.
    """
    return _model.embed_image(image)

def segment_with_prompts(
    embedding: dict[str, np.ndarray],
    original_height_width: tuple[int, int],
    input_prompts: Prompts
) -> tuple[np.ndarray, np.ndarray]:
    """ Segment an image using prompts.
        Args:
            embedding (dict[str, np.ndarray]): The embedding of the image to segment.
            original_height_width (tuple[int, int]): The original height and width of the image.
            input_prompts (Prompts): The prompts to use for segmentation.

        Returns:
            A tuple containing a CxHxW array, where C is the number of masks, and an array of length C,
             where each entry is the quality of the corresponding mask.
    """
    return _model.segment_with_prompts(embedding, original_height_width, input_prompts)

def segment_without_prompts(image: Union[np.ndarray, Image.Image]) -> tuple[np.ndarray, np.ndarray]:
    """ Segment an image without prompts.
        Args:
            image (Union[np.ndarray, Image.Image]): The image to segment.

        Returns:
            A tuple containing a CxHxW array, where C is the number of masks, and an array of length C,
             where each entry is the quality of the corresponding mask.
    """
    return _model.segment_without_prompts(image)

def segment_stack_with_prompts(
    embedding: dict[str, np.ndarray],
    input_prompts: Prompts
) -> tuple[np.ndarray, np.ndarray]:
    """ Segment a stack of images using prompts.
        Args:
            embedding (dict[str, np.ndarray]): The embedding of the image to segment.
            input_prompts (Prompts): The prompts to use for segmentation.

        Returns:
            A tuple containing a CxHxW array, where C is the number of masks, and an array of length C,
             where each entry is the quality of the corresponding mask.
    """
    raise NotImplementedError("Stack segmentation with prompts is not implemented yet.")

def segment_stack_without_prompts(image: Union[np.ndarray, Image.Image]) -> tuple[np.ndarray, np.ndarray]:
    """ Segment a stack of images without prompts.
        Args:
            image (Union[np.ndarray, Image.Image]): The image to segment.

        Returns:
            A tuple containing a CxHxW array, where C is the number of masks, and an array of length C,
             where each entry is the quality of the corresponding mask.
    """
    raise NotImplementedError("Stack segmentation without prompts is not implemented yet.")

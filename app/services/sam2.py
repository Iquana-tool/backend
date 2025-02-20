import torch
import numpy as np
import PIL.Image as Image
from sam2.build_sam import build_sam2 as build
from sam2.sam2_image_predictor import SAM2ImagePredictor, SAM2Base
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from typing import Union
from app.services.prompts import Prompts
from config import ModelConfig


# Base classes for segmentation
def assert_nd_array(image: Union[Image, np.ndarray]) -> np.ndarray:
    return np.array(image) if isinstance(image, type(Image)) else image


class SAM2:
    def __init__(self, device='auto'):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build(ckpt_path=ModelConfig.get_active_model_config().weights,
                           config_file=ModelConfig.get_active_model_config().config,
                           device=self.device)
        self.model_name = ModelConfig.get_active_model_config().__name__
        self.prompt_predictor = SAM2ImagePredictor(self.model)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)

    def embed_image(self, image: Union[np.ndarray, Image.Image]) -> dict[str, torch.Tensor]:
        """ Compute embeddings for image.
            Args:
                image (Union[np.ndarray, Image.Image]): The image to embed.

            Returns:
                dict[str, torch.Tensor]: A dictionary containing the embeddings. The dict has two entries: 'image_embed'
                 and 'high_res_feats'.
        """
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.prompt_predictor.set_image(assert_nd_array(image))
            return {key: value.cpu().detach().numpy() for key, value in self.prompt_predictor._features}

    def segment_prompts(self,
                        embedding: dict[str, np.ndarray],
                        input_prompts: Prompts):
        """ Segment an image using prompts.
            Args:
                embedding: (dict[str, np.ndarray]): The embedding of the image to segment.
                input_prompts (Prompts): The prompts to use for segmentation.

            Returns:
                A tuple containing a CxHxW array, where C is the number of masks, and an array of length C,
                 where each entry is the quality of the corresponding mask.
        """
        embedding = {key: torch.tensor(value, device=self.device) for key, value in embedding.items()}
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.prompt_predictor._features = embedding
            self.prompt_predictor._is_image_set = True
            masks, quality, _ = self.prompt_predictor.predict(**input_prompts.to_SAM2_input())
        return masks, quality

    def segment_without_prompts(self, image: Union[np.ndarray, Image.Image]):
        """ Segment an image without prompts.
            Args:
                image (Union[np.ndarray, Image.Image]): The image to segment.

            Returns:
                list(dict(str, any)): A list over records for masks. Each record is
                 a dict containing the following keys:
                   segmentation (dict(str, any) or np.ndarray): The mask. If
                     output_mode='binary_mask', is an array of shape HW. Otherwise,
                     is a dictionary containing the RLE.
                   bbox (list(float)): The box around the mask, in XYWH format.
                   area (int): The area in pixels of the mask.
                   predicted_iou (float): The model's own prediction of the mask's
                     quality. This is filtered by the pred_iou_thresh parameter.
                   point_coords (list(list(float))): The point coordinates input
                     to the model to generate this mask.
                   stability_score (float): A measure of the mask's quality. This
                     is filtered on using the stability_score_thresh parameter.
                   crop_box (list(float)): The crop of the image used to generate
                     the mask, given in XYWH format.
        """
        return self.mask_generator.generate(assert_nd_array(image))

    def segment_stack(self, stack: np.ndarray, input_prompts: Prompts):
        raise NotImplementedError("This method is not implemented yet!")

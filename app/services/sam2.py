import torch
import numpy as np
import PIL.Image as Image
from sam2.build_sam import build_sam2 as build
from sam2.sam2_image_predictor import SAM2ImagePredictor, SAM2Base
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_video_predictor import SAM2VideoPredictor
from typing import Union
from prompts import Prompts
from app.config import ModelConfig

# Base classes for segmentation


class ImageSegmenter:
    def __init__(self, device='auto'):
        self.device = device if device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model: SAM2Base = None
        self.prompt_predictor: SAM2ImagePredictor = None
        self.mask_generator: SAM2AutomaticMaskGenerator = None

    def segment_prompts(self,
                        image: Union[np.ndarray, Image.Image],
                        input_prompts: Prompts):
        """ Segment an image using prompts.
            Args:
                image (Union[np.ndarray, Image.Image]): The image to segment.
                input_prompts (Prompts): The prompts to use for segmentation.

            Returns:
                A tuple containing a CxHxW array, where C is the number of masks, and an array of length C,
                 where each entry is the quality of the corresponding mask.
        """
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.prompt_predictor.set_image(image)
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
        return self.mask_generator.generate(image)


class StackSegmenter:
    """ TODO: Implement this class """
    pass


# Use case specific models


class MesoScaleImageSegmenter(ImageSegmenter):
    """ A class for segmenting coral images at the mesoscale. In the future this class should have finetuned models
        for the mesoscale. """
    def __init__(self, device='auto'):
        super().__init__(device)
        self.model = build(ModelConfig.MesoModel.weights, ModelConfig.MesoModel.config, self.device)
        self.prompt_predictor = SAM2ImagePredictor(self.model)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)

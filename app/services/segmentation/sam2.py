import torch
import numpy as np
from PIL.Image import Image  
from sam2.build_sam import build_sam2 as build
from sam2.sam2_image_predictor import SAM2ImagePredictor, SAM2Base
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from typing import Union
from app.services.prompts import Prompts
from config import ModelConfig, SAM2Config


class SAM2:
    def __init__(self, model_config: SAM2Config, device='auto'):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build(ckpt_path=model_config.weights,
                           config_file=model_config.config,
                           device=self.device)
        self.model_name = model_config.__name__
        self.prompt_predictor = SAM2ImagePredictor(self.model)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)

    def embed_image(self, image: np.ndarray) -> dict[str, Union[np.ndarray, list[np.ndarray]]]:
        """ Compute embeddings for image.
            Args:
                image (Union[np.ndarray, Image.Image]): The image to embed.

            Returns:
                dict[str, torch.Tensor]: A dictionary containing the embeddings. The dict has two entries: 'image_embed'
                 and 'high_res_feats'.
        """
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.prompt_predictor.set_image(image)
            return {"image_embed": self.prompt_predictor._features['image_embed'].cpu().detach().numpy(),
                    "high_res_feats": [feat.float().cpu().detach().numpy() for feat in self.prompt_predictor._features['high_res_feats']]}

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
        embedding["image_embed"] = torch.from_numpy(embedding["image_embed"]).to(self.device)
        embedding["high_res_feats"] = [torch.from_numpy(feat).to(self.device) for feat in embedding["high_res_feats"]]
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.prompt_predictor._features = embedding  # Sets the embedding
            self.prompt_predictor._is_image_set = True
            self.prompt_predictor._orig_hw = [original_height_width]
            masks, quality, _ = self.prompt_predictor.predict(**input_prompts.to_SAM2_input())
        return masks, quality

    def segment_without_prompts(self, image: np.ndarray):
        """ Segment an image without prompts.
            Args:
                image (Union[np.ndarray, Image.Image]): The image to segment.

            Returns:
                An array containing the segmentation masks, and an array of the predicted iou scores.
        """
        masks, scores = [], []
        for entry in self.mask_generator.generate(image):
            masks.append(entry['segmentation'])
            scores.append(entry['predicted_iou'])
        return np.array(masks), np.array(scores)

    def segment_stack(self, stack: np.ndarray, input_prompts: Prompts):
        raise NotImplementedError("This method is not implemented yet!")

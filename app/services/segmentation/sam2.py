import logging
import os
import urllib

import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2 as build, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
import config
from app.services.prompts import Prompts
from app.services.segmentation.base_model import ScanSegmentationBaseModel
from app.services.database_access import load_image_as_array_from_disk
from app.schemas.segmentation_and_masks import PromptedSegmentationRequest, AutomaticSegmentationRequest
from app.services.cropping import crop_image
from config import SAM2Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def download_checkpoint(ckpt_path: str) -> int:
    """
    Downloads a checkpoint file from a predefined base URL if it does not already exist locally.

    Args:
        ckpt_path (str): The local file path where the checkpoint should be saved. If the file
                         already exists at this path, no download is performed.

    Returns:
        int: A status code indicating the outcome:
             - 0: File already exists.
             - 1: File successfully downloaded.

    Raises:
        ValueError: If the checkpoint URL cannot be constructed or is invalid.
        urllib.error.URLError: If the download fails due to a network issue or invalid URL.
    """
    # Check if the checkpoint file already exists locally
    if os.path.exists(ckpt_path):
        logger.info(f"Checkpoint file already exists at {ckpt_path}. Skipping download.")
        return 0

    try:
        # Construct the full URL for the checkpoint file
        ckpt_filename = os.path.basename(ckpt_path)  # Extract the filename from the path
        ckpt_url = f"{config.Paths.SAM2p1_BASE_URL.rstrip('/')}/{ckpt_filename}"

        # Download the checkpoint file
        logger.info(f"Downloading checkpoint from {ckpt_url} to {ckpt_path}...")
        urllib.request.urlretrieve(ckpt_url, ckpt_path)
        logger.info(f"Checkpoint successfully downloaded to {ckpt_path}.")

        return 1
    except ValueError as e:
        logger.error(f"Error constructing checkpoint URL: {e}")
        raise
    except urllib.error.URLError as e:
        logger.error(f"Failed to download checkpoint: {e}")
        raise


class SAM2(ScanSegmentationBaseModel):
    def __init__(self, model_config: SAM2Config, device='auto'):
        """ Initialize the SAM2 model.
            Args:
                model_config (config.SAM2Config): The configuration for the SAM2 model.
                device (str): The device to run the model on. Can be 'cpu', 'cuda', or 'auto'.
        """
        super().__init__()
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        download_checkpoint(ckpt_path=model_config.weights)
        self.config = model_config
        self.model = build(ckpt_path=model_config.weights,
                           config_file=model_config.config,
                           device=self.device)
        self.model_name = model_config.__name__
        self.prompt_predictor = SAM2ImagePredictor(self.model)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model, multimask_output=False)
        self.stack_predictor: SAM2VideoPredictor = build_sam2_video_predictor(self.config.config, self.config.weights, self.device)
        self.set_image_id = None

    def process_prompted_request(self, request: PromptedSegmentationRequest) -> tuple[np.ndarray, np.ndarray]:
        """ Process the segmentation request.
            Args:
                request (PromptedSegmentationRequest): The segmentation request containing the image and prompts.

            Returns:
                tuple: A tuple containing an array of masks and an array of predicted iou scores.
        """
        # Check if cropping is needed
        use_crop = request.min_x > 0 or request.min_y > 0 or request.max_x < 1 or request.max_y < 1
        # If we do not have a crop, we can load the embedding directly
        logger.info("Starting segmentation...")
        prompts = Prompts()
        prompts.from_segmentation_request(request)
        request_unique_id = f"{request.image_id}_{request.min_x}_{request.min_y}_{request.max_x}_{request.max_y}"
        if use_crop or (request_unique_id != self.set_image_id):
            # If cropping is needed or the image_id has changed, we need to set the image
            # Temporary fix for embedding loading
            image = load_image_as_array_from_disk(request.image_id)
            image = crop_image(request.min_x, request.min_y,
                               request.max_x, request.max_y,
                               image)
            self.prompt_predictor.set_image(image)
            self.set_image_id = request_unique_id
        mask, scores, _ = self.prompt_predictor.predict(**prompts.to_SAM2_input(),
                                                        multimask_output=False,
                                                        normalize_coords=False)
        return mask, scores

    def process_automatic_request(self, request: AutomaticSegmentationRequest) -> tuple[np.ndarray, np.ndarray]:
        image = load_image_as_array_from_disk(request.image_id)
        # Check if cropping is needed
        use_crop = request.min_x > 0 or request.min_y > 0 or request.max_x < 1 or request.max_y < 1
        if use_crop or (request.image_id != self.set_image_id):
            # If cropping is needed or the image_id has changed, we need to set the image
            image = crop_image(request.min_x, request.min_y,
                               request.max_x, request.max_y,
                               image)
        result = self.mask_generator.generate(image)
        masks = np.array([mask['segmentation'] for mask in result])
        scores = np.array([mask['stability_score'] for mask in result])
        return masks, scores

    def propagate_mask(self, **kwargs) -> tuple[list, list]:
        """ Propagate the mask across the scan.
            This method should be overridden by subclasses to provide model-specific mask propagation logic.
        """
        self.stack_predictor.reset_state()


class SAM2Tiny(SAM2):
    def __init__(self):
        """ Initialize the SAM2 Tiny model. """
        super().__init__(model_config=SAM2Config.SAM2Tiny(), device='auto')
        self.model_name = 'SAM2Tiny'

class SAM2Small(SAM2):
    def __init__(self):
        """ Initialize the SAM2 Small model. """
        super().__init__(model_config=SAM2Config.SAM2Small(), device='auto')
        self.model_name = 'SAM2Small'


class SAM2Large(SAM2):
    def __init__(self):
        """ Initialize the SAM2 Large model. """
        super().__init__(model_config=SAM2Config.SAM2Large(), device='auto')
        self.model_name = 'SAM2Large'


class SAM2BasePlus(SAM2):
    def __init__(self):
        """ Initialize the SAM2 Base Plus model. """
        super().__init__(model_config=SAM2Config.SAM2BasePlus(), device='auto')
        self.model_name = 'SAM2BasePlus'

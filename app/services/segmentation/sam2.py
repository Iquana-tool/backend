from logging import getLogger
from app.services.logging import log_execution_time
import os
import urllib

import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2 as build, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from app.services.prompts import Prompts
from app.services.segmentation.base_model import *
from app.services.database_access import load_image_as_array_from_disk, get_scan_image_folder_path, get_image_query
from app.schemas.segmentation.segmentations import PromptedSegmentationRequest, AutomaticSegmentationRequest
from app.services.cropping import crop_image

logger = getLogger(__name__)


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
        ckpt_url = f"{'https://dl.fbaipublicfiles.com/segment_anything_2/092824'.rstrip('/')}/{ckpt_filename}"

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


class SAM2Base(SegmentationBaseModel):
    """ Base class for SAM2 models. This class should not be instantiated directly. """
    def __init__(self, path_to_weights, path_to_config, device='auto'):
        self.model = None
        self.model_name = "SAM2"
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = path_to_config
        self.weights = path_to_weights
        download_checkpoint(ckpt_path=path_to_weights)
        self.set_image_id = None  # To track the current image being processed


class SAM2Prompted(SAM2Base, PromptedSegmentationBaseModel):
    def __init__(self,  path_to_weights, path_to_config, device='auto'):
        """ Initialize the prompted SAM2 model.
            Args:
                path_to_weights (str): Path to the model weights file.
                path_to_config (str): Path to the model configuration file.
                device (str): The device to run the model on. Can be 'cpu', 'cuda', or 'auto'.
        """
        super().__init__(path_to_weights, path_to_config, device)
        self.model = build(ckpt_path=self.weights,
                           config_file=self.config,
                           device=self.device)
        self.prompt_predictor = SAM2ImagePredictor(self.model)

    @log_execution_time
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


class SAM2Automatic(SAM2Base, AutomaticSegmentationBaseModel):
    def __init__(self, path_to_weights, path_to_config, device='auto'):
        """ Initialize the automatic SAM2 model.
            Args:
                path_to_weights (str): Path to the model weights file.
                path_to_config (str): Path to the model configuration file.
                device (str): The device to run the model on. Can be 'cpu', 'cuda', or 'auto'.
        """
        super().__init__(path_to_weights, path_to_config, device)
        self.model = build(ckpt_path=self.weights,
                           config_file=self.config,
                           device=self.device)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model, multimask_output=False)

    @log_execution_time
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


class SAM2Prompted3D(SAM2Prompted, PromptedSegmentation3DBaseModel):
    def __init__(self, path_to_weights, path_to_config, device='auto'):
        """ Initialize the SAM2 model.
            Args:
                path_to_weights (str): Path to the model weights file.
                path_to_config (str): Path to the model configuration file.
                device (str): The device to run the model on. Can be 'cpu', 'cuda', or 'auto'.
        """
        super().__init__(path_to_weights, path_to_config, device)
        self.stack_predictor: SAM2VideoPredictor = build_sam2_video_predictor(self.config,
                                                                              self.weights,
                                                                              self.device)
        self.init_state = None

    @log_execution_time
    def set_scan(self, scan_id: int):
        """ Set the scan for the model.
            Args:
                scan_id (int): The ID of the scan to set.
        """
        if not (scan_id == self.set_image_id or self.set_image_id is None):
            # If the scan_id has changed, we need to reset the state
            self.stack_predictor.init_state()
            self.set_image_id = scan_id
            self.init_state = self.stack_predictor.init_state(get_scan_image_folder_path(scan_id))
        else:
            # If the scan_id has not changed, we can reuse the state
            self.stack_predictor.reset_state(self.init_state)

    @log_execution_time
    def add_slice_prompt(self, request: PromptedSegmentationRequest, object_id: int = 1):
        """ Add a slice prompt to the stack predictor.
            Args:
                request (PromptedSegmentationRequest): The segmentation request containing the image and prompts.
        """
        prompts = Prompts()
        prompts.from_segmentation_request(request)
        image = get_image_query(request.image_id)
        if image is None:
            raise ValueError(f"Image with ID {request.image_id} not found in database.")
        if image.scan_id !=  self.set_image_id:
            raise ValueError(f"Image with ID {request.image_id} does not belong "
                             f"to the current scan {self.set_image_id}.")
        prompts.to_SAM2_input()
        _, out_obj_ids, out_mask_logits = self.stack_predictor.add_new_points_or_box(
            inference_state=self.init_state,
            frame_idx=image.index_in_scan,
            obj_id=object_id,
            points=prompts.point_prompts,
            labels=[int(label) for label in prompts.point_labels],
            box=prompts.box_prompts,
        )

    @log_execution_time
    def process_prompted_segmentation_3D_request(self, request: ScanPromptedSegmentationRequest) -> dict:
        """ Propagate the mask across the scan.
            Args:
                request (ScanPromptedSegmentationRequest): The segmentation request containing the scan and prompts.

            Returns:
                tuple: A tuple containing an array of masks and an array of predicted iou scores.
        """
        # FIXME: This method is not correctly implemented. It only tracks one object.
        self.set_scan(request.scan_id)
        for prompt_request in request.prompted_requests:
            self.add_slice_prompt(prompt_request)
        video_segments = {}
        for idx, obj_ids, mask_logits in self.stack_predictor.propagate_in_video(self.init_state):
            video_segments[idx] = {
                out_obj_id: (mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(obj_ids)
            }
        return video_segments

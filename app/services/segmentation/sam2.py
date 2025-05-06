import logging
import os
import urllib
from typing import Union

import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2 as build
from sam2.sam2_image_predictor import SAM2ImagePredictor
from app.database import get_context_session
from app.database.images import ImageEmbeddings
import config
from app.services.prompts import Prompts
from app.services.segmentation import SegmentationBaseModel
from app.services.database_access import load_image_as_array_from_disk, save_embedding, get_height_width_of_image
from config import SAM2Config
from app.schemas.segmentation_and_masks import SegmentationRequest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Global variable to store the current image_id being processed
_current_image_id = None


def set_current_image_id(image_id):
    """Set the current image_id being processed"""
    global _current_image_id
    _current_image_id = image_id


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


class SAM2(SegmentationBaseModel):
    def __init__(self, model_config: SAM2Config, device='auto'):
        """ Initialize the SAM2 model.
            Args:
                model_config (SAM2Config): The configuration for the SAM2 model.
                device (str): The device to run the model on. Can be 'cpu', 'cuda', or 'auto'.
        """
        super().__init__()
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        download_checkpoint(ckpt_path=model_config.weights)
        self.model = build(ckpt_path=model_config.weights,
                           config_file=model_config.config,
                           device=self.device)
        self.model_name = model_config.__name__
        self.prompt_predictor = SAM2ImagePredictor(self.model)
        self.mask_generator = SAM2AutomaticMaskGenerator(self.model)

    def process_request(self, request: SegmentationRequest) -> tuple[np.ndarray, np.ndarray]:
        """ Process the segmentation request.
            Args:
                request (SegmentationRequest): The segmentation request containing the image and prompts.

            Returns:
                tuple: A tuple containing an array of masks and an array of predicted iou scores.
        """
        # Check if cropping is needed
        use_crop = request.min_x > 0 or request.min_y > 0 or request.max_x < 1 or request.max_y < 1
        # If we do not have a crop, we can load the embedding directly
        embedding = self.load_embedding(request.image_id)
        if True and embedding is None or use_crop:
            # If we do not have an embedding or we have a crop, we need to load the image and embed it
            image = load_image_as_array_from_disk(request.image_id,
                                                  request.min_x, request.min_y,
                                                  request.max_x, request.max_y)
            embedding = self.embed_image(image)
            if False and not use_crop:
                # Save the embedding for the full image
                save_embedding(request, embedding)
        # Save the original height and width of the image
        height, width = get_height_width_of_image(request.image_id)
        if use_crop:
            # Save the new height and width of the image after cropping
            width = int((request.max_x - request.min_x) * width)
            height = int((request.max_y - request.min_y) * height)
        logger.info("Starting segmentation...")
        if request.use_prompts:
            prompts = Prompts()
            prompts.from_segmentation_request(request)

            # Temporary fix for embedding loading
            image = load_image_as_array_from_disk(request.image_id)
            self.prompt_predictor.set_image(image)
            mask, scores, _ = self.prompt_predictor.predict(**prompts.to_SAM2_input(),normalize_coords=False)
            return mask, scores
            # Fix end

            # return self.segment_with_prompts(embedding, (width, height), prompts)
        else:
            image = load_image_as_array_from_disk(request.image_id)
            return self.segment_without_prompts(image)

    def load_embedding(self, image_id: int):
        """Load an image embedding from the database by its embedding ID."""
        with get_context_session() as session:
            embedding = session.query(ImageEmbeddings).filter_by(image_id=image_id, model=self.model_name).first()
        if embedding:
            try:
                loaded_data = np.load(os.path.join(config.Paths.embedding_dir,
                                                   str(embedding.image_id),
                                                   self.model_name + ".npz"))
                files = set(loaded_data.files)
                new_dict = {"image_embed": loaded_data["image_embed"]}
                files.remove("image_embed")
                new_dict["high_res_feats"] = [loaded_data[high_res_feat] for high_res_feat in files]
                logger.info(f"Loaded embedding for image ID {image_id} for model {self.model_name}.")
                return new_dict
            except FileNotFoundError:
                logger.warning(f"File not found for embedding ID {embedding.id}. "
                               f"Path: {os.path.join(config.Paths.embedding_dir, str(embedding.image_id), self.model_name + '.npz')}")
                return None
        else:
            logger.info(f"No embedding found for image ID {image_id} for model {self.model_name}.")
            return None

    def prepare_input(self, **kwargs):
        """ Prepare the input for the model.
            Args:
                kwargs: The keyword arguments containing the image to segment.

            Returns:
                dict: A dictionary containing the image to segment.
        """
        pass

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
                    "high_res_feats": [feat.float().cpu().detach().numpy() for feat in
                                       self.prompt_predictor._features['high_res_feats']]}

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
        # Load the embeddings onto the device
        embedding["image_embed"] = torch.from_numpy(embedding["image_embed"]).to(self.device)
        embedding["high_res_feats"] = [torch.from_numpy(feat).to(self.device) for feat in
                                       embedding["high_res_feats"]]

        # Set up the predictor with the embeddings
        # Hacky solution here. There is some mismatch between the height and the width and im not entirely sure, how
        # to resolve it.
        try:
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                self.prompt_predictor._features = embedding  # Sets the embedding
                self.prompt_predictor._is_image_set = True
                self.prompt_predictor._orig_hw = [original_height_width]
                masks, quality, _ = self.prompt_predictor.predict(**input_prompts.to_SAM2_input(),
                                                                  normalize_coords=False)
        except RuntimeError:
            logger.warning("RuntimeError: Mismatch between height and width. Trying to fix it.")
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                self.prompt_predictor._features = embedding  # Sets the embedding
                self.prompt_predictor._is_image_set = True
                self.prompt_predictor._orig_hw = [original_height_width[::-1]]
                masks, quality, _ = self.prompt_predictor.predict(**input_prompts.to_SAM2_input(),
                                                                  normalize_coords=False)
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

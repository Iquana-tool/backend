import config
from logging import getLogger
from app.services.segmentation.mockup import MockupSegmentationModel
from app.services.segmentation.sam2 import SAM2


logger = getLogger(__name__)


class SegmentationBaseModel:
    """Base class for segmentation models.
    """
    def __init__(self):
        self.model = None

    def embed_image(self, image):
        """Embed the image using the model.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def segment_with_prompts(self, embedding, size, prompts):
        """Segment the image using prompts.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def segment_without_prompts(self, image):
        """Segment the image without prompts.
        """
        raise NotImplementedError("Subclasses should implement this method.")


def get_model_via_identifier(identifier: str):
    """
    Returns the model class based on the identifier.
    """
    if identifier == "Mockup":
        logger.debug("Using Mockup segmentation model.")
        return MockupSegmentationModel()
    elif identifier == "SAM2Tiny":
        logger.debug("Using SAM2Tiny segmentation model.")
        return SAM2(config.ModelConfig.available_models["SAM2Tiny"]())
    elif identifier == "SAM2Small":
        logger.debug("Using SAM2Small segmentation model.")
        return SAM2(config.ModelConfig.available_models["SAM2Small"]())
    elif identifier == "SAM2Large":
        logger.debug("Using SAM2Large segmentation model.")
        return SAM2(config.ModelConfig.available_models["SAM2Large"]())
    elif identifier == "SAM2BasePlus":
        logger.debug("Using SAM2BasePlus segmentation model.")
        return SAM2(config.ModelConfig.available_models["SAM2BasePlus"]())
    else:
        logger.error(f"Unknown model identifier: {identifier}")
        raise ValueError(f"Unknown model identifier: {identifier}")

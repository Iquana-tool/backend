import config
from logging import getLogger
from app.services.segmentation.base_model import SegmentationBaseModel
from app.services.segmentation.mockup import MockupSegmentationModel
from app.services.segmentation.sam2 import SAM2

logger = getLogger(__name__)


def get_model_via_identifier(identifier: str) -> SegmentationBaseModel:
    """
    Returns the model class based on the identifier.
    """
    if identifier == "Mockup":
        logger.debug("Using Mockup segmentation model.")
        return MockupSegmentationModel()
    elif identifier == "SAM2Tiny":
        logger.debug("Using SAM2Tiny segmentation model.")
        return SAM2(config.PromptedSegmentationModelsConfig.available_models["SAM2Tiny"]())
    elif identifier == "SAM2Small":
        logger.debug("Using SAM2Small segmentation model.")
        return SAM2(config.PromptedSegmentationModelsConfig.available_models["SAM2Small"]())
    elif identifier == "SAM2Large":
        logger.debug("Using SAM2Large segmentation model.")
        return SAM2(config.PromptedSegmentationModelsConfig.available_models["SAM2Large"]())
    elif identifier == "SAM2BasePlus":
        logger.debug("Using SAM2BasePlus segmentation model.")
        return SAM2(config.PromptedSegmentationModelsConfig.available_models["SAM2BasePlus"]())
    else:
        logger.error(f"Unknown model identifier: {identifier}")
        raise ValueError(f"Unknown model identifier: {identifier}")

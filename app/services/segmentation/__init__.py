import config
from logging import getLogger
from app.services.segmentation.mockup import MockupSegmentationModel
from app.services.segmentation.sam2 import SAM2
from app.schemas.segmentation_and_masks import SegmentationRequest
from sqlalchemy.orm import Session


logger = getLogger(__name__)


class SegmentationBaseModel:
    """Base class for segmentation models.
    """
    def __init__(self):
        self.model = None

    def process_request(self, request: SegmentationRequest) -> tuple[list, list]:
        """Process the segmentation request. This function calls the prepare_input and segment methods sequentially.
        This allows for individual preparation methods for each model. Additionally, the segment method can be
        overridden in subclasses to provide model-specific segmentation logic like prompted or unprompted segmentation.
        Each model can also override this method to allow for even more flexibility.
        """
        logger.debug(f"Processing segmentation request: {request}")
        return self.segment(**self.prepare_input(**request.dict()))

    def prepare_input(self, **kwargs):
        """Prepare the image for the model.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def segment(self, **kwargs) -> tuple[list, list]:
        """Segment the given input.
        """
        raise NotImplementedError("Subclasses should implement this method.")


def get_model_via_identifier(identifier: str) -> SegmentationBaseModel:
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

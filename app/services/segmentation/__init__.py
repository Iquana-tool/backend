import config
from logging import getLogger
from app.services.segmentation.base_model import SegmentationBaseModel
from app.services.segmentation.mockup import MockupSegmentationModel
from app.services.segmentation.sam2 import SAM2

logger = getLogger(__name__)


class ModelCache:
    def __init__(self, identifier_to_model_func: dict[str, callable]):
        """
        Initializes the ModelCache service which manages segmentation models.
        """
        logger.debug("PromptedModel service initialized.")
        self.model = None
        self.set_model_identifier = None
        self.identifier_to_model_func = identifier_to_model_func

    def set_model(self, identifier: str):
        """
        Sets the segmentation model based on the provided identifier.

        Args:
            identifier (str): The identifier for the segmentation model.

        Raises:
            ValueError: If the identifier does not match any known model.
        """
        if self.set_model_identifier is None or self.set_model_identifier != identifier:
            logger.debug(f"Model identifier changed from {self.set_model_identifier} to {identifier}.")
            self.set_model_identifier = identifier
            self.model = self.identifier_to_model_func[identifier]()
        else:
            logger.debug(f"Model identifier remains unchanged: {self.set_model_identifier}.")

    def get_model(self) -> SegmentationBaseModel:
        """
        Returns the currently set segmentation model.

        Returns:
            SegmentationBaseModel: The currently set segmentation model.

        Raises:
            ValueError: If no model has been set.
        """
        if self.model is None:
            logger.error("No model has been set.")
            raise ValueError("No model has been set.")
        return self.model

    def set_and_get_model(self, identifier: str) -> SegmentationBaseModel:
        """
        Sets the segmentation model based on the identifier and returns it.

        Args:
            identifier (str): The identifier for the segmentation model.

        Returns:
            SegmentationBaseModel: The segmentation model set based on the identifier.

        Raises:
            ValueError: If the identifier does not match any known model.
        """
        self.set_model_identifier(identifier)
        return self.get_model()

def get_prompted_model_via_identifier(self, identifier: str) -> SegmentationBaseModel:
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

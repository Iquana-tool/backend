from logging import getLogger
from app.services.segmentation.base_model import SegmentationBaseModel
from app.database.models import Models
from app.services.segmentation.mockup import MockupSegmentationModel
from app.services.logging import log_execution_time
from app.database import get_context_session
from configs.available_models import AvailableModels

logger = getLogger(__name__)


class ModelCache:
    def __init__(self):
        """
        Initializes the ModelCache service which manages segmentation models.
        """
        logger.debug("PromptedModel service initialized.")
        self.model = None
        self.set_model_identifier = None

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
            with get_context_session() as session:
                model_class = session.query(Models).filter_by(identifier=identifier).first()
                if model_class is None:
                    logger.error(f"Model with identifier '{identifier}' not found.")
                    raise ValueError(f"Model with identifier '{identifier}' not found.")
                self.model = AvailableModels[model_class.model_type][model_class.base_model_identifier](
                    model_class.weights, model_class.config
                )
                logger.debug(f"Model set to: {model_class.name} with identifier {identifier}.")
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

    @log_execution_time
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
        self.set_model(identifier)
        return self.get_model()

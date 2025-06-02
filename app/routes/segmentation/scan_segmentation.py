import logging

from fastapi import APIRouter

from app.services.segmentation import MockupSegmentationModel, ModelCache
from app.services.segmentation.sam2 import SAM2Tiny, SAM2Small, SAM2Large, SAM2BasePlus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/segmentation", tags=["segmentation"])


class PromptedSegmentationModelsConfig:
    """ This class contains the configuration options for the model. """
    selected_model = 'SAM2Tiny'
    available_models = {
        'Mockup': MockupSegmentationModel,
        'SAM2Tiny': SAM2Tiny,
        'SAM2Small': SAM2Small,
        'SAM2Large': SAM2Large,
        'SAM2BasePlus': SAM2BasePlus
    }


prompted_model_cache = ModelCache(PromptedSegmentationModelsConfig.available_models)

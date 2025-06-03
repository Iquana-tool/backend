import logging

from fastapi import APIRouter

from app.services.segmentation import MockupSegmentationModel, ModelCache
from app.services.segmentation.sam2 import SAM2Prompted3D
from config import SAM2TinyConfig, SAM2SmallConfig, SAM2LargeConfig, SAM2BasePlusConfig


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/segmentation", tags=["segmentation"])


class ScanPromptedSegmentationModelsConfig:
    """ This class contains the configuration options for the model. """
    selected_model = 'SAM2Tiny'
    available_models = {
        'SAM2Tiny': SAM2Prompted3D(SAM2TinyConfig),
        'SAM2Small': SAM2Prompted3D(SAM2SmallConfig),
        'SAM2Large': SAM2Prompted3D(SAM2LargeConfig),
        'SAM2BasePlus': SAM2Prompted3D(SAM2BasePlusConfig),
    }


prompted_model_cache = ModelCache(ScanPromptedSegmentationModelsConfig.available_models)

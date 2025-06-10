import logging

from fastapi import APIRouter

from app.services.segmentation import MockupSegmentationModel, ModelCache
from app.services.segmentation.base_model import PromptedSegmentation3DBaseModel, AutomaticSegmentation3DBaseModel
from app.schemas.segmentation.segmentations import ScanPromptedSegmentationRequest, SegmentationResponse
from app.services.segmentation.sam2 import SAM2Prompted3D
from config import SAM2TinyConfig, SAM2SmallConfig, SAM2LargeConfig, SAM2BasePlusConfig
from app.routes.segmentation.util import get_masks_responses

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/segmentation", tags=["segmentation"])


class ScanPromptedSegmentationModelsConfig:
    """ This class contains the configuration options for the model. """
    selected_model = 'SAM2Tiny'
    available_models = {
        'SAM2Tiny': (SAM2Prompted3D, SAM2TinyConfig),
        'SAM2Small': (SAM2Prompted3D, SAM2SmallConfig),
        'SAM2Large': (SAM2Prompted3D, SAM2LargeConfig),
        'SAM2BasePlus': (SAM2Prompted3D, SAM2BasePlusConfig),
    }


prompted_model_cache = ModelCache(ScanPromptedSegmentationModelsConfig.available_models)


@router.post('/segment_scan')
async def segment_scan(request: ScanPromptedSegmentationRequest):
    """Perform segmentation on a scan with optional prompts.

    This function handles the segmentation of scans based on the provided request.
    It validates the request, retrieves the appropriate model, and processes the scan.

    Args:
        request (ScanPromptedSegmentationRequest): The request object containing scan data and parameters.

    Returns:
        SegmentationResponse: The response object containing the segmentation results.
    """

    model: PromptedSegmentation3DBaseModel = prompted_model_cache.set_and_get_model(request.model)

    todo = model.process_prompted_segmentation_3D_request(request)
    # TODO: Implement the logic to handle the todo variable, which should contain the segmentation results.

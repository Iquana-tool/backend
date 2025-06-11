import logging

from fastapi import APIRouter

from app.services.segmentation import MockupSegmentationModel, ModelCache
from app.services.segmentation.base_model import PromptedSegmentation3DBaseModel, AutomaticSegmentation3DBaseModel
from app.schemas.segmentation.segmentations import ScanPromptedSegmentationRequest, SegmentationResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/segmentation", tags=["segmentation"])

prompted_model_cache = ModelCache()


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

import logging
from fastapi import APIRouter
from app.schemas.prompted_segmentation.segmentations import SegmentationResponse


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/prompted_segmentation", tags=["prompted_segmentation"])


@router.post('/segment_scan', deprecated=True, response_model=SegmentationResponse)
async def segment_scan(request):
    """Perform prompted_segmentation on a scan with optional prompts.

    This function handles the prompted_segmentation of scans based on the provided request.
    It validates the request, retrieves the appropriate model, and processes the scan.

    Args:
        request: The request object containing scan data and parameters.

    Returns:
        SegmentationResponse: The response object containing the prompted_segmentation results.
    """
    raise NotImplementedError("Scan prompted_segmentation is not yet implemented.")

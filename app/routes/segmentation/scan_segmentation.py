import logging

import numpy as np
from fastapi import APIRouter

from app.services.segmentation import ModelCache
from app.services.segmentation.base_model import PromptedSegmentation3DBaseModel, AutomaticSegmentation3DBaseModel
from app.schemas.segmentation.segmentations import ScanPromptedSegmentationRequest, SegmentationResponse
from app.routes.segmentation.util import get_masks_responses
from app.services.database_access import get_height_width_of_scan, get_image_id_via_scan_index

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
    height, width = get_height_width_of_scan(request.scan_id)
    object_id_to_label = {object_id: p_requests[0].label for object_id, p_requests in request.prompted_requests.items()}
    scan_segmentations = model.process_prompted_segmentation_3D_request(request)
    responses = []
    for idx, objects in scan_segmentations.items():
        # Convert the object masks to a multiclass mask
        objects = {object_id_to_label[object_id]: mask for object_id, mask in objects.items()}
        multiclass_mask = convert_to_multiclass_mask(objects, height, width)
        # Get contours and qualities for the masks
        contours_response = get_masks_responses([multiclass_mask], [1.0])
        image_id = get_image_id_via_scan_index(request.scan_id, idx, reset_index=True)
        responses.append(SegmentationResponse(
            masks=contours_response,
            image_id=image_id,
            model=request.model
        ))
    return {"success": True, "segmentations": responses}






def convert_to_multiclass_mask(object_mask_dict, height, width):
    """Convert a dictionary of object masks to a multiclass mask."""
    multiclass_mask = np.zeros((height, width), dtype=np.uint8)
    for label, mask in object_mask_dict.items():
        multiclass_mask[mask] = label
    return multiclass_mask

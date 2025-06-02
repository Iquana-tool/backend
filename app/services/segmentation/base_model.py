from app.schemas.segmentation.segmentations import PromptedSegmentationRequest, AutomaticSegmentationRequest


class SegmentationBaseModel:
    """Base class for segmentation models.
    """
    def __init__(self):
        self.model = None
        self.model_name = None

    def process_prompted_request(self, request: PromptedSegmentationRequest) -> tuple[list, list]:
        """Process the segmentation request. Each prompted segmentation model should implement this method.
        Args:
            request (PromptedSegmentationRequest): The segmentation request containing the image and prompts.
        Returns:
            tuple: A tuple containing an array of masks and an array of predicted iou scores. Each mask is a binary mask
            in HW format.
        """
        raise NotImplementedError("Subclasses should implement this method to process the segmentation request.")

    def process_automatic_request(self, request: AutomaticSegmentationRequest) -> tuple[list, list]:
        """Process the automatic segmentation request. Each automatic segmentation model should implement this method.
        Args:
            request (AutomaticSegmentationRequest): The segmentation request containing the image and parameters.
        Returns:
            tuple: A tuple containing an array of masks and an array of predicted iou scores. Each mask is a mask in HW
            format, where each pixel contains its label.
        """
        raise NotImplementedError("Subclasses should implement this method to process the automatic segmentation request.")


class ScanSegmentationBaseModel(SegmentationBaseModel):
    """Base class for segmentation models that handle scans.
    This class is intended for models that work with scans, such as SAM2.
    It provides a common interface for preparing input and segmenting scans.
    """
    def segment_scan(self, **kwargs) -> tuple[list, list]:
        """Segment the given scan input.
        This method should be overridden by subclasses to provide model-specific segmentation logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def propagate_mask(self, **kwargs) -> tuple[list, list]:
        """Propagate the mask across the scan.
        This method should be overridden by subclasses to provide model-specific mask propagation logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")

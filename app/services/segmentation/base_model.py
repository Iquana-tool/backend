from app.schemas.segmentation_and_masks import PromptedSegmentationRequest


class SegmentationBaseModel:
    """Base class for segmentation models.
    """
    def __init__(self):
        self.model = None
        self.model_name = None

    def process_prompted_request(self, request: PromptedSegmentationRequest) -> tuple[list, list]:
        """Process the segmentation request. This function calls the prepare_input and segment methods sequentially.
        This allows for individual preparation methods for each model. Additionally, the segment method can be
        overridden in subclasses to provide model-specific segmentation logic like prompted or unprompted segmentation.
        Each model can also override this method to allow for even more flexibility.
        """
        raise NotImplementedError("Subclasses should implement this method to process the segmentation request.")

    def process_automatic_request(self, **kwargs) -> tuple[list, list]:
        """Process the automatic segmentation request.
        This function calls the prepare_input and segment methods sequentially.
        It allows for individual preparation methods for each model.
        Each model can override this method to provide model-specific segmentation logic.
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

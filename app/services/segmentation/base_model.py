from app.schemas.segmentation_and_masks import SegmentationRequest


class SegmentationBaseModel:
    """Base class for segmentation models.
    """
    def __init__(self):
        self.model = None
        self.model_name = None

    def process_request(self, request: SegmentationRequest) -> tuple[list, list]:
        """Process the segmentation request. This function calls the prepare_input and segment methods sequentially.
        This allows for individual preparation methods for each model. Additionally, the segment method can be
        overridden in subclasses to provide model-specific segmentation logic like prompted or unprompted segmentation.
        Each model can also override this method to allow for even more flexibility.
        """
        return self.segment(**self.prepare_input(**request.dict()))

    def prepare_input(self, **kwargs):
        """Prepare the image for the model.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def segment(self, **kwargs) -> tuple[list, list]:
        """Segment the given input.
        """
        raise NotImplementedError("Subclasses should implement this method.")

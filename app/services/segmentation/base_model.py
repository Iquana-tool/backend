from app.schemas.segmentation.segmentations import (PromptedSegmentationRequest, AutomaticSegmentationRequest,
                                                    ScanPromptedSegmentationRequest, ScanAutomaticSegmentationRequest)
from abc import ABC, abstractmethod


class SegmentationBaseModel(ABC):
    """Base class for segmentation models.
    """
    def __init__(self):
        self.model = None
        self.model_name = None


class PromptedSegmentationBaseModel(ABC, SegmentationBaseModel):
    """Base class for prompted segmentation models.
    This class is intended for models that require user prompts to perform segmentation.
    It provides a common interface for processing prompted segmentation requests.
    """
    @abstractmethod
    def process_prompted_request(self, request: PromptedSegmentationRequest) -> tuple[list, list]:
        """Process the segmentation request. Each prompted segmentation model should implement this method.
        Args:
            request (PromptedSegmentationRequest): The segmentation request containing the image and prompts.
        Returns:
            tuple: A tuple containing an array of masks and an array of predicted iou scores. Each mask is a binary mask
            in HW format.
        """
        pass


class AutomaticSegmentationBaseModel(ABC, SegmentationBaseModel):
    """Base class for automatic segmentation models.
    This class is intended for models that perform automatic segmentation without user prompts.
    It provides a common interface for processing automatic segmentation requests.
    """
    @abstractmethod
    def process_automatic_request(self, request: AutomaticSegmentationRequest) -> tuple[list, list]:
        """Process the automatic segmentation request. Each automatic segmentation model should implement this method.
        Args:
            request (AutomaticSegmentationRequest): The segmentation request containing the image and parameters.
        Returns:
            tuple: A tuple containing an array of masks and an array of predicted iou scores. Each mask is a mask in HW
            format, where each pixel contains its label.
        """
        pass


class PromptedSegmentation3DBaseModel(ABC, SegmentationBaseModel):
    """Base class for segmentation models that handle scans.
    This class is intended for models that work with scans, such as SAM2.
    It provides a common interface for preparing input and segmenting scans.
    """
    @abstractmethod
    def process_prompted_segmentation_3D_request(self, request: ScanPromptedSegmentationRequest):
        """Propagate the mask across the scan.
        This method should be overridden by subclasses to provide model-specific mask propagation logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class AutomaticSegmentation3DBaseModel(ABC, SegmentationBaseModel):
    """Base class for automatic segmentation models that handle scans.
    This class is intended for models that work with scans, such as SAM2.
    It provides a common interface for preparing input and segmenting scans.
    """
    @abstractmethod
    def process_automatic_segmentation_3D_request(self, request: ScanAutomaticSegmentationRequest) -> tuple[list, list]:
        """Propagate the mask across the scan.
        This method should be overridden by subclasses to provide model-specific mask propagation logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")

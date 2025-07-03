from app.schemas.segmentation.segmentations import (PromptedSegmentationRequest, AutomaticSegmentationRequest,
                                                    ScanPromptedSegmentationRequest, ScanAutomaticSegmentationRequest)
from abc import ABC, abstractmethod


class SegmentationBaseModel(ABC):
    """Base class for prompted_segmentation models.
    """
    @abstractmethod
    def __init__(self, path_to_weights: str, path_to_config: str):
        """Initialize the prompted_segmentation model with weights and configuration. Each model must be initializable with
        exactly these two parameters.
        Args:
            path_to_weights (str): Path to the model weights file.
            path_to_config (str): Path to the model configuration file.
        """
        pass


class PromptedSegmentationBaseModel(SegmentationBaseModel):
    """Base class for prompted prompted_segmentation models.
    This class is intended for models that require user prompts to perform prompted_segmentation.
    It provides a common interface for processing prompted prompted_segmentation requests.
    """
    @abstractmethod
    def process_prompted_request(self, request: PromptedSegmentationRequest) -> tuple[list, list]:
        """Process the prompted_segmentation request. Each prompted prompted_segmentation model should implement this method.
        Args:
            request (PromptedSegmentationRequest): The prompted_segmentation request containing the image and prompts.
        Returns:
            tuple: A tuple containing an array of masks and an array of predicted iou scores. Each mask is a binary mask
            in HW format.
        """
        pass


class AutomaticSegmentationBaseModel(SegmentationBaseModel):
    """Base class for automatic prompted_segmentation models.
    This class is intended for models that perform automatic prompted_segmentation without user prompts.
    It provides a common interface for processing automatic prompted_segmentation requests.
    """
    @abstractmethod
    def process_automatic_request(self, request: AutomaticSegmentationRequest) -> tuple[list, list]:
        """Process the automatic prompted_segmentation request. Each automatic prompted_segmentation model should implement this method.
        Args:
            request (AutomaticSegmentationRequest): The prompted_segmentation request containing the image and parameters.
        Returns:
            tuple: A tuple containing an array of masks and an array of predicted iou scores. Each mask is a mask in HW
            format, where each pixel contains its label.
        """
        pass


class PromptedSegmentation3DBaseModel(SegmentationBaseModel):
    """Base class for prompted_segmentation models that handle scans.
    This class is intended for models that work with scans, such as SAM2.
    It provides a common interface for preparing input and segmenting scans.
    """
    @abstractmethod
    def process_prompted_segmentation_3D_request(self, request: ScanPromptedSegmentationRequest):
        """Propagate the mask across the scan.
        This method should be overridden by subclasses to provide model-specific mask propagation logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class AutomaticSegmentation3DBaseModel(SegmentationBaseModel):
    """Base class for automatic prompted_segmentation models that handle scans.
    This class is intended for models that work with scans, such as SAM2.
    It provides a common interface for preparing input and segmenting scans.
    """
    @abstractmethod
    def process_automatic_segmentation_3D_request(self, request: ScanAutomaticSegmentationRequest) -> tuple[list, list]:
        """Propagate the mask across the scan.
        This method should be overridden by subclasses to provide model-specific mask propagation logic.
        """
        raise NotImplementedError("Subclasses should implement this method.")

from operator import concat
from typing import List, Annotated, Union, Dict

import numpy as np
from pydantic import BaseModel, Field, field_validator
from app.database import get_context_session
from app.database.images import Images
from app.database.scans import Scans
from app.routes.prompted_segmentation.util import get_contour_models
from app.schemas.contours import ContourModel
from app.schemas.labels import LabelHierarchy
from app.schemas.segmentation.prompts import PointPrompt, BoxPrompt, PolygonPrompt, CirclePrompt
from logging import getLogger
from collections import defaultdict
from app.services.contours import get_contours_from_binary_mask
from app.services.postprocessing import postprocess_binary_mask

logger = getLogger(__name__)


class Prompts(BaseModel):
    """ Model for validating a prompted segmentation request. One request represents one object to be segmented."""
    point_prompts: list[PointPrompt] | None = Field(default=None,
                                             description="A list of point prompts. Each point prompt must have x, y, and label.")
    box_prompt: BoxPrompt | None = Field(default=None,
                                         description="A bounding box prompt. Must have min_x, min_y, max_x, and max_y.")
    circle_prompt: CirclePrompt | None = Field(default=None,
                                               description="A circle prompt. Must have center_x, center_y, and radius.")
    polygon_prompt: PolygonPrompt | None = Field(default=None,
                                                 description="A polygon prompt. Must have a list of vertices.")


class PromptedSegmentationRequest(BaseModel):
    """ Model for validating the prompted_segmentation request. """
    apply_post_processing: bool = True
    image_id: int = 1
    mask_id: int = None
    parent_contour_id: int = None
    previous_contours: List[ContourModel] = None
    model: Union[int, str] = "sam2_tiny"
    prompts: Prompts
    label: Annotated[int, "Label of the mask."] = 0

    @field_validator('image_id')
    def validate_image_id(cls, value):
        with get_context_session() as session:
            if value <= 0:
                raise ValueError("image_id must be a positive integer.")
            elif session.query(Images).filter_by(id=value).first() is None:
                raise ValueError("image_id does not exist in the database.")
            return value


class AutomaticSegmentationRequest(BaseModel):
    """ Model for validating the prompted_segmentation request. """
    #apply_post_processing: bool = False
    image_id: Annotated[int, "ID of the image to segment."]
    model: Annotated[Union[int, str], "Model to use for automatic prompted_segmentation."] = "SAM2Tiny"
    min_x: Annotated[float, "Coordinates must be between 0 and 1."] = 0
    min_y: Annotated[float, "Coordinates must be between 0 and 1."] = 0
    max_x: Annotated[float, "Coordinates must be between 0 and 1."] = 1
    max_y: Annotated[float, "Coordinates must be between 0 and 1."] = 1

    @field_validator('image_id')
    def validate_image_id(cls, value):
        with get_context_session() as session:
            if value <= 0:
                raise ValueError("image_id must be a positive integer.")
            elif session.query(Images).filter_by(id=value).first() is None:
                raise ValueError("image_id does not exist in the database.")
            return value

    @field_validator("model")
    def validate_model(cls, value):
        with get_context_session() as session:
            value = check_model(value, "automatic")
        return value

    @field_validator('min_x', 'min_y', 'max_x', 'max_y')
    def validate_coordinates(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("Coordinates must be between 0 and 1.")
        return value


class ScanAutomaticSegmentationRequest(BaseModel):
    """ Model for validating the mask propagation request. """
    scan_id: int = 1
    model: Annotated[Union[int, str], "Model to use for prompted_segmentation."] = "SAM2Tiny"

    @field_validator('scan_id')
    def validate_scan_id(cls, value):
        with get_context_session() as session:
            if value <= 0:
                raise ValueError("scan_id must be a positive integer.")
            elif session.query(Scans).filter_by(id=value).first() is None:
                raise ValueError("scan_id does not exist in the database.")
            return value

    @field_validator("model")
    def validate_model(cls, value):
        with get_context_session() as session:
            value = check_model(value, "automatic_3d")
        return value


class ScanPromptedSegmentationRequest(BaseModel):
    """ Model for validating the mask propagation request. """
    scan_id: int = 1
    prompted_requests: Dict[int, List[PromptedSegmentationRequest]] = (
        Field(default_factory=dict,
              description="Dictionary that maps objects to their prompted prompted_segmentation requests. Each key is one object "
                          "ID, and can have multiple prompted_segmentation requests from different slices."))
    model: Annotated[Union[int, str], "Model to use for prompted_segmentation."] = "SAM2Tiny"

    @field_validator('scan_id')
    def validate_scan_id(cls, value):
        with get_context_session() as session:
            if value <= 0:
                raise ValueError("scan_id must be a positive integer.")
            elif session.query(Scans).filter_by(id=value).first() is None:
                raise ValueError("scan_id does not exist in the database.")
            return value

    @field_validator("model")
    def validate_model(cls, value):
        with get_context_session() as session:
            value = check_model(value, "prompted_3d")
        return value

    @field_validator('prompted_requests')
    def validate_prompted_requests(cls, value):
        if not isinstance(value, dict):
            raise ValueError("prompted_requests must be a dictionary.")
        object_id_to_label = {}  # Keeps track of object IDs and their labels. Each object ID can only have one label.
        for key, requests in value.items():
            if not isinstance(key, int) or key <= 0:
                raise ValueError("Keys in prompted_requests must be positive integers representing object IDs.")
            if not isinstance(requests, list):
                raise ValueError("Values in prompted_requests must be lists of PromptedSegmentationRequest.")
            for request in requests:
                if not key in object_id_to_label:
                    object_id_to_label[key] = request.label  # Set the label for the object ID if not already set.
                elif not object_id_to_label[key] == request.label:
                    # If the label for the object ID is already set, it must match the current request's label.
                    # If not, this means that some requests for the same object ID have different labels.
                    # E.g. a coral object has been annotated with the labels coral and background by mistake.
                    raise ValueError("All requests for the same object ID must have the same label!")
                if not isinstance(request, PromptedSegmentationRequest):
                    raise ValueError("Each item in the list must be a PromptedSegmentationRequest.")
            image_ids = [request.image_id for request in requests]
            if len(np.unique(image_ids)) != len(image_ids):
                raise ValueError("You cannot have multiple requests for the same image ID in prompted_requests.")
        return value


class SemanticSegmentationMaskModel(BaseModel):
    """ Model for the mask. """
    contours: List[ContourModel] = Field(default=[], description="List of objects represented by their contours.")
    confidence: float = Field(default=0.0, description="Confidence score of the segmentation. This can be a predicted"
                                                       " IoU for example.")

    @classmethod
    def from_numpy_mask(cls, np_mask, confidence, label_hierarchy: LabelHierarchy):
        """ Get a semantic segmentation mask model from a mask, a confidence score and a label hierarchy."""
        # Get contours of the postprocessed mask if postprocessing is enabled
        # Postprocessing might improve performance by removing noise
        contour_models_of_label_value = {}
        flat_contours_list = []
        unique_labels = np.unique(np_mask)
        flat_label_hierarchy = [label.value for label in label_hierarchy.build_flat_hierarchy(breadth_first=True)]
        for label in unique_labels:
            # Check whether all labels are okay
            if label not in flat_label_hierarchy:
                raise ValueError(f"Mask contains label {label}, which is not part of the label hierarchy!")

        for label in flat_label_hierarchy:
            # Go through the labels by a breadth first search
            if label == 0:
                # Skip the background label (usually 0)
                continue
            # First: Extract the mask for the current label and create Contour Models
            mask_label = postprocess_binary_mask((np_mask == label).astype(np.uint8))
            contours = get_contours_from_binary_mask(mask_label, only_return_biggest=False)
            contour_models = [ContourModel.from_cv_contour(contour, label, np_mask.shape[1], np_mask.shape[0])
                              for contour in contours]

            # Second: Iterate through the models and check for parent links
            parent = label_hierarchy.get_parent_by_value_of_child(label)
            if parent is not None:
                for contour in contour_models:
                    for parent_contour in contour_models_of_label_value[parent.value]:
                        if contour in parent_contour:
                            contour.parent_contour_id = parent_contour.id
                    else:
                        logger.error("Contour could not be added to a parent contour")

            contour_models_of_label_value[label] = contour_models
            flat_contours_list += contour_models
        return SemanticSegmentationMaskModel(contours=flat_contours_list, confidence=confidence)


class SegmentationResponse(BaseModel):
    """ Model for the prompted_segmentation response. """
    masks: List[SemanticSegmentationMaskModel]
    image_id: int = 0
    model: str

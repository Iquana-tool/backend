import os.path
from typing import List, Annotated, Union, Literal, Dict

import numpy as np
from pydantic import BaseModel, Field, field_validator, Extra, validator
from app.database import get_context_session
from app.database.images import Images
from app.database.scans import Scans
from app.schemas.segmentation.contours_and_quantifications import ContourModel
from app.schemas.segmentation.prompts import PointPrompt, BoxPrompt, PolygonPrompt, CirclePrompt
from logging import getLogger


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


class SegmentationMaskModel(BaseModel):
    """ Model for the mask. """
    contours: List[ContourModel]
    predicted_iou: Annotated[float, "Predicted IoU of the mask."] = 0.0


class SegmentationResponse(BaseModel):
    """ Model for the prompted_segmentation response. """
    masks: List[SegmentationMaskModel]
    image_id: int = 0
    model: str

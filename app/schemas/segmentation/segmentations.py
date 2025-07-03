import os.path
from typing import List, Annotated, Union, Literal, Dict

import numpy as np
from pydantic import BaseModel, Field, field_validator, Extra, validator
from app.database import get_context_session
from app.database.images import Images, Scans
from app.schemas.segmentation.contours_and_quantifications import ContourModel
from app.schemas.segmentation.prompts import PointPrompt, BoxPrompt, PolygonPrompt, CirclePrompt
from app.database.models import Models
from logging import getLogger


logger = getLogger(__name__)


def check_model(model_id: Union[int, str], model_type: Literal["prompted", "automatic", "prompted_3d", "automatic_3d"]) \
        -> int:
    """Check if the model exists in the database and return it."""
    with get_context_session() as session:
        if isinstance(model_id, str):
            try:
                model_id = int(model_id)
            except ValueError:
                model = session.query(Models).filter_by(name=model_id, model_type=model_type).first()
                if not model:
                    raise ValueError(f"Model with identifier '{model_id}' does not exist in the database. Please "
                                     f"make sure you ran the scripts/add_models_to_db.py script to add the models "
                                     f"to the database.")
                model_id = model.id
        model = session.query(Models).filter_by(id=model_id).first()
        if not model:
            raise ValueError(f"Model with id {model_id} does not exist in the database. Please make sure you ran the "
                             f"scripts/add_models_to_db.py script to add the models to the database.")
        if model.model_type not in model_type:
            raise ValueError(f"Model with id {model_id} is not an prompted prompted_segmentation model.")
        if not (os.path.exists(model.weights) and os.path.exists(model.config)):
            raise ValueError(f"Model with id {model_id} has invalid paths for weights or config. Please make sure "
                             f"they are correctly added.")
    return model_id


class PromptedSegmentationRequest(BaseModel):
    """ Model for validating the prompted_segmentation request. """
    apply_post_processing: bool = True
    image_id: int = 1
    mask_id: int = None
    parent_contour_id: int = None
    previous_contours: List[ContourModel] = None
    model: Union[int, str] = "SAM2Tiny"
    point_prompts: List[PointPrompt] = Field(default_factory=list,
                                             description="List of point prompts supplied by the user.")
    box_prompt: BoxPrompt = None
    polygon_prompt: PolygonPrompt = None
    circle_prompt: CirclePrompt = None
    label: Annotated[int, "Label of the mask."] = 0

    # Deprecated fields, kept for backwards compatibility
    min_x: float = 0
    min_y: float = 0
    max_x: float = 1
    max_y: float = 1

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
            value = check_model(value, "prompted")
        return value

    @field_validator('min_x', 'min_y', 'max_x', 'max_y')
    def validate_coordinates(cls, value):
        logger.warning("The min_x, min_y, max_x, max_y fields are deprecated and will be removed in a future version. "
                       "Instead, pass the parent contour id.")
        if not (0 <= value <= 1):
            raise ValueError("Coordinates must be between 0 and 1.")
        return value


class AutomaticSegmentationRequest(BaseModel):
    """ Model for validating the prompted_segmentation request. """
    #apply_post_processing: bool = False
    image_id: Annotated[int, "ID of the image to segment."]
    model: Annotated[Union[int, str], "Model to use for prompted_segmentation."] = "SAM2Tiny"
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
    model: int

import os.path
from typing import List, Annotated
from pydantic import BaseModel, Field, field_validator, Extra, ConfigDict
from app.database import get_context_session
from app.database.images import Images, Scans
from app.schemas.segmentation.contours_and_quantifications import ContourModel
from app.schemas.segmentation.prompts import PointPrompt, BoxPrompt, PolygonPrompt, CirclePrompt
from database.models import Models


class PromptedSegmentationRequest(BaseModel):
    """ Model for validating the segmentation request. """
    apply_post_processing: Annotated[bool, "Apply post-processing to the segmentation."] = True
    image_id: Annotated[int, "ID of the image to segment."] = 1
    model: Annotated[str, "Model to use for segmentation."] = "SAM2Tiny"
    min_x: Annotated[float, "Coordinates must be between 0 and 1."] = 0
    min_y: Annotated[float, "Coordinates must be between 0 and 1."] = 0
    max_x: Annotated[float, "Coordinates must be between 0 and 1."] = 1
    max_y: Annotated[float, "Coordinates must be between 0 and 1."] = 1
    point_prompts: List[PointPrompt] = Field(default_factory=list,
                                             description="List of point prompts supplied by the user.")
    box_prompts: List[BoxPrompt] = Field(default_factory=list,
                                         description="List of box prompts supplied by the user.")
    polygon_prompts: List[PolygonPrompt] = Field(default_factory=list,
                                                 description="List of polygon prompts supplied by the user.")
    circle_prompts: List[CirclePrompt] = Field(default_factory=list,
                                               description="List of circle prompts supplied by the user.")
    label: Annotated[int, "Label of the mask."] = 0

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
            model = session.query(Models).filter_by(id=value).first()
            if not model:
                raise ValueError(f"Model with id {value} does not exist in the database. Please make sure you ran the "
                                 f"scripts/add_models_to_db.py script to add the models to the database.")
            if model.model_type not in "prompted":
                raise ValueError(f"Model with id {value} is not an prompted segmentation model.")
            if not (os.path.exists(model.weights) and os.path.exists(model.config)):
                raise ValueError(f"Model with id {value} has invalid paths for weights or config. Please make sure "
                                 f"they are correctly added.")
        return value

    @field_validator('min_x', 'min_y', 'max_x', 'max_y')
    def validate_coordinates(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("Coordinates must be between 0 and 1.")
        return value


class AutomaticSegmentationRequest(BaseModel):
    """ Model for validating the segmentation request. """
    #apply_post_processing: bool = False
    image_id: Annotated[int, "ID of the image to segment."]
    model: Annotated[str, "Model id to use for segmentation."]
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
            model = session.query(Models).filter_by(id=value).first()
            if not model:
                raise ValueError(f"Model with id {value} does not exist in the database. Please make sure you ran the "
                                 f"scripts/add_models_to_db.py script to add the models to the database.")
            if model.model_type not in "automatic":
                raise ValueError(f"Model with id {value} is not an automatic segmentation model.")
            if not (os.path.exists(model.weights) and os.path.exists(model.config)):
                raise ValueError(f"Model with id {value} has invalid paths for weights or config. Please make sure "
                                 f"they are correctly added.")
        return value

    @field_validator('min_x', 'min_y', 'max_x', 'max_y')
    def validate_coordinates(cls, value):
        if not (0 <= value <= 1):
            raise ValueError("Coordinates must be between 0 and 1.")
        return value


class ScanAutomaticSegmentationRequest(BaseModel):
    """ Model for validating the mask propagation request. """
    scan_id: int = 1
    model: str = "SAM2Tiny"

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
            model = session.query(Models).filter_by(id=value).first()
            if not model:
                raise ValueError(f"Model with id {value} does not exist in the database. Please make sure you ran the "
                                 f"scripts/add_models_to_db.py script to add the models to the database.")
            if model.model_type not in "automatic_3d":
                raise ValueError(f"Model with id {value} is not an automatic 3D segmentation model.")
            if not (os.path.exists(model.weights) and os.path.exists(model.config)):
                raise ValueError(f"Model with id {value} has invalid paths for weights or config. Please make sure "
                                 f"they are correctly added.")
        return value


class ScanPromptedSegmentationRequest(BaseModel):
    """ Model for validating the mask propagation request. """
    scan_id: int = 1
    prompted_requests: List[PromptedSegmentationRequest] = Field(default_factory=list,
                                                                    description="List of prompted segmentation requests "
                                                                                "for the scan.")
    model: str = "SAM2Tiny"

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
            model = session.query(Models).filter_by(id=value).first()
            if not model:
                raise ValueError(f"Model with id {value} does not exist in the database. Please make sure you ran the "
                                 f"scripts/add_models_to_db.py script to add the models to the database.")
            if model.model_type not in "prompted_3d":
                raise ValueError(f"Model with id {value} is not a prompted 3D model segmentation model.")
            if not (os.path.exists(model.weights) and os.path.exists(model.config)):
                raise ValueError(f"Model with id {value} has invalid paths for weights or config. Please make sure "
                                 f"they are correctly added.")
        return value


class SegmentationMaskModel(BaseModel):
    """ Model for the mask. """
    contours: List[ContourModel]
    predicted_iou: Annotated[float, "Predicted IoU of the mask."] = 0.0


class SegmentationResponse(BaseModel):
    """ Model for the segmentation response. """
    masks: List[SegmentationMaskModel]
    image_id: int = 0
    model: str = "SAM2Tiny"

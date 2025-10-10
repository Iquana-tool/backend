from operator import concat
from typing import List, Annotated, Union, Dict

import numpy as np
from pydantic import BaseModel, Field, field_validator
from app.database import get_context_session
from app.database.images import Images
from app.database.scans import Scans
from app.routes.prompted_segmentation.util import get_contour_models
from app.schemas.contours import Contour
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
    previous_contours: List[Contour] = None
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


class SemanticSegmentationMask(BaseModel):
    """ Model for semantic masks. """
    contours: List[Contour] = Field(default=[], description="List of objects represented by their contours.")
    confidence: float = Field(default=0.0, description="Confidence score of the segmentation. This can be a predicted"
                                                       " IoU for example.")


class SegmentationResponse(BaseModel):
    """ Model for the prompted_segmentation response. """
    masks: List[SemanticSegmentationMask]
    image_id: int = 0
    model: str

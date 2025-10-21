from typing import List, Annotated, Union

from pydantic import BaseModel, Field, field_validator
from app.database import get_context_session
from app.database.images import Images
from app.schemas.contours import Contour
from app.schemas.prompted_segmentation.prompts import Prompts
from logging import getLogger

logger = getLogger(__name__)


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
    confidence: float = Field(default=0.0, description="Confidence score of the prompted_segmentation. This can be a predicted"
                                                       " IoU for example.")


class SegmentationResponse(BaseModel):
    """ Model for the prompted_segmentation response. """
    masks: List[SemanticSegmentationMask]
    image_id: int = 0
    model: str

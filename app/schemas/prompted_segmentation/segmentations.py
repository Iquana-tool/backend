from typing import List, Annotated, Union

from pydantic import BaseModel, Field, field_validator
from app.database import get_context_session
from app.database.images import Images
from app.schemas.contours import Contour
from app.schemas.prompted_segmentation.prompts import Prompts
from logging import getLogger

logger = getLogger(__name__)


class PromptedSegmentationHTTPRequest(BaseModel):
    """ Model for validating the prompted_segmentation request. """
    apply_post_processing: bool = Field(default=True, description="Apply post-processing to the segmentation mask.")
    image_id: int = Field(..., description="The id of the image to be segmented.")
    mask_id: int = Field(..., description="The id of the mask to which the found contours shall be linked.")
    parent_contour_id: int | None = Field(None, description="The id of the parent contour to which the new contour will "
                                                            "belong. The new contour will be added as a child of this "
                                                            "contour and cannot lie outside of it. Additionally, the "
                                                            "prompted segmentation will only look at the patch of the "
                                                            "parent contour for segmentation. None if the new contour "
                                                            "has no parent.")
    refine_contour_id: int | None = Field(None, description="The id of the contour that you want to refine. None if you"
                                                            " don't want to refine any contour, but run normal prompted "
                                                            "segmentation.")
    model_registry_key: str = Field(default="sam2_tiny", description="The model registry key to use for segmentation. "
                                                                     "This key is used to look up the model in the model "
                                                                     "registry of the prompted service.")
    prompts: Prompts = Field(..., description="The prompts to use for segmentation.")

    @field_validator('image_id')
    def validate_image_id(cls, value):
        with get_context_session() as session:
            if value <= 0:
                raise ValueError("image_id must be a positive integer.")
            elif session.query(Images).filter_by(id=value).first() is None:
                raise ValueError("image_id does not exist in the database.")
            return value


class PromptedSegmentationWebsocketRequest(BaseModel):
    """ Model for 2D segmentation form data. """
    model_identifier: str = Field(..., title="Model identifier", description="Model identifier string. "
                                                                             "Used to select the model.")
    user_id: str | int = Field(..., title="User ID", description="Unique identifier for the user.")
    prompts: Prompts = Field(..., title="Prompts", description="Prompts for segmentation")
    previous_mask: list[list[bool]] | None = Field(None, title="Previous Mask",
                                                  description="Optional previous mask to provide context.")


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

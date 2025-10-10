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
            contour_models = [Contour.from_cv_contour(contour, label, np_mask.shape[1], np_mask.shape[0])
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
        return SemanticSegmentationMask(contours=flat_contours_list, confidence=confidence)


class SegmentationResponse(BaseModel):
    """ Model for the prompted_segmentation response. """
    masks: List[SemanticSegmentationMask]
    image_id: int = 0
    model: str

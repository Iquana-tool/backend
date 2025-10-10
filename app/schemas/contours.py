import json
from collections import deque
from logging import getLogger
from typing import List

import cv2
import numpy as np
from pydantic import BaseModel, field_validator, Field, model_validator
from sqlalchemy.orm import Query, Session

from app.database.contours import Contours
from app.routes.contours import add_contour
from app.schemas.quantification import QuantificationModel
from app.services.contours import get_contours_from_binary_mask
from app.services.postprocessing import postprocess_binary_mask

logger = getLogger(__name__)


class Contour(BaseModel):
    """ Model for a contour to be added. """
    id: int | None = Field(default=None, description="Contour id. Only pass None if the id is not yet known.")
    label: int | None = Field(default=None, description="ID of the label of the mask. None for unlabelled contour.")
    parent_id: int | None = Field(default=None, description="ID of the parent contour. None if the contour has "
                                                                    "no parent")
    children: list["Contour"] = Field(default=[], description="List of objects represented by their contours.")

    x: List[float] = Field(default_factory=list, description="X-coordinates of the contour.")
    y: List[float] = Field(default_factory=list, description="Y-coordinates of the contour.")

    added_by: str = Field(default_factory=str, description="ID of the user or model who added this contour.")
    confidence: float = Field(default=1., description="Confidence score of the contour.")
    temporary: bool = Field(default=False, description="Whether or not this contour is temporarily added (eg. if a model added it).")
    quantification: QuantificationModel | None = Field(default=None, description="Quantification of the contour. Does "
                                                                                 "not need to be provided.")

    @model_validator(mode="after")
    def validate_after(self):
        """ Validate after initialization. Check if quantifications are computed. """
        if self.quantification is None:
            self.quantification = QuantificationModel.from_cv_contour(self.contour)
        elif self.quantification.is_empty:
            self.quantification.parse_cv_contour(self.contour)

    @field_validator('x', 'y')
    def validate_coordinates(cls, value):
        return [min(max(coord, 0.), 1.) for coord in value]

    @property
    def contour(self) -> np.ndarray:
        """ As a opencv contour. """
        return np.array([[self.x, self.y]])

    @property
    def points(self) -> np.ndarray[tuple[float, float]]:
        return np.array(zip(self.x, self.y))

    def to_db_entry(self, mask_id):
        return Contours(
            id=self.id,
            mask_id=self.mask_id,
            x=self.x,
            y=self.y,
            label=self.label,
            parent_id=self.parent_id,
            temporary=self.temporary,
            added_by=self.added_by,
            confidence_score=self.confidence,
            area=self.quantification.area,
            perimeter=self.quantification.perimeter,
            circularity=self.quantification.circularity,
            diameter=self.quantification.max_diameter,
        )

    @classmethod
    def from_normalized_cv_contour(cls, normalized_cv_contour, label, added_by, temporary):
        x_coords = normalized_cv_contour[..., 0].flatten()
        y_coords = normalized_cv_contour[..., 1].flatten()
        return cls(
            x=x_coords.tolist(),
            y=y_coords.tolist(),
            label=label,
            added_by=added_by,
            temporary=temporary,
        )

    @classmethod
    def from_db(cls, contour: Contours):
        return cls(
            id=contour.id,
            parent_id=contour.parent_id,
            label=contour.label,
            x=json.loads(contour.x),
            y=json.loads(contour.y),
            added_by=contour.added_by,
            temporary=contour.temporary,
            quantification=QuantificationModel(
                area=contour.area,
                perimeter=contour.perimeter,
                circularity=contour.circularity,
                max_diameter=contour.diameter,
            )
        )

    def __in__(self, other):
        """
        Is this contour contained in another contour. Checks whether all points in this contour are inside another.
        """
        return all(cv2.pointPolygonTest(
            (other.contour * 10_000).astype(np.uint32),
            pt=(pt * 10_000).astype(np.uint32),
            measureDist=False) >= 0 for pt in self.points)

    def add_child(self, child):
        self.children.append(child)


class ContourHierarchy(BaseModel):
    """ A hierarchy of contours. """
    root_contours: list[Contour] = Field(default=[], description="List of objects represented by their contours.")
    id_to_contour: dict[int, Contour] = Field(default=None, description="Dict mapping contour id to object.")
    label_id_to_contour: dict[int, Contour] = Field(default=None, description="Dict mapping label id to object.")

    @classmethod
    def from_query(cls, query: Query[type[Contours]]) -> "ContourHierarchy":
        """ Adds all contours in a breadth first search, then connects them to a hierarchy. """
        # Fetch all root contours (parent_id is None)
        root_contours = query.filter_by(parent_id=None).all()
        root_ids = [contour.id for contour in root_contours]

        # Fetch all labels in the hierarchy
        queue = deque(root_contours)

        # Build a map from id to Label
        id_to_contour = {}
        label_id_to_contour = {}

        # Build the hierarchy
        while queue:
            contour = queue.popleft()
            contour_obj = Contour.from_db(contour)
            id_to_contour[contour.id] = contour_obj
            label_id_to_contour[contour.label] = contour_obj
            if contour.parent_id is not None:
                parent = id_to_contour[contour.parent_id]
                parent.add_child(contour_obj)
            queue.extend(query.filter_by(parent_id=contour.id).all())

        # Return the root-level labels
        return cls(
            root_contours=[id_to_contour[root_id] for root_id in root_ids],
            id_to_contour=id_to_contour,
            label_id_to_contour=label_id_to_contour,
        )

    @classmethod
    async def add_to_db(cls, mask_id, np_mask, label_hierarchy, added_by, temporary, db: Session):
        """
        Get a contour hierarchy from a mask and a label hierarchy. The hierarchy will respect both the label
        hierarchy as well as spatial hierarchy, i.e. each child contour lies within its parent.
        """
        contour_models_with_label_id = {}
        root_contours = []
        id_to_contour = {}
        flat_label_hierarchy = label_hierarchy.build_flat_hierarchy(breadth_first=True)
        height, width = np_mask.shape[:2]
        for label in flat_label_hierarchy:
            # Go through the labels by a breadth first search
            if label.value == 0:
                # Skip the background label (usually 0)
                continue

            # First: Extract the mask for the current label and create Contour Models
            mask_label = postprocess_binary_mask((np_mask == label.value).astype(np.uint8))
            contours = get_contours_from_binary_mask(mask_label, only_return_biggest=False)

            contour_models = []
            contour_entries = []
            # Second: Add them to the database to get an id for each contour
            for contour in contours:
                # Normalize to [0, 1] coords
                contour[..., 0] /= width
                contour[..., 1] /= height
                contour_model = Contour.from_normalized_cv_contour(contour,
                                                                   label,
                                                                   added_by,
                                                                   temporary)
                entry = contour_model.to_db_entry(mask_id)
                db.add(entry)
                db.flush()
                # Update with id
                contour_model.id = entry.id
                contour_entries.append(entry)
                contour_models.append(contour_model)
                id_to_contour[entry.id] = contour_model

            # Third: Iterate through the models and check for parent links
            parent = label_hierarchy.get_parent_by_value_of_child(label)
            if parent is not None:
                for contour, entry in zip(contour_models, contour_entries):
                    # For each contour, that we found, we check:
                    for parent_contour in contour_models_with_label_id[parent.value]:
                        # Does any parent label contour exist, in which the contour lies
                        # Depending on the nesting, this can take quite a while
                        if contour in parent_contour:
                            contour.parent_id = parent_contour.id
                            entry.parent_id = parent_contour.id
                            parent_contour.add_child(contour)
                            break
                    else:
                        # This should not happen, something is wrong.
                        logger.error("Contour could not be added to a parent contour")
            else:
                root_contours.extend(contour_models)
            contour_models_with_label_id[label] = contour_models
        return cls(
            root_contours=root_contours,
            label_id_to_contour=contour_models_with_label_id,
            id_to_contour=id_to_contour,
        )
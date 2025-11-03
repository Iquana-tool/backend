import json
from collections import deque, defaultdict
from logging import getLogger
from typing import List

import cv2
import numpy as np
from pydantic import BaseModel, field_validator, Field, model_validator
from sqlalchemy.orm import Query, Session

from app.database.contours import Contours
from app.schemas.labels import LabelHierarchy
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

    x: list[float] = Field(default_factory=list, description="X-coordinates of the contour.")
    y: list[float] = Field(default_factory=list, description="Y-coordinates of the contour.")
    path: str | None = Field(default=None, description="SVG path string for rendering the contour.")

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
        return self

    @field_validator('x', 'y')
    def validate_coordinates(cls, value):
        return [min(max(coord, 0.), 1.) for coord in value]

    @property
    def contour(self) -> np.ndarray:
        """
        As a opencv contour.
        Opencv contours have the form Number of points x empty dimension x Tuple of x and y coordinate.
        """
        return np.expand_dims(self.points, axis=1)

    @property
    def points(self) -> np.ndarray[tuple[float, float]]:
        return np.array(list(zip(self.x, self.y)))

    def compute_path(self, image_width: int, image_height: int):
        """Compute SVG path from normalized coordinates (0-1) to pixel coordinates."""
        if not self.x or not self.y or len(self.x) == 0:
            self.path = ""
            return
        first_x = round(self.x[0] * image_width)
        first_y = round(self.y[0] * image_height)
        path = f"M {first_x} {first_y}"
        for i in range(1, len(self.x)):
            x = round(self.x[i] * image_width)
            y = round(self.y[i] * image_height)
            path += f" L {x} {y}"
        self.path = path + " Z"

    def to_rescaled_contour(self, height, width):
        """ Return a rescaled contour given the height and width. """
        rescaled_x = (np.array(self.x) * height).asytpe(int)
        rescaled_y = (np.array(self.y) * width).astype(int)
        return np.expand_dims(np.array(zip(rescaled_x, rescaled_y)), axis=1)

    def to_db_entry(self, mask_id):
        return Contours(
            id=self.id,
            mask_id=mask_id,
            x=json.dumps(self.x),
            y=json.dumps(self.y),
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
    def from_db(cls, contour: Contours, image_width: int | None = None, image_height: int | None = None):
        contour_obj = cls(
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
        # Compute SVG path if image dimensions are provided
        if image_width is not None and image_height is not None:
            contour_obj.compute_path(image_width, image_height)
        return contour_obj

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
    label_id_to_contours: dict[int | None, list[Contour]] = Field(default=defaultdict(list),
                                                           description="Dict mapping label id to a list of objects.")

    @classmethod
    def from_query(cls, query: Query[type[Contours]]) -> "ContourHierarchy":
        """ Adds all contours in a breadth first search, then connects them to a hierarchy. """
        # Get image dimensions (all contours share same mask : image)
        first_contour = query.first()
        image_width, image_height = 1, 1
        if first_contour:
            from app.database.masks import Masks
            from app.database.images import Images
            mask = query.session.query(Masks).filter_by(id=first_contour.mask_id).first()
            if mask:
                image = query.session.query(Images).filter_by(id=mask.image_id).first()
                if image:
                    image_width, image_height = image.width, image.height

        # Fetch all root contours (parent_id is None)
        root_contours = query.filter_by(parent_id=None).all()
        root_ids = [contour.id for contour in root_contours]

        # Fetch all labels in the hierarchy
        queue = deque(root_contours)

        # Build a map from id to Label
        id_to_contour = {}
        label_id_to_contour = defaultdict(list)

        # Build the hierarchy
        while queue:
            contour = queue.popleft()
            contour_obj = Contour.from_db(contour, image_width, image_height)
            id_to_contour[contour.id] = contour_obj
            label_id_to_contour[contour.label].append(contour_obj)
            if contour.parent_id is not None:
                parent = id_to_contour[contour.parent_id]
                parent.add_child(contour_obj)
            queue.extend(query.filter_by(parent_id=contour.id).all())

        # Return the root-level labels
        return cls(
            root_contours=[id_to_contour[root_id] for root_id in root_ids],
            id_to_contour=id_to_contour,
            label_id_to_contours=label_id_to_contour,
        )

    def dump_contours_as_list(self, breadth_first: bool = True) -> list[Contour]:
        """ Dump all contours in the hierarchy as a list. Can be done in breadth first or depth first order. """
        contours_list = []
        queue = deque(self.root_contours)
        while queue:
            contour = queue.popleft()
            contours_list.append(contour)
            if breadth_first:
                queue.extend(contour.children)
            else:
                queue.extendleft(reversed(contour.children))
        return contours_list

    async def add_contours_from_mask_to_self_and_db(self,
                                                    mask_id: int,
                                                    np_mask: np.ndarray,
                                                    label_hierarchy: LabelHierarchy,
                                                    added_by: str,
                                                    temporary: bool,
                                                    db: Session):
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
                self.id_to_contour[entry.id] = contour_model
                self.label_id_to_contours[label.id].append(contour_model)

            # Third: Iterate through the models and check for parent links
            parent = label_hierarchy.get_parent_by_value_of_child(label)
            if parent is not None:
                for contour, entry in zip(contour_models, contour_entries):
                    # For each contour, that we found, we check:
                    for parent_contour in self.label_id_to_contours[parent.value]:
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
        db.commit()
        return self

    def to_semantic_mask(self, height, width, label_id_to_value_map: dict[int, int]) -> np.ndarray:
        """ Turn the hierarchy into a semantic mask of the given shape. In a semantic mask each pixel value represents a
        class. """
        # Create empty canvas
        canvas = np.zeros((height, width), dtype=np.uint8)
        # Create empty queue
        queue = deque()
        # Enqueue root contours
        queue.extend(self.root_contours)
        while queue:
            # Remove the oldest entry
            contour = queue.popleft()
            # If the contour has no label we cannot add it to the mask
            if contour.label:
                canvas = cv2.drawContours(canvas,
                                          contour.to_rescaled_contour(height, width),
                                          -1, # -1 means fill the contour
                                          [label_id_to_value_map[contour.label]],
                                          1)
                # Add all children to the queue
                if len(contour.children) > 0:
                    queue.extend(contour.children)
        # Return the filled array
        return canvas

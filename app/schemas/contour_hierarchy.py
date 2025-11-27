from collections import defaultdict, deque

import cv2
import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.orm import Query, Session

from app.database.contours import Contours
from app.database.images import Images
from app.database.masks import Masks
from app.schemas.contours import Contour, logger
from app.schemas.labels import LabelHierarchy
from app.services.contours import get_contours_from_binary_mask
from app.services.postprocessing import postprocess_binary_mask


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

    @classmethod
    async def from_semantic_mask(cls,
                                 mask_id: int,
                                 np_mask: np.ndarray,
                                 label_hierarchy: LabelHierarchy,
                                 added_by: str,
                                 db: Session):
        """
        Get a contour hierarchy from a mask and a label hierarchy. The hierarchy will respect both the label
        hierarchy as well as spatial hierarchy, i.e. each child contour lies within its parent.
        """
        contour_models_with_label_id = {}
        root_contours = []
        id_to_contour = {}
        label_id_to_contours = defaultdict(list)
        flat_label_hierarchy = label_hierarchy.build_flat_hierarchy(breadth_first=True)
        height, width = np_mask.shape[:2]
        for label in flat_label_hierarchy:
            # Go through the labels by a breadth first search
            if label.value == 0:
                # Skip the background label (usually 0)
                continue

            # First: Extract the mask for the current label and create Contour Models
            mask_label = postprocess_binary_mask((np_mask == label.value).astype(np.uint8))
            contour_models = get_contours_from_binary_mask(mask_label,
                                                           only_return_biggest=False,
                                                           limit=None,
                                                           added_by=added_by,
                                                           label_id=label.id,
                                                           )

            contour_entries = []
            # Second: Add them to the database to get an id for each contour
            for contour_model in contour_models:
                entry = contour_model.to_db_entry(mask_id)
                db.add(entry)
                db.flush()
                # Update with id
                contour_model.id = entry.id
                contour_entries.append(entry)
                contour_models.append(contour_model)
                id_to_contour[entry.id] = contour_model
                label_id_to_contours[label.id].append(contour_model)

            # Third: Iterate through the models and check for parent links
            parent = label_hierarchy.get_parent_by_value_of_child(label)
            if parent is not None:
                for contour, entry in zip(contour_models, contour_entries):
                    # For each contour, that we found, we check:
                    for parent_contour in label_id_to_contours[parent.value]:
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
        return cls(
            root_contours=root_contours,
            id_to_contour=id_to_contour,
            label_id_to_contours=label_id_to_contours,
        )

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
            if contour.label_id:
                canvas = cv2.drawContours(canvas,
                                          contour.to_rescaled_contour(height, width),
                                          -1,  # -1 means fill the contour
                                          [label_id_to_value_map[contour.label_id]],
                                          1)
                # Add all children to the queue
                if len(contour.children) > 0:
                    queue.extend(contour.children)
        # Return the filled array
        return canvas

    def get_label_quantification(self, label_id):
        # First get all relevant contours
        contours = self.label_id_to_contours[label_id]

        # Second track all relevant metrics
        metrics = defaultdict(list)
        child_counts = defaultdict(list)
        for contour in contours:
            # Get the quantifications from the quantification model
            for quant_key, quant_value in contour.quantification.model_dump().items():
                metrics[quant_key].append(quant_value)

            # Count the children
            _child_counts = defaultdict(lambda: 0)
            for child_contour in contour.children:
                _child_counts[child_contour.label_id] += 1

            for label_id in _child_counts:
                child_counts[label_id].append(_child_counts[label_id])
        return {
            "metrics": metrics,
            "child_counts": child_counts,
        }

    def get_all_quantifications(self):
        response = {}
        for label in self.label_id_to_contours.keys():
            response[label] = self.get_label_quantification(label)
        return response

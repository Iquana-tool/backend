import json
from logging import getLogger
import cv2
import numpy as np
from pydantic import BaseModel, field_validator, Field, model_validator
from pycocotools import mask as maskUtils
from app.database import get_context_session
from app.database.contours import Contours
from app.schemas.quantification import QuantificationModel

logger = getLogger(__name__)


class Contour(BaseModel):
    """ Model for a contour to be added. """
    id: int | None = Field(default=None, description="Contour id. Only pass None if the id is not yet known.")
    label_id: int | None = Field(default=None, description="ID of the label of the mask. None for unlabelled contour.")
    parent_id: int | None = Field(default=None, description="ID of the parent contour. None if the contour has "
                                                                    "no parent")
    children: list["Contour"] = Field(default=[], description="List of objects represented by their contours.")
    reviewed_by: list[str] = Field(default=[], description="List of users who reviewed the contour.")

    x: list[float] = Field(default_factory=list, description="X-coordinates of the contour.")
    y: list[float] = Field(default_factory=list, description="Y-coordinates of the contour.")
    path: str | None = Field(default=None, description="SVG path string for rendering the contour.")

    added_by: str = Field(default_factory=str, description="ID of the user or model who added this contour.")
    confidence: float = Field(default=1., description="Confidence score of the contour.")
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

    def get_children_by_label(self, label_id):
        children = []
        for child in self.children:
            if child.label_id == label_id:
                children.append(child)
        return children

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

    def to_binary_mask(self, height, width) -> np.ndarray:
        """ Return a binary mask given the height and width. """
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        binary_mask = cv2.drawContours(
            image=binary_mask,
            contours=[self.to_rescaled_contour(height, width)],
            contourIdx=-1,  # -1 means fill the contour
            color=1,
            thickness=cv2.FILLED
        )
        return binary_mask.astype(bool)

    def to_rle_encoding(self, height, width):
        bin_mask = self.to_binary_mask(height, width)
        return maskUtils.encode(np.asfortranarray(bin_mask.astype(np.uint8)))

    def to_rescaled_contour(self, height, width):
        """ Return a rescaled contour given the height and width. """
        rescaled_x = (np.array(self.x) * width).astype(int)
        rescaled_y = (np.array(self.y) * height).astype(int)
        return np.expand_dims(np.array(list(zip(rescaled_x, rescaled_y))), axis=1)

    def to_db_entry(self, mask_id):
        return Contours(
            id=self.id,
            mask_id=mask_id,
            x=json.dumps(self.x),
            y=json.dumps(self.y),
            label=self.label_id,
            parent_id=self.parent_id,
            added_by=self.added_by,
            confidence_score=self.confidence,
            area=self.quantification.area,
            perimeter=self.quantification.perimeter,
            circularity=self.quantification.circularity,
            diameter=self.quantification.max_diameter,
        )

    @classmethod
    def from_normalized_cv_contour(cls, normalized_cv_contour, label, added_by):
        x_coords = normalized_cv_contour[..., 0].flatten()
        y_coords = normalized_cv_contour[..., 1].flatten()
        return cls(
            x=x_coords.tolist(),
            y=y_coords.tolist(),
            label_id=label,
            added_by=added_by,
        )

    @classmethod
    def from_db(cls, contour: Contours | type[Contours], image_width: int | None = None, image_height: int | None = None):
        contour_obj = cls(
            id=contour.id,
            parent_id=contour.parent_id,
            label_id=contour.label,
            x=json.loads(contour.x),
            y=json.loads(contour.y),
            added_by=contour.added_by,
            reviewed_by=[user.username for user in contour.reviewed_by],
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

    @classmethod
    def from_id(cls, id: int):
        with get_context_session() as session:
            return cls.from_db(session.query(Contours).filter_by(id=id).first())

    @classmethod
    def from_binary_mask(cls,
                         binary_mask: np.ndarray,
                         label,
                         added_by):
        if binary_mask.dtype != np.uint8:
            binary_mask = binary_mask.astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea).astype(float)
        contour[..., 0]  = contour[..., 0] / binary_mask.shape[1]
        contour[..., 1] = contour[..., 1] / binary_mask.shape[0]
        return cls.from_normalized_cv_contour(
            normalized_cv_contour=contour,
            label=label,
            added_by=added_by,
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

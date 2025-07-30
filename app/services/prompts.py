# This file contains classes for prompts
import cv2
import numpy as np
from app.schemas.segmentation.segmentations import PromptedSegmentationRequest
from logging import getLogger

logger = getLogger(__name__)


class Prompts:
    """ A class to track multiple prompts supported by SAM2. Prompts tracked by this class represent one object. To
        track multiple objects, create multiple instances of this class. The prompts can be added and removed as needed.
        The prompts can be converted to the format expected by the SAM2 model using the `to_SAM2_input` method.
    """

    def __init__(self):
        self.point_prompts = []
        self.point_labels = []
        self.box_prompts = []

    def from_segmentation_request(self, seg_req: PromptedSegmentationRequest):
        """ Initialize the prompts from a segmentation request. This method will extract the prompts from the
            segmentation request and add them to the list of prompts. The segmentation request should contain the
            following attributes:
            - point_prompts: A list of point prompts, where each point prompt is a dictionary with keys 'x', 'y', and 'label'.
            - box_prompt: A dictionary with keys 'min_x', 'min_y', 'max_x', and 'max_y' representing the bounding box.
            - circle_prompt: A dictionary with keys 'center_x', 'center_y', and 'radius' representing the circle.
            - polygon_prompt: A dictionary with a key 'vertices' containing a list of vertices, where each vertex is a
                list of two floats representing the x and y coordinates of the vertex.
        """
        for point in seg_req.point_prompts:
            self.add_point_annotation(point.x, point.y, point.label)
        box = seg_req.box_prompt
        if box:
            self.add_box_annotation(box.min_x, box.min_y, box.max_x, box.max_y)
        circle = seg_req.circle_prompt
        if circle:
            self.add_circle_annotation(circle.center_x, circle.center_y, circle.radius)
        polygon = seg_req.polygon_prompt
        if polygon:
            self.add_polygon_annotation(polygon.vertices)

    def add_point_annotation(self, x: float, y: float, label: int):
        """ Add a point annotation to the list of prompts.
            Args:
                x (float): The x-coordinate of the point. Must be between 0 and 1.
                y (float): The y-coordinate of the point. Must be between 0 and 1.
                label (int): The label of the point. Must be 0 for background (negative annotation) or 1 for foreground
                             (positive annotation).
        """
        if not isinstance(label, int):
            try:
                label = int(label)
            except ValueError:
                raise ValueError(f"Label must be 0 for background (negative annotation) "
                                 f"or 1 for foreground (positive annotation). Got {label}.")
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"x and y must be between 0 and 1. x: {x}, y: {y}")
        self.point_prompts.append([x, y])
        self.point_labels.append(label)

    def remove_point_annotation(self, index: int):
        """ Remove a point annotation from the list of prompts by index. The index corresponds to the order in which the
            points were added. Note: Indices of box prompts and point prompts are counted separately.
        """
        if index >= len(self.point_prompts):
            raise IndexError("Index out of range. Make sure to only count the point prompts excluding box prompts!")
        self.point_prompts.pop(index)
        self.point_labels.pop(index)

    def add_box_annotation(self, min_x: float, min_y: float, max_x: float, max_y: float):
        """ Add a box annotation to the list of prompts. Box annotations require no label, as they are always
            positive annotations.
            Args:
                min_x: The minimum x-coordinate of the box. Must be between 0 and 1.
                min_y: The minimum y-coordinate of the box. Must be between 0 and 1.
                max_x: The maximum x-coordinate of the box. Must be between 0 and 1.
                max_y: The maximum y-coordinate of the box. Must be between 0 and 1.
        """
        if min_x < 0 or min_y < 0 or max_x > 1 or max_y > 1:
            raise ValueError("min_x, min_y, max_x, max_y must be between 0 and 1.")
        self.box_prompts.append(np.array([min_x, min_y, max_x, max_y]))

    def remove_box_annotation(self, index: int):
        """ Remove a box annotation from the list of prompts by index. The index corresponds to the order in which the
            boxes were added. Note: Indices of box prompts and point prompts are counted separately.
        """
        if index >= len(self.box_prompts):
            raise IndexError("Index out of range. Make sure to only count the box prompts excluding point prompts!")
        self.box_prompts.pop(index)

    def add_polygon_annotation(self, vertices: list[list[float]]):
        """ Add a polygon prompt to the list of prompts. The polygon vertices should be in the format [(x1, y1), (x2, y2), ...].
            Every point enclosed by the polygon will be treated as a positive annotation.
        """
        max_x = np.max([v[0] for v in vertices])
        min_x = np.min([v[0] for v in vertices])
        max_y = np.max([v[1] for v in vertices])
        min_y = np.min([v[1] for v in vertices])
        grid_size = 4  # Define the grid size for sampling
        contour = (np.array(vertices) * 1000).astype(np.int32)  # Scale vertices to pixel coordinates
        # Create a box annotation around the polygon to sample points
        # and add negative annotations for points outside the polygon
        self.add_box_annotation(min_x, min_y, max_x, max_y)
        for point_x in np.arange(min_x, max_x, (max_x - min_x) / grid_size):
            for point_y in np.arange(min_y, max_y, (max_y - min_y) / grid_size):
                # Check if the point is inside the circle
                if cv2.pointPolygonTest(contour, (point_x * 1000, point_y * 1000), False) <= 0:
                    self.add_point_annotation(point_x, point_y, 0)

    def add_circle_annotation(self, x: float, y: float, radius: float):
        """ Add a circle prompt to the list of prompts. The circle prompt will add a positive annotation for all points
            within the circle.
        """
        # We create a slightly large circle to avoid edge effects when sampling points
        # Then we lay a grid over the circle and check if the points are inside the circle
        grid_size = 4  # Define the grid size for sampling
        min_x = max(0, x - radius)
        min_y = max(0, y - radius)
        max_x = min(1, x + radius)
        max_y = min(1, y + radius)
        # Create a box annotation around the polygon to sample points
        # and add negative annotations for points outside the polygon
        self.add_box_annotation(min_x, min_y, max_x, max_y)
        for point_x in np.arange(min_x, max_x, (max_x - min_x) / grid_size):
            for point_y in np.arange(min_y, max_y, (max_y - min_y) / grid_size):
                # Check if the point is inside the circle
                if ((point_x - x) ** 2 + (point_y - y) ** 2) <= radius ** 2:
                    self.add_point_annotation(point_x, point_y, 0)

    def get_prompts_as_ndarray(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Get the prompts as numpy arrays. The returned arrays should be used as input to the SAM2 model.
            Returns:
                A tuple containing the point prompts, point labels, and box prompts as numpy arrays.
        """
        point_prompts = np.array(self.point_prompts) if self.point_prompts else None
        points_labels = np.array(self.point_labels) if self.point_labels else None
        box_prompts = np.array(self.box_prompts) if self.box_prompts else None
        return point_prompts, points_labels, box_prompts

    def to_SAM2_input(self) -> dict[str, np.ndarray]:
        """ Convert the prompts to the format expected by the SAM2 model. Returns a dictionary with keys 'point_prompts',
            'point_labels', and 'box_prompts' as expected by the SAM2 `predict` method.\n

            Example usage:
                ```
                prompts = Prompts()
                prompts.add_point_annotation(100, 100, 1)
                prompts.add_point_annotation(200, 200, 0)
                prompts.add_box_annotation(50, 50, 150, 150)
                masks, _, _ = model.predict(prompts.to_SAM2_input())
                ```
        """
        point_prompts, points_labels, box_prompts = self.get_prompts_as_ndarray()
        return_dict = {}
        if point_prompts is not None:
            return_dict['point_coords'] = point_prompts
            return_dict['point_labels'] = points_labels
        if box_prompts is not None:
            return_dict['box'] = box_prompts
        logger.debug(f"Converted prompts to SAM2 input: {return_dict}")
        return return_dict

    @staticmethod
    def convert_polygon_vertices_to_point_annotations(vertices: list[list[float]]) -> list[tuple[int, int]]:
        """ Convert a list of polygon vertices to a list of point annotations. The polygon vertices should be in the
            format [(x1, y1), (x2, y2), ...]. Every point enclosed by the polygon will be treated as a positive
            annotation. The method will return a dictionary with two keys: True and False. The
            True key will contain the positive annotations, and the False key will contain the negative annotations.
            The method will return the points as integer coordinates."""
        positive_annotations = []
        # Create an empty canvas and plot the polygon on it
        canvas = np.zeros((1000, 1000))
        polygon = np.array(vertices) * 1000
        canvas = cv2.fillPoly(canvas, [polygon.astype(dtype=np.int32)], 1)
        # Compute the general span of the polygon to identify the samnpling frequency
        min_x = min(polygon[:, 0])
        min_y = min(polygon[:, 1])
        max_x = max(polygon[:, 0])
        max_y = max(polygon[:, 1])
        stretch_x = max_x - min_x
        stretch_y = max_y - min_y
        # We want about 3 points per row and column to be sampled
        sampling_freq_x = int(stretch_x / 3)
        sampling_freq_y = int(stretch_y / 3)
        # Find all points enclosed by the polygon
        for x in range(int(min_x), int(max_x), sampling_freq_x):
            for y in range(int(min_y), int(max_y), sampling_freq_y):
                if canvas[y, x] == 1:
                    positive_annotations.append((x / 1000, y / 1000))
        return positive_annotations

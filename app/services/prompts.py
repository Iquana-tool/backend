# This file contains classes for prompts
import numpy as np


class Prompts:
    """ A class to track multiple prompts supported by SAM2. Prompts tracked by this class represent one object. To
        track multiple objects, create multiple instances of this class. The prompts can be added and removed as needed.
        The prompts can be converted to the format expected by the SAM2 model using the `to_SAM2_input` method.
    """
    def __init__(self):
        self.point_prompts = []
        self.point_labels = []
        self.box_prompts = []

    def add_point_annotation(self, x: float, y: float, label: int):
        """ Add a point annotation to the list of prompts.
            Args:
                x (float): The x-coordinate of the point. Must be between 0 and 1.
                y (float): The y-coordinate of the point. Must be between 0 and 1.
                label (int): The label of the point. Must be 0 for background (negative annotation) or 1 for foreground
                             (positive annotation).
        """
        if label not in [0, 1]:
            raise ValueError("Label must be 0 for background (negative annotation) "
                             "or 1 for foreground (positive annotation).")
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
        self.box_prompts.append([min_x, min_y, max_x, max_y])

    def remove_box_annotation(self, index: int):
        """ Remove a box annotation from the list of prompts by index. The index corresponds to the order in which the
            boxes were added. Note: Indices of box prompts and point prompts are counted separately.
        """
        if index >= len(self.box_prompts):
            raise IndexError("Index out of range. Make sure to only count the box prompts excluding point prompts!")
        self.box_prompts.pop(index)

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
        return return_dict


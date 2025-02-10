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

    def add_point_annotation(self, x: int, y: int, label: int):
        """ Add a point annotation to the list of prompts.
            Args:
                x (int): The x-coordinate of the point.
                y (int): The y-coordinate of the point.
                label (int): The label of the point. Must be 0 for background (negative annotation) or 1 for foreground
                             (positive annotation).
        """
        if label != 0 or label != 1:
            raise ValueError("Label must be 0 for background (negative annotation) "
                             "or 1 for foreground (positive annotation).")
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

    def add_box_annotation(self, min_x: int, min_y: int, max_x: int, max_y: int):
        """ Add a box annotation to the list of prompts. Box annotations require no label, as they are always
            positive annotations.
            Args:
                min_x: The minimum x-coordinate of the box.
                min_y: The minimum y-coordinate of the box.
                max_x: The maximum x-coordinate of the box.
                max_y: The maximum y-coordinate of the box.
        """
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
        point_prompts = np.array(self.point_prompts)
        points_labels = np.array(self.point_labels)
        box_prompts = np.array(self.box_prompts)
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
        return {'point_coords': point_prompts, 'point_labels': points_labels, 'box': box_prompts}


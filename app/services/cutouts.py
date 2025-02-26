import warnings

import numpy as np
import cv2


def get_contours(mask: np.ndarray) -> np.ndarray:
    """ Get the contours of the mask.
        Args:
            mask (np.ndarray): The mask to get the contours of.

        Returns:
            np.ndarray: The contours of the mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def cutout_contour_from_image(image: np.ndarray,
                              contour: np.ndarray,
                              resize_factor: float = 1.0,
                              darken_outside_contour: bool = False,
                              darkening_factor: float = 0.7) -> tuple[np.ndarray, float, float]:
    """ Cut out a box from the image, which includes the mask. The box is the smallest box that includes the mask unless
        size_factor is specified.
        Args:
            image (np.ndarray): The image to cut the mask from.
            contour (np.ndarray): The contour of the object to cut out.
            resize_factor (float): The factor to multiply the size of the box by. Default is 1.0. Smaller values will make
                the box smaller and not include the entire mask. Larger values will make the box larger and include more
                of the image, that does not contain the mask anymore.
            darken_outside_contour (bool): If True, the area outside the contour will be darkened. Default is False.
            darkening_factor (float): The factor to darken the area outside the contour by. Default is 0.7.

        Returns:
            tuple[np.ndarray, float, float]: The image with the mask cut out, the x-coordinate of the lower-left corner
            of the box, and the y-coordinate of the lower-left corner of the box.

    """
    min_y, max_y = np.min(contour[0]), np.max(contour[0])
    min_x, max_x = np.min(contour[1]), np.max(contour[1])
    # Resize the box using the size factor
    min_y = max(0, int(min_y - (max_y - min_y) * resize_factor))
    max_y = min(image.shape[0], int(max_y + (max_y - min_y) * resize_factor))
    min_x = max(0, int(min_x - (max_x - min_x) * resize_factor))
    max_x = min(image.shape[1], int(max_x + (max_x - min_x) * resize_factor))
    image_copy = image.copy()
    if darken_outside_contour:
        image_copy[:min_y, :] *= darkening_factor
        image_copy[max_y:, :] *= darkening_factor
        image_copy[:, :min_x] *= darkening_factor
        image_copy[:, max_x:] *= darkening_factor
    return image_copy[min_y:max_y, min_x:max_x], min_x, min_y


def cutout_objects_on_mask_from_image(image: np.ndarray, mask: np.ndarray, size_factor: float = 1.0,
                                        darken_outside_contours: bool = False, darkening_factor: float = 0.7) \
        -> list[np.ndarray]:
    """ Cut out a box from the image, which includes the mask. The box is the smallest box that includes the mask unless
        size_factor is specified.
        Args:
            image (np.ndarray): The image to cut the mask from.
            mask (np.ndarray): The mask to cut out.
            size_factor (float): The factor to multiply the size of the box by. Default is 1.0. Smaller values will make
                the box smaller and not include the entire mask. Larger values will make the box larger and include more
                of the image, that does not contain the mask anymore.
            darken_outside_contours (bool): If True, the area outside the contours will be darkened. Default is False.
            darkening_factor (float): The factor to darken the area outside the contours by. Default is 0.7.

        Returns:
            np.ndarray: The image with the mask cut out.
    """
    contours = get_contours(mask)
    cutouts = []
    for contour in contours:
        cutouts.append(cutout_contour_from_image(image, contour, size_factor, darken_outside_contours, darkening_factor))
    return cutouts

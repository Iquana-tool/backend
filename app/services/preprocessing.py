import cv2
import numpy as np

def create_cleaned_mask(img, min_contour_area=5000):
    """
    Generate a binary mask by thresholding a grayscale version of the image,
    followed by morphological operations and filtering out small contours.

    Args:
        img (np.ndarray): Input image (BGR).
        min_contour_area (int): Minimum contour area to retain.

    Returns:
        mask (np.ndarray): Cleaned binary mask.
        valid_contours (list[np.ndarray]): List of contours above the area threshold.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary thresholding
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find and filter contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)
    valid_contours = []

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, cv2.FILLED)
            valid_contours.append(contour)

    return filtered_mask, valid_contours

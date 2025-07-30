import numpy as np

def compute_pixel_scale_from_points(p1, p2, known_distance):
    """
    Compute the real-world scale (unit/pixel) from two points and the known real-world distance.

    This is used when a user draws a line (e.g., between two ruler ticks) on an image and specifies
    the real-world distance between them (e.g., 10 mm). The function returns the scale to convert
    pixel distances into physical units.

    Args:
        p1 (tuple): (x1, y1) coordinates of the first point in pixels.
        p2 (tuple): (x2, y2) coordinates of the second point in pixels.
        known_distance (float): Real-world distance between the two points (e.g., in mm).

    Returns:
        tuple: (scale_x, scale_y) where both are equal unit-per-pixel values (e.g., mm/pixel).
    """
    x1, y1 = p1
    x2, y2 = p2

    # Calculate pixel distance using Euclidean formula
    pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if pixel_distance == 0:
        raise ValueError("Cannot compute scale from two identical points.")

    # Scale = real-world distance (mm or other unit) / pixel distance
    scale_per_pixel = known_distance / pixel_distance

    # Return the same scale in both X and Y directions
    return scale_per_pixel, scale_per_pixel

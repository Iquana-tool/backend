

def crop_image(min_x, min_y, max_x, max_y, image):
    """Crop the image to the specified range. Only crops the image if necessary."""
    if min_x > 0 or min_y > 0 or max_x < 1 or max_y < 1:
        # Crop the image to the specified range
        return image[int(min_y * image.shape[0]):int(max_y * image.shape[0]),
                     int(min_x * image.shape[1]):int(max_x * image.shape[1])]
    return image

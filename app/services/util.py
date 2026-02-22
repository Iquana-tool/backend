import os


def get_mask_path_from_image_path(path: str):
    """ Given an image path, returns the corresponding mask path. """
    parts = path.split(os.path.sep)
    parts[-2] = "masks"  # Replace the parent directory
    full_path = os.path.sep.join(parts)
    return full_path.rsplit(".", 1)[0] + ".png"

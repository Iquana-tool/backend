"""Domain-specific exceptions for the IQuana backend.

Routes catch only the specific exception type they handle; every other exception
bubbles up uncaught so FastAPI returns a 500 with a full traceback in the logs.
This prevents silent misclassification of bugs as "404 Not Found" (RULE 1).
"""


class IQuanaBaseError(Exception):
    """Base class for all domain-specific errors in this project."""
    pass


class ImageNotFoundError(IQuanaBaseError):
    """Raised when an image_id does not match any row in the images table."""
    pass


class InvalidScaleError(IQuanaBaseError):
    """Raised when scale inputs are logically invalid (e.g. non-positive values,
    zero-length drawn line, or missing unit)."""
    pass


class DatasetNotFoundError(IQuanaBaseError):
    """Raised when a dataset_id does not match any row in the datasets table."""
    pass


class UnknownCalibrationKindError(IQuanaBaseError):
    """Raised when a calibration kind is not registered (see calibration.registry)."""
    pass


class InvalidMetadataError(IQuanaBaseError):
    """Raised when an image metadata key or value is empty or over its length cap."""
    pass


class InvalidCalibrationError(IQuanaBaseError):
    """Raised when calibration parameters fail their kind's validation.

    The scale kind keeps raising the older, more specific `InvalidScaleError`
    instead, so the existing /scale routes' 422 mapping is unchanged.
    """
    pass

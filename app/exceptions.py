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


class InvalidLabelFilterError(IQuanaBaseError):
    """Raised when COCO export label filtering parameters are invalid, empty, duplicate, or reference foreign labels (HTTP 422)."""
    pass


class DatasetArchiveExportError(IQuanaBaseError):
    """Raised when an IQUANA dataset archive export fails due to invalid or unrepresentable dataset state."""
    pass


class DatasetArchiveImportError(IQuanaBaseError):
    """Base class for all errors occurring during IQUANA dataset archive import."""
    pass


class DatasetArchiveValidationError(DatasetArchiveImportError):
    """Raised when an IQUANA dataset archive fails schema, reference, or integrity validation (HTTP 422)."""
    pass


class DatasetArchiveNameConflictError(DatasetArchiveImportError):
    """Raised when an imported dataset name conflicts with an existing dataset (HTTP 409)."""
    pass


class DatasetArchiveSizeLimitError(DatasetArchiveImportError):
    """Raised when an archive exceeds compressed or uncompressed size/member limits (HTTP 413)."""
    pass

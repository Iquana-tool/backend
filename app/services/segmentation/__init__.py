import config
from app.services.segmentation.mockup import MockupSegmentationModel
from app.services.segmentation.sam2 import SAM2


def get_model_via_identifier(identifier: str):
    """
    Returns the model class based on the identifier.
    """
    if identifier == "Mockup":
        return MockupSegmentationModel()
    elif identifier == "SAM2Tiny":
        return SAM2(config.ModelConfig.available_models["SAM2Tiny"]())
    elif identifier == "SAM2Small":
        return SAM2(config.ModelConfig.available_models["SAM2Small"]())
    elif identifier == "SAM2Large":
        return SAM2(config.ModelConfig.available_models["SAM2Large"]())
    elif identifier == "SAM2BasePlus":
        return SAM2(config.ModelConfig.available_models["SAM2BasePlus"]())
    else:
        raise ValueError(f"Unknown model identifier: {identifier}")

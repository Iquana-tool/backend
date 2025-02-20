import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")

# Model settings
SAM_CHECKPOINT_PATH = os.path.join(BASE_DIR, "sam_vit_b_01ec64.pth")
DEFAULT_MODEL_TYPE = "vit_b"

# Application settings
DEBUG = True
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Model configuration
class ModelConfig:
    class SAM2Tiny:
        weights = os.path.join(BASE_DIR, "sam_vit_b_01ec64.pth")
        config = "tiny"
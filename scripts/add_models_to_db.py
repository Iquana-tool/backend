"""This script adds segmentation models to the database. Add your models with their configurations here."""
import os.path
from paths import Paths
from app.database import get_context_session
from app.database.models import Models
from available_models import AvailableModels
from logging import getLogger

logger = getLogger(__name__)

example_entry = {
    # This is an example entry for a segmentation model.
    # You need to add it to the right list below.
    "base_model_identifier": "SAM2Prompted",  # This should match the base model identifier used in an entry
    # of a subdict in paths.py AvailableModels
    "name": "SAM2Tiny",
    "description": "A very small version of the SAM2 model optimized for speed and efficiency.",
    "model_type": "prompted",  # This should match one of the keys used in paths.py Available models class
    "version": "1.0",
    "created_at": None,  # Use a timestamp or datetime object if available
    "updated_at": None,  # Use a timestamp or datetime object if available
    "weights": "app/services/segmentation/weights/sam2.1_hiera_tiny.pt",
    "config": "app/services/segmentation/configs/sam2.1_hiera_tiny.yaml"
}

prompted_models_to_add = [
    {
        "base_model_identifier": "SAM2Prompted",
        "name": "SAM2Tiny",
        "description": "A very small version of the SAM2 model optimized for speed and efficiency.",
        "model_type": "prompted",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/sam2.1_hiera_tiny.pt",
        "config": "sam2.1_hiera_tiny.yaml"
    },
    {
        "base_model_identifier": "SAM2Prompted",
        "name": "SAM2Small",
        "description": "A small version of the SAM2 model with a balance between performance and speed.",
        "model_type": "prompted",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/sam2.1_hiera_small.pt",
        "config": "sam2.1_hiera_small.yaml"
    },
    {
        "base_model_identifier": "SAM2Prompted",
        "name": "SAM2Large",
        "description": "A larger version of the SAM2 model designed for high accuracy.",
        "model_type": "prompted",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/sam2.1_hiera_large.pt",
        "config": "sam2.1_hiera_large.yaml"
    },
    {
        "base_model_identifier": "SAM2Prompted",
        "name": "SAM2BasePlus",
        "description": "The Base version of the SAM2 model.",
        "model_type": "prompted",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/sam2.1_hiera_large.pt",
        "config": "sam2.1_hiera_large.yaml"
    },
]

automatic_models_to_add = [
    {
        "base_model_identifier": "SAM2Automatic",
        "name": "SAM2Tiny",
        "description": "A very small version of the SAM2 model optimized for speed and efficiency.",
        "model_type": "automatic",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/sam2.1_hiera_tiny.pt",
        "config": "sam2.1_hiera_tiny.yaml"
    },
    {
        "base_model_identifier": "SAM2Automatic",
        "name": "SAM2Small",
        "description": "A small version of the SAM2 model with a balance between performance and speed.",
        "model_type": "automatic",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/sam2.1_hiera_small.pt",
        "config": "sam2.1_hiera_small.yaml"
    },
    {
        "base_model_identifier": "SAM2Automatic",
        "name": "SAM2Large",
        "description": "A larger version of the SAM2 model designed for high accuracy.",
        "model_type": "automatic",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/sam2.1_hiera_large.pt",
        "config": "sam2.1_hiera_large.yaml"
    },
    {
        "base_model_identifier": "SAM2Automatic",
        "name": "SAM2BasePlus",
        "description": "The Base version of the SAM2 model.",
        "model_type": "automatic",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/sam2.1_hiera_large.pt",
        "config": "sam2.1_hiera_large.yaml"
    },
 {
        "base_model_identifier": "Unet",
        "name": "UnetCoral",
        "description": "Standard U-Net model trained on Coral dataset.",
        "model_type": "automatic",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/unet_coral_model.pth",
        "config": "app/services/segmentation/configs/unet_config.yaml"
    },
    {
        "base_model_identifier": "UnetPlusPlus",
        "name": "UnetPlusPlusCoral",
        "description": "UNet++ model trained on Coral dataset.",
        "model_type": "automatic",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/unetplusplus_coral_model.pth",
        "config": "app/services/segmentation/configs/unetplusplus_config.yaml"
    },
    {
        "base_model_identifier": "DeepLabV3PP",
        "name": "DeepLabV3PPCoral",
        "description": "DeepLabV3++ model trained on Coral dataset.",
        "model_type": "automatic",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/deeplabv3pp_coral_model.pth",
        "config": "app/services/segmentation/configs/deeplabv3pp_config.yaml"
    },
]
prompted_3D_models_to_add = [
    {
        "base_model_identifier": "SAM2Prompted3D",
        "name": "SAM2Tiny",
        "description": "A very small version of the SAM2 model optimized for speed and efficiency.",
        "model_type": "prompted_3d",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/sam2.1_hiera_tiny.pt",
        "config": "sam2.1_hiera_tiny.yaml"
    },
    {
        "base_model_identifier": "SAM2Prompted3D",
        "name": "SAM2Small",
        "description": "A small version of the SAM2 model with a balance between performance and speed.",
        "model_type": "prompted_3d",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/sam2.1_hiera_small.pt",
        "config": "sam2.1_hiera_small.yaml"
    },
    {
        "base_model_identifier": "SAM2Prompted3D",
        "name": "SAM2Large",
        "description": "A larger version of the SAM2 model designed for high accuracy.",
        "model_type": "prompted_3d",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/sam2.1_hiera_large.pt",
        "config": "sam2.1_hiera_large.yaml"
    },
    {
        "base_model_identifier": "SAM2Prompted3D",
        "name": "SAM2BasePlus",
        "description": "The Base version of the SAM2 model.",
        "model_type": "prompted_3d",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "./app/services/segmentation/weights/sam2.1_hiera_base_plus.pt",
        "config": "sam2.1_hiera_large.yaml"  #
    },
]

automatic_3D_models_to_add = [

]


def add_models_to_db():
    """Add models to the database."""
    with get_context_session() as session:
        # Combine all models into a single list
        models_to_add = (
                prompted_models_to_add +
                automatic_models_to_add +
                prompted_3D_models_to_add +
                automatic_3D_models_to_add
        )
        for model in models_to_add:
            if " " in model["name"]:
                logger.error(f"Model name '{model['name']}' contains spaces. Please use underscores instead.")
                continue
            # Check if the model is already present in the database
            existing_model = session.query(Models).filter_by(**model).first()
            if existing_model:
                logger.warning(f"Model {model['name']} already exists in the database."
                             f"Skipping to avoid duplicates. For updating models, please use the respective routes.")
                continue

            # Check that the base_model_identifier is available in the configuration. This is important so we can map
            # the model type to the correct class in the segmentation service.
            if model["base_model_identifier"] not in (AvailableModels[model["model_type"]]).keys():
                logger.error(f"Base model {model['base_model_identifier']} is not available in the configuration. "
                             f"Skipping model {model['name']}. Please add it to the AvailableModels dictionary, before"
                             f" adding models to the database.")
                continue

            # Check that weight paths and config paths are valid if they are given
            if not model["weights"]:
                logger.warning(f"Model {model['name']} does not have a weights path specified. Ignore this, if there "
                               f"is no weights file for this model.")
            elif not os.path.exists(os.path.join(Paths.base_dir, model['weights'])):
                logger.error(f"Model {model['name']} has an invalid weights path. {model['weights']} not found. "
                             f"Skipping to avoid incomplete entries.")
                continue
            if not model["config"]:
                logger.warning(f"Model {model['name']} does not have a config path specified. Ignore this, if there "
                               f"is no config file for this model.")
            elif not os.path.exists(os.path.join(Paths.base_dir, model['config'])):
                logger.error(f"Model {model['name']} has an invalid config path. {model['config']} not found. "
                             f"Skipping to avoid incomplete entries.")
                continue

            new_model = Models(**model)
            session.add(new_model)
            logger.info(f"Added model {model['name']} to the database.")

        session.commit()


if __name__ == "__main__":
    add_models_to_db()
    logger.info("All models have been added to the database.")

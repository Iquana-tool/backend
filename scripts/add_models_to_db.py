"""This script adds segmentation models to the database. Add your models with their configurations here."""
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
    "name": "SAM2 Tiny",
    "description": "A very small version of the SAM2 model optimized for speed and efficiency.",
    "model_type": "prompted",  # This should match one of the keys used in paths.py Available models class
    "version": "1.0",
    "created_at": None,  # Use a timestamp or datetime object if available
    "updated_at": None,  # Use a timestamp or datetime object if available
    "weights": "app/services/segmentation/weights/sam2.1_hiera_tiny.pth",
    "config": "app/services/segmentation/configs/sam2.1_hiera_tiny.yaml"
}


prompted_models_to_add = [
    {
        "base_model_identifier": "SAM2Prompted",
        "name": "SAM2 Tiny",
        "description": "A very small version of the SAM2 model optimized for speed and efficiency.",
        "model_type": "prompted",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "app/services/segmentation/weights/sam2.1_hiera_tiny.pth",
        "config": "app/services/segmentation/configs/sam2.1_hiera_tiny.yaml"
    },
    {
        "base_model_identifier": "SAM2Prompted",
        "name": "SAM2 Small",
        "description": "A small version of the SAM2 model with a balance between performance and speed.",
        "model_type": "prompted",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "app/services/segmentation/weights/sam2.1_hiera_small.pth",
        "config": "app/services/segmentation/configs/sam2.1_hiera_small.yaml"
    },
    {
        "base_model_identifier": "SAM2Prompted",
        "name": "SAM2 Large",
        "description": "A larger version of the SAM2 model designed for high accuracy.",
        "model_type": "prompted",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "app/services/segmentation/weights/sam2.1_hiera_large.pth",
        "config": "app/services/segmentation/configs/sam2.1_hiera_large.yaml"
    },
    {
        "base_model_identifier": "SAM2Prompted",
        "name": "SAM2 Base Plus",
        "description": "The Base version of the SAM2 model.",
        "model_type": "prompted",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "app/services/segmentation/weights/sam2.1_hiera_large.pth",
        "config": "app/services/segmentation/configs/sam2.1_hiera_large.yaml"
    },
]


automatic_models_to_add = [
    {
        "base_model_identifier": "SAM2Automatic",
        "name": "SAM2 Tiny",
        "description": "A very small version of the SAM2 model optimized for speed and efficiency.",
        "model_type": "automatic",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "app/services/segmentation/weights/sam2.1_hiera_tiny.pth",
        "config": "app/services/segmentation/configs/sam2.1_hiera_tiny.yaml"
    },
    {
        "base_model_identifier": "SAM2Automatic",
        "name": "SAM2 Small",
        "description": "A small version of the SAM2 model with a balance between performance and speed.",
        "model_type": "automatic",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "app/services/segmentation/weights/sam2.1_hiera_small.pth",
        "config": "app/services/segmentation/configs/sam2.1_hiera_small.yaml"
    },
    {
        "base_model_identifier": "SAM2Automatic",
        "name": "SAM2 Large",
        "description": "A larger version of the SAM2 model designed for high accuracy.",
        "model_type": "automatic",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "app/services/segmentation/weights/sam2.1_hiera_large.pth",
        "config": "app/services/segmentation/configs/sam2.1_hiera_large.yaml"
    },
    {
        "base_model_identifier": "SAM2Automatic",
        "name": "SAM2 Base Plus",
        "description": "The Base version of the SAM2 model.",
        "model_type": "automatic",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "app/services/segmentation/weights/sam2.1_hiera_large.pth",
        "config": "app/services/segmentation/configs/sam2.1_hiera_large.yaml"
    },
]


prompted_3D_models_to_add = [
    {
        "base_model_identifier": "SAM2Prompted3D",
        "name": "SAM2 Tiny",
        "description": "A very small version of the SAM2 model optimized for speed and efficiency.",
        "model_type": "prompted_3d",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "app/services/segmentation/weights/sam2.1_hiera_tiny.pth",
        "config": "app/services/segmentation/configs/sam2.1_hiera_tiny.yaml"
    },
    {
        "base_model_identifier": "SAM2Prompted3D",
        "name": "SAM2 Small",
        "description": "A small version of the SAM2 model with a balance between performance and speed.",
        "model_type": "prompted_3d",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "app/services/segmentation/weights/sam2.1_hiera_small.pth",
        "config": "app/services/segmentation/configs/sam2.1_hiera_small.yaml"
    },
    {
        "base_model_identifier": "SAM2Prompted3D",
        "name": "SAM2 Large",
        "description": "A larger version of the SAM2 model designed for high accuracy.",
        "model_type": "prompted_3d",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "app/services/segmentation/weights/sam2.1_hiera_large.pth",
        "config": "app/services/segmentation/configs/sam2.1_hiera_large.yaml"
    },
    {
        "base_model_identifier": "SAM2Prompted3D",
        "name": "SAM2 Base Plus",
        "description": "The Base version of the SAM2 model.",
        "model_type": "prompted_3d",
        "version": "1.0",
        "created_at": None,
        "updated_at": None,
        "weights": "app/services/segmentation/weights/sam2.1_hiera_large.pth",
        "config": "app/services/segmentation/configs/sam2.1_hiera_large.yaml"
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
            existing_model = session.query(Models).filter_by(id=model['id']).first()
            if model["model_type"] not in AvailableModels:
                logger.warning(f"Model type {model['model_type']} is not available in the configuration. "
                               f"Skipping model {model['name']}.")
                continue
            if existing_model:
                logger.warning(f"Model {model['name']} already exists in the database."
                               f"Skipping to avoid duplicates.")
                continue

            new_model = Models(**model)
            session.add(new_model)
            logger.info(f"Added model {model['name']} to the database.")

        session.commit()


if __name__ == "__main__":
    add_models_to_db()
    logger.info("All models have been added to the database.")

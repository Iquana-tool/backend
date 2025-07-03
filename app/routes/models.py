from logging import getLogger
from fastapi import APIRouter, Depends
from app.database import get_session, SessionLocal
from app.database.models import Models

logger = getLogger(__name__)  # Create a router for the models API
router = APIRouter(
    prefix="/models",
    tags=["models"],
)


@router.get("/get_prompted_models")
async def get_prompted_models(db: SessionLocal = Depends(get_session)):
    """Retrieve all prompted prompted_segmentation models from the database."""
    logger.debug("Fetching prompted prompted_segmentation models from the database.")
    models = db.query(Models).filter(Models.model_type == "prompted").all()
    if not models:
        logger.warning("No prompted prompted_segmentation models found in the database.")
        return {"message": "No prompted prompted_segmentation models found."}
    return {"success": True, "models": models}


@router.get("/get_automatic_models")
async def get_automatic_models(db: SessionLocal = Depends(get_session)):
    """Retrieve all automatic prompted_segmentation models from the database."""
    logger.debug("Fetching automatic prompted_segmentation models from the database.")
    models = db.query(Models).filter(Models.model_type == "automatic").all()
    if not models:
        logger.warning("No automatic prompted_segmentation models found in the database.")
        return {"message": "No automatic prompted_segmentation models found."}
    return {"success": True, "models": models}


@router.get("/get_prompted_3d_models")
async def get_prompted_3d_models(db: SessionLocal = Depends(get_session)):
    """Retrieve all prompted 3D prompted_segmentation models from the database."""
    logger.debug("Fetching prompted 3D prompted_segmentation models from the database.")
    models = db.query(Models).filter(Models.model_type == "prompted_3d").all()
    if not models:
        logger.warning("No prompted 3D prompted_segmentation models found in the database.")
        return {"message": "No prompted 3D prompted_segmentation models found."}
    return {"success": True, "models": models}


@router.get("/get_automatic_3d_models")
async def get_automatic_3d_models(db: SessionLocal = Depends(get_session)):
    """Retrieve all automatic 3D prompted_segmentation models from the database."""
    logger.debug("Fetching automatic 3D prompted_segmentation models from the database.")
    models = db.query(Models).filter(Models.model_type == "automatic_3d").all()
    if not models:
        logger.warning("No automatic 3D prompted_segmentation models found in the database.")
        return {"message": "No automatic 3D prompted_segmentation models found."}
    return {"success": True, "models": models}


@router.get("/get_automatic_models_for_dataset/{dataset_id}")
async def get_automatic_models_for_dataset(dataset_id: int, db: SessionLocal = Depends(get_session)):
    """Retrieve all automatic prompted_segmentation models for a specific dataset."""
    logger.debug(f"Fetching automatic prompted_segmentation models for dataset {dataset_id}.")
    models = db.query(Models).filter(
        Models.model_type == "automatic",
        Models.dataset_id == dataset_id
    ).all()
    if not models:
        logger.warning(f"No automatic prompted_segmentation models found for dataset {dataset_id}.")
        return {"message": f"No automatic prompted_segmentation models found for dataset {dataset_id}."}
    return {"success": True,
            "message": f"Found {len(models)} automatic prompted_segmentation models for dataset {dataset_id}.",
            "models": models}

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import paths
from app.routes.segmentation.image_segmentation import router as segmentation_router
from app.routes.images import router as image_router
from app.routes.mask_generation import router as mask_router
from app.routes.labels import router as label_router
from app.routes.export import router as export_router
from app.routes.datasets import router as dataset_router
from app.routes.models import router as model_router
from app.routes.segmentation.scan_segmentation import router as scan_segmentation_router
from app.database import init_db
import scripts.add_models_to_db as add_models_to_db
from logging import getLogger
import hydra


logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)


@hydra.main(version_base=None, config_path="./services/segmentation/configs")
def create_app():
    logger.debug("Creating FastAPI application")
    # Load environment variables
    load_dotenv()
    
    # Get allowed origins from environment variable
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    logger.debug(f"Allowed origins: {allowed_origins}")

    # Initialize the directories
    for directory in [directory for directory in dir(paths.Paths) if "dir" in directory and not "__" in directory]:
        os.makedirs(getattr(paths.Paths, directory), exist_ok=True)
        logger.debug(f"Created directory {getattr(paths.Paths, directory)}")

    init_db()

    app = FastAPI(
        title="Coral Segmentation API",
        description="FastAPI backend for interactive coral segmentation",
        version="0.1.0",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Root endpoint
    @app.get("/")
    async def root():
        return {"message": "This is the API for DFKI"}

    # Status endpoint
    @app.get("/status")
    async def status():
        return {"status": "ok", "message": "API is running"}

    # Include the routers
    app.include_router(dataset_router)
    app.include_router(image_router)
    app.include_router(segmentation_router)
    app.include_router(scan_segmentation_router)
    app.include_router(mask_router)
    app.include_router(label_router)
    app.include_router(export_router)
    app.include_router(model_router)

    add_models_to_db.add_models_to_db()

    return app

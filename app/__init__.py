import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import paths
from app.routes.prompted_segmentation.image_segmentation import router as prompted_segmentation_router

from app.routes.semantic_segmentation import router as automatic_general_router
from app.routes.semantic_segmentation.inference import router as automatic_inference_router
from app.routes.semantic_segmentation.training import router as automatic_training_router
from app.routes.semantic_segmentation.models import router as automatic_models_router
from app.routes.semantic_segmentation.upload_data import router as automatic_upload_router

from app.routes.images import router as image_router
from app.routes.masks import router as mask_router
from app.routes.contours import router as contour_router
from app.routes.labels import router as label_router
from app.routes.export import router as export_router
from app.routes.datasets import router as dataset_router
from app.routes.prompted_segmentation.scan_segmentation import router as scan_segmentation_router
from app.routes.image_annotation_session import router as image_annotation_session_router
from app.database import init_db
from logging import getLogger


logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
        description="FastAPI backend for interactive coral prompted_segmentation",
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
    app.include_router(prompted_segmentation_router)
    app.include_router(automatic_general_router)
    app.include_router(automatic_training_router)
    app.include_router(automatic_inference_router)
    app.include_router(automatic_models_router)
    app.include_router(automatic_upload_router)
    app.include_router(image_annotation_session_router)
    app.include_router(scan_segmentation_router)
    app.include_router(mask_router)
    app.include_router(contour_router)
    app.include_router(label_router)
    app.include_router(export_router)

    return app

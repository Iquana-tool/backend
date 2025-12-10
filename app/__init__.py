import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import paths
from app.routes.services import semantic_router
from app.routes.services.prompted_router import router as prompted_segmentation_router
from app.routes.services.completion_router import router as completion_segmentation_router
from app.routes.services.semantic_router import router as semantic_segmentation_router
from app.routes.general.auth import router as auth_router
from app.routes.general.images import router as image_router
from app.routes.general.masks import router as mask_router
from app.routes.general.contours import router as contour_router
from app.routes.general.labels import router as label_router
from app.routes.general.export import router as export_router
from app.routes.general.datasets import router as dataset_router
from app.routes.websockets.image_annotation_session import router as image_annotation_session_router
from app.database import init_db
from logging import getLogger


logger = getLogger(__name__)

def create_app():
    logger.setLevel(logging.DEBUG)
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
    # General Routers
    app.include_router(auth_router)
    app.include_router(dataset_router)
    app.include_router(image_router)
    app.include_router(image_annotation_session_router)
    app.include_router(mask_router)
    app.include_router(contour_router)
    app.include_router(label_router)
    app.include_router(export_router)

    # Services; Add your own service here!
    app.include_router(prompted_segmentation_router)
    app.include_router(completion_segmentation_router)
    app.include_router(semantic_segmentation_router)

    return app

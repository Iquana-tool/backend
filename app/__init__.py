import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import config
from app.routes.segmentation import router as segmentation_router
from app.routes.images import router as image_router
from app.routes.mask_generation import router as mask_router
from app.database import init_db
from logging import getLogger


logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_app():
    # Load environment variables
    load_dotenv()
    
    # Get allowed origins from environment variable
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    logger.debug(f"Allowed origins: {allowed_origins}")

    # Initialize the directories
    for directory in [directory for directory in dir(config.Paths) if "dir" in directory and not "__" in directory]:
        os.makedirs(getattr(config.Paths, directory), exist_ok=True)
        logger.debug(f"Created directory {getattr(config.Paths, directory)}")

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
    app.include_router(segmentation_router)
    app.include_router(image_router)
    app.include_router(mask_router)

    return app
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from app.routes.segmentation import router as segmentation_router
from app.routes.images import router as image_router
from app.database import init_db


def create_app():
    # Initialize the directories
    for directory in [directory for directory in dir(config.Paths) if "dir" in directory and not "__" in directory]:
        os.makedirs(getattr(config.Paths, directory), exist_ok=True)

    init_db()

    app = FastAPI(
        title="Coral Segmentation API",
        description="FastAPI backend for interactive coral segmentation",
        version="0.1.0",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the router
    app.include_router(segmentation_router, prefix="/api")
    app.include_router(image_router, prefix="/api")

    return app
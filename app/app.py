from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import logging

from app.routes.segmentation import router as segmentation_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get absolute path to the project root - FIXED: going up one level from app directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger.info(f"Base directory: {BASE_DIR}")

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

# Ensure directories exist
directories = {
    "database": os.path.join(BASE_DIR, "database"),
    "uploads": os.path.join(BASE_DIR, "uploads"),
    "output_masks": os.path.join(BASE_DIR, "output_masks"),
    "selected_masks": os.path.join(BASE_DIR, "selected_masks"),
    "polyps_masks": os.path.join(BASE_DIR, "polyps_masks"),
    "fine_tune_masks": os.path.join(BASE_DIR, "fine_tune_masks")
}

# Create directories if they don't exist
for dir_path in directories.values():
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Ensuring directory exists: {dir_path}")

# Mount static directories
for route, dir_path in directories.items():
    logger.info(f"Mounting static directory: {dir_path} at /{route}")
    if os.path.exists(dir_path):
        # List contents of directory for debugging
        files = os.listdir(dir_path)
        logger.info(f"Contents of {dir_path}: {files}")
        try:
            app.mount(
                f"/{route}",
                StaticFiles(directory=dir_path, check_dir=True),
                name=route
            )
            logger.info(f"Successfully mounted {route}")
        except Exception as e:
            logger.error(f"Failed to mount {dir_path}: {str(e)}")
    else:
        logger.error(f"Directory does not exist: {dir_path}")

# Include the router
app.include_router(segmentation_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Coral Segmentation API is running"}
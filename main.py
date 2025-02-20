from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    "database": os.path.join(BASE_DIR, "database")
}

# Create directories if they don't exist
for dir_path in directories.values():
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Ensuring directory exists: {dir_path}")

# Include the router
app.include_router(segmentation_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Coral Segmentation API is running"}
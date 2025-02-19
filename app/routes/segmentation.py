import os
import uuid
import shutil
import io
import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import torch
from torchvision import transforms
from app.services.vector_db import vector_db
from app.encoder.dummy_encoder import DummyImageEncoder  # Import dummy encoder
from app.image_segmentation import select_mask
from app.further_segmentation import segment_selected_region, create_polygon_and_filter_masks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Constants for directory paths - using absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_masks")
SELECTED_DIR = os.path.join(BASE_DIR, "selected_masks")
POLYPS_DIR = os.path.join(BASE_DIR, "polyps_masks")
FINE_TUNE_DIR = os.path.join(BASE_DIR, "fine_tune_masks")
DATABASE_DIR = os.path.join(BASE_DIR, "database")

# Initialize dummy encoder
dummy_encoder = DummyImageEncoder()

# Ensure all required directories exist
def ensure_directories():
    directories = [UPLOAD_DIR, OUTPUT_DIR, SELECTED_DIR, POLYPS_DIR, FINE_TUNE_DIR, DATABASE_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensuring directory exists: {directory}")

def validate_image_file(filename: str) -> bool:
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)

@router.get("/database-images")
def list_database_images():
    """Lists all images stored in the database directory."""
    ensure_directories()
    images = []

    if os.path.exists(DATABASE_DIR):
        all_files = os.listdir(DATABASE_DIR)
        for idx, f in enumerate(all_files):
            if validate_image_file(f):
                file_path = os.path.join(DATABASE_DIR, f)
                if os.path.exists(file_path):
                    images.append({"id": idx, "name": f, "path": f"/database/{f}"})
    
    return {"images": images}

@router.get("/database/{filename}")
async def get_database_image(filename: str):
    """Retrieves an image from the database."""
    if not validate_image_file(filename):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = os.path.join(DATABASE_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(path=file_path)

@router.get("/results")
def list_segmentation_results():
    """Lists segmentation results stored in various directories."""
    ensure_directories()
    base_url = "http://127.0.0.1:8000"
    
    results = {
        "selected": [],
        "polyps": [],
        "fine_tune": []
    }

    def get_images_from_dir(directory: str, url_prefix: str) -> List[str]:
        if not os.path.exists(directory):
            return []
        return [f"{base_url}/{url_prefix}/{f}" for f in os.listdir(directory) if validate_image_file(f)]

    results["selected"] = get_images_from_dir(SELECTED_DIR, "selected_masks")
    results["polyps"] = get_images_from_dir(POLYPS_DIR, "polyps_masks")
    results["fine_tune"] = get_images_from_dir(FINE_TUNE_DIR, "fine_tune_masks")

    return results

@router.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    """Processes an uploaded image, generates dummy embeddings, and stores them in FAISS."""
    try:
        ensure_directories()

        if not validate_image_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPG and PNG files are allowed.")

        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")

        # Convert image to tensor (dummy input)
        image_tensor = torch.rand((3, 224, 224))  # Simulating an image input

        # Generate dummy embedding
        embedding = dummy_encoder.encode(image_tensor).flatten()  # Convert to 1D

        # Debugging outputs
        print(f"🔹 Dummy Embedding shape: {embedding.shape}")
        print(f"🔹 Embedding sample (first 5 values): {embedding[:5]}")

        # Store embedding in vector database
        vector_db.add_embedding(file_path, embedding)

        # Debugging: Verify vector database after adding embedding
        vector_db.debug_index()

        return {"message": "Image segmented and embedding stored", "image_path": file_path}

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_similar_images(file: UploadFile = File(...), top_k: int = 5):
    """Finds similar images using stored embeddings in FAISS."""
    try:
        ensure_directories()

        if not validate_image_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type.")

        # Convert uploaded image to tensor (dummy input)
        image_tensor = torch.rand((3, 224, 224))  # Simulating an image input

        # Ensure FAISS has at least 1 vector
        if vector_db.index.ntotal == 0:
            raise HTTPException(status_code=500, detail="Vector database is empty. Upload images first.")

        # Generate dummy query embedding
        query_embedding = dummy_encoder.encode(image_tensor).flatten()  # Convert to 1D

        # Debugging outputs
        print(f"🔍 Query Embedding shape: {query_embedding.shape}")
        print(f"🔍 Query Embedding sample: {query_embedding[:5]}")

        # Retrieve similar images
        results = vector_db.search(query_embedding, top_k)
        print(f"📌 Search Results: {results}")

        return {"similar_images": results}

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

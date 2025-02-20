import os
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List

from app.services.image_loader import save_uploaded_image, validate_image_file
import config

logger = logging.getLogger(__name__)
router = APIRouter(tags=["images"])

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file"""
    try:
        file_path = await save_uploaded_image(file)
        if file_path is None:
            raise HTTPException(status_code=400, detail="Invalid file or upload failed")
            
        return {
            "success": True,
            "file_path": file_path,
            "filename": os.path.basename(file_path)
        }
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
def list_images():
    """List all uploaded images"""
    try:
        if not os.path.exists(config.UPLOADS_DIR):
            return {"images": []}
            
        image_files = []
        for filename in os.listdir(config.UPLOADS_DIR):
            if validate_image_file(filename):
                file_path = os.path.join(config.UPLOADS_DIR, filename)
                image_files.append({
                    "name": filename,
                    "path": f"/uploads/{filename}",
                    "size": os.path.getsize(file_path)
                })
                
        return {"images": image_files}
    except Exception as e:
        logger.error(f"List images error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{filename}")
async def get_image(filename: str):
    """Get a specific image"""
    if not validate_image_file(filename):
        raise HTTPException(status_code=400, detail="Invalid file type")
        
    file_path = os.path.join(config.UPLOADS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
        
    return FileResponse(file_path)
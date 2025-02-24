import os
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.services.dataloader import save_image
from app.database.images import Images
import config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/images", tags=["images"])

@router.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file"""
    try:
        image_id = await save_image(file)
        if image_id is None:
            raise HTTPException(status_code=400, detail="Invalid file or upload failed")
            
        return {
            "success": True,
            "file_path": image_id
        }
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list_images")
def list_images():
    """List all uploaded images"""
    try:
        return {"images": Images.query.all().id.tolist()}
    except Exception as e:
        logger.error(f"List images error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get_image({image_id}")
async def get_image(image_id: int):
    """Get a specific image"""
    file_path = Images.query.filter_by(id=image_id).first().path
    return FileResponse(file_path)
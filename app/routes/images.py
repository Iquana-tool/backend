import os
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.services.dataloader import save_image
from app.database import get_db  # Import the dependency for the database session
from app.models import Images  # Ensure this is the correct import for your models

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/images", tags=["images"])

@router.post("/upload_image")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload an image file"""
    try:
        image_id = await save_image(file)
        if image_id is None:
            raise HTTPException(status_code=400, detail="Invalid file or upload failed")

        # Save the image record to the database
        new_image = Images(id=image_id, path=file.filename, type=file.content_type, size=file.size)
        db.add(new_image)
        db.commit()

        return {
            "success": True,
            "file_path": image_id
        }
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list_images")
def list_images(db: Session = Depends(get_db)):
    """List all uploaded images"""
    try:
        images = db.query(Images).all()
        return {"images": [image.id for image in images]}
    except Exception as e:
        logger.error(f"List images error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get_image/{image_id}")
async def get_image(image_id: int, db: Session = Depends(get_db)):
    """Get a specific image"""
    try:
        image = db.query(Images).filter(Images.id == image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(image.path)
    except Exception as e:
        logger.error(f"Get image error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

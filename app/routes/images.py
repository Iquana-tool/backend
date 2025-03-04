import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from sqlalchemy.orm import Session

from app.services.database_access import save_image_to_disk_and_db, load_image_as_base64_from_disk
from app.services.database_access import delete_image_from_disk_and_db
from app.database import get_session
from app.database.images import Images

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/images", tags=["images"])


@router.post("/upload_image")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_session)):
    """Upload an image file"""
    try:
        image_id = await save_image_to_disk_and_db(file)
        if image_id is None:
            raise HTTPException(status_code=400, detail="Invalid file or upload failed")

        return {
            "success": True,
            "image_id": image_id,
            "message": f"Successfully uploaded image. Assigned id {image_id}"
        }
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete_image/{image_id}")
async def delete_image(image_id: int):
    try:
        delete_image_from_disk_and_db(image_id)
        return {"success": True,
                "message": f"Deleted image {image_id}."}
    except Exception as e:
        logger.error(f"Delete image error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list_images")
def list_images(db: Session = Depends(get_session)):
    """List all uploaded image ids"""
    try:
        images = db.query(Images).all()
        return {
            "success": True,
            "images": images
        }
    except Exception as e:
        logger.error(f"List images error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get_image/{image_id}", response_model=dict[int, str])
async def get_image(image_id: int):
    """Get images via ids.

    Args:
        image_id: Image ID to retrieve.

    Returns:
        A dict mapping from image ID to base64 encoded image.
    """
    try:
        response = {}
        image = load_image_as_base64_from_disk(image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        response[image_id] = image
        return response
    except Exception as e:
        logger.error(f"Get image error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from sqlalchemy.orm import Session

from app.schemas.images import ImageID, Base64Image
from app.services.database_access import save_image, load_image_as_base64
from app.services.database_access import delete_image_files
from app.database import get_session
from app.database.images import Images

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/images", tags=["images"])


@router.post("/upload_image")
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_session)):
    """Upload an image file"""
    try:
        image_id = await save_image(file)
        if image_id is None:
            raise HTTPException(status_code=400, detail="Invalid file or upload failed")

        return {
            "success": True,
            "image_id": ImageID(image_id=image_id),
            "message": f"Successfully uploaded image. Assigned id {image_id}"
        }
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete_image/{image_id}")
async def delete_image(image_id: ImageID):
    try:
        delete_image_files(image_id.image_id)
        return {"success": True,
                "message": f"Deleted image {image_id.image_id}."}
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


@router.get("/get_images/{image_ids}", response_model=list[Base64Image])
async def get_images(image_ids: list[ImageID]):
    """Get images via ids.

    Args:
        image_ids: List of image IDs to retrieve.

    Returns:
        A list of Base64Image objects containing the image ID and the image as a base64 string.
    """
    try:
        if len(image_ids) > 100:
            logger.warning("Requesting more than 100 images at once. This may take a while.")
        images = []
        for image_id in image_ids:
            image = load_image_as_base64(image_id.image_id)
            if not image:
                raise HTTPException(status_code=404, detail="Image not found")
            images.append(Base64Image(image_id=image_id, image=image))
        return images
    except Exception as e:
        logger.error(f"Get image error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

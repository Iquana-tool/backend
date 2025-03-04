import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from sqlalchemy.orm import Session

from app.schemas.image_processing import CutoutsRequest, CutoutsResponse, ImagesResponse, ImagesRequest
from app.services.database_access import save_image, load_image_as_base64
from app.services.cutouts import cutout_objects_on_mask_from_image
from app.services.database_access import load_image, delete_image_files
from app.database import get_session
from app.database.images import Images, Cutouts
from app.schemas.util import validate_request

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
            "image_id": image_id,
            "message": f"Successfully uploaded image. Assigned id {image_id}"
        }
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete_image/{image_id}")
async def delete_image(image_id: int):
    try:
        delete_image_files(image_id)
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


@router.post("/get_images", response_model=dict[int, dict])
async def get_images(image_ids: list[int], db: Session = Depends(get_session)):
    """Get images via ids.

    Args:
        image_ids: List of image IDs to retrieve.

    Returns:
        A dictionary that has ids as keys and dictionaries as values.
        The dictionary has base64, height, and width as keys.
    """
    try:
        if len(image_ids) > 100:
            logger.warning("Requesting more than 100 images at once. This may take a while.")
        images = {}
        for image_id in image_ids:
            image = load_image_as_base64(image_id)
            if not image:
                raise HTTPException(status_code=404, detail="Image not found")
            images[image_id] = {
                "base64": image,
                "height": db.query(Images).filter_by(id=image_id).first().height,
                "width": db.query(Images).filter_by(id=image_id).first().width
            }
        return images
    except Exception as e:
        logger.error(f"Get image error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get_cutouts")
async def get_cutouts(request: CutoutsRequest, db: Session = Depends(get_session)):
    """Get cutouts from an image"""
    try:
        cutouts = cutout_objects_on_mask_from_image(load_image(request.image_id),
                                                    request.mask,
                                                    request.resize_factor,
                                                    request.darken_outside_contours,
                                                    request.darkening_factor)
        for cutout, lower_left_x, lower_left_y in cutouts:
            db.add(Cutouts(image_id=request.image_id, width=cutout.shape[1], height=cutout.shape[0],
                           lower_left_x=lower_left_x, lower_left_y=lower_left_y))
        return CutoutsResponse(cutouts=cutouts)
    except Exception as e:
        logger.error(f"Get cutouts error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from sqlalchemy.orm import Session

from app.schemas.image_processing import CutoutsRequest, CutoutsResponse, ImagesResponse, ImagesRequest
from app.services.dataloader import save_image
from app.services.cutouts import cutout_objects_on_mask_from_image
from app.services.dataloader import load_image
from app.database import get_session  # Import the dependency for the database session
from app.database.images import Images  # Ensure this is the correct import for your models
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
def list_images(db: Session = Depends(get_session)):
    """List all uploaded image ids"""
    try:
        images = Images.query.id.all()
        return {"images": images}
    except Exception as e:
        logger.error(f"List images error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get_images")
async def get_image(request: Request, db: Session = Depends(get_session)):
    """Get a specific image"""
    try:
        validated_data = validate_request(await request.json(), ImagesRequest)
        if len(validated_data.ids) > 100:
            logger.warning("Requesting more than 100 images at once. This may take a while.")
        images = {}
        for image_id in validated_data.ids:
            image = load_image(image_id)
            if not image:
                raise HTTPException(status_code=404, detail="Image not found")
            images[image_id] = image
        return ImagesResponse(images=images)
    except Exception as e:
        logger.error(f"Get image error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get_cutouts")
async def get_cutouts(request: Request, db: Session = Depends(get_session)):
    """Get cutouts from an image"""
    try:
        validated_data = validate_request(await request.json(), CutoutsRequest)
        cutouts = cutout_objects_on_mask_from_image(load_image(validated_data.image_id),
                                                    validated_data.mask,
                                                    validated_data.resize_factor,
                                                    validated_data.darken_outside_contours,
                                                    validated_data.darkening_factor)
        # Process the image here
        return CutoutsResponse(cutouts=cutouts)
    except Exception as e:
        logger.error(f"Get cutouts error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

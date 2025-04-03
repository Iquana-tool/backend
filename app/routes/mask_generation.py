import json
import logging
import os.path

import cv2
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

import config
from app.database import get_session
from app.database.mask_generation import Masks, Labels, Contours
from app.schemas.segmentation_and_masks import ContourModel
from app.services.encoding import base64_decode_string, base64_encode_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/masks", tags=["masks"])


@router.put("/create_mask/{image_id}")
async def create_mask(image_id: int, db: Session = Depends(get_session)):
    try:
        # Create a new mask
        new_mask = Masks(image_id=image_id)
        db.add(new_mask)
        db.commit()
        return {
            "success": True,
            "message": "Mask created successfully.",
            "mask_id": new_mask.id
        }
    except Exception as e:
        logger.error(f"Error creating mask: {e}")
        raise HTTPException(status_code=500, detail="Error creating mask.")


@router.get("/get_contours_of_mask/{mask_id}")
async def get_contours_of_mask(mask_id: int, db: Session = Depends(get_session)):
    contours = db.query(Contours).filter_by(mask_id=mask_id).all()
    if not contours:
        raise HTTPException(status_code=404, detail="No contours found for mask.")
    return contours


@router.get("/get_mask/{mask_id}")
async def get_mask(mask_id: int, db: Session = Depends(get_session)):
    mask = db.query(Masks).filter_by(id=mask_id).first()
    if mask is None:
        raise HTTPException(status_code=404, detail="Mask not found.")
    mask_path = os.path.join(config.Paths.masks_dir,
                             str(mask.image_id),
                             f"{mask.mask_label}_{mask.counter}.png")
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Mask file not found.")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return base64_encode_image(mask)


@router.delete("/delete_mask/{mask_id}")
async def delete_mask(mask_id: int, db: Session = Depends(get_session)):
    mask = db.query(Masks).filter_by(id=mask_id).first()
    if mask is None:
        raise HTTPException(status_code=404, detail="Mask not found.")
    db.delete(mask)
    db.commit()
    return {"success": True, "message": "Mask deleted successfully."}


@router.get("/get_masks_for_image/{image_id}")
async def get_masks_for_image(image_id: int, db: Session = Depends(get_session)):
    masks = db.query(Masks).filter_by(image_id=image_id).all()
    if not masks:
        raise HTTPException(status_code=404, detail="No masks found for image.")
    return masks


@router.post("/add_contour")
async def add_contour(mask_id: int, contour: ContourModel, db: Session = Depends(get_session)):
    try:
        # Check if mask exists
        existing_mask = db.query(Masks).filter_by(id=mask_id).first()
        if not existing_mask:
            mask_id = await create_mask(db=db)

        # Create a new contour
        coords = {"x": contour.x, "y": contour.y}
        new_contour = Contours(
            mask_id=mask_id,
            coords=json.dumps(coords),
            label=contour.label,
            area=contour.area,
            perimeter=contour.perimeter,
            circularity=contour.circularity,
            diameters=json.dumps(contour.diameters),
        )
        db.add(new_contour)
        db.commit()
        return {
            "success": True,
            "message": "Contour added successfully.",
            "contour_id": new_contour.id
        }
    except Exception as e:
        logger.error(f"Error adding contour: {e}")
        raise HTTPException(status_code=500, detail="Error adding contour.")


@router.delete("/delete_contour/{contour_id}")
async def delete_contour(contour_id: int, db: Session = Depends(get_session)):
    try:
        # Check if contour exists
        existing_contour = db.query(Contours).filter_by(id=contour_id).first()
        if not existing_contour:
            raise HTTPException(status_code=404, detail="Contour not found.")

        # Delete the contour
        db.delete(existing_contour)
        db.commit()
        return {
            "success": True,
            "message": "Contour deleted successfully."
        }
    except Exception as e:
        logger.error(f"Error deleting contour: {e}")
        raise HTTPException(status_code=500, detail="Error deleting contour.")

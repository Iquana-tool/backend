import logging
import os.path

import cv2
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

import config
from app.database import get_session
from app.database.masks import Masks
from app.schemas.masks import MaskRequest
from app.services.encoding import base64_decode_string, base64_encode_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/masks", tags=["masks"])


@router.post("/save_mask")
async def save_mask(request: MaskRequest, db: Session = Depends(get_session)):
    try:
        counter = db.query(Masks).filter_by(image_id=request.image_id, mask_label=request.label).count()
        # Save the mask to the database
        new_mask = Masks(
            image_id=request.image_id,
            mask_label=request.label,
            counter=counter
        )
        db.add(new_mask)
        db.commit()
        base_path = os.path.join(config.Paths.masks_dir, str(request.image_id))
        os.makedirs(base_path, exist_ok=True)
        mask = base64_decode_string(request.base64_mask) * 255
        cv2.imwrite(str(os.path.join(base_path,
                                     f"{new_mask.mask_label}_{new_mask.counter}.png")), mask)
        return {
            "success": True,
            "message": "Mask saved successfully.",
            "mask_id": new_mask.id
        }
    except Exception as e:
        logger.error(f"Error saving mask: {e}")
        raise HTTPException(status_code=500, detail="Error saving mask.")


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
    mask_path = os.path.join(config.Paths.masks_dir,
                             str(mask.image_id),
                             f"{mask.mask_label}_{mask.counter}.png")
    if os.path.exists(mask_path):
        os.remove(mask_path)
    db.delete(mask)
    db.commit()
    return {"success": True, "message": "Mask deleted successfully."}


@router.get("/get_masks_for_image/{image_id}")
async def get_masks_for_image(image_id: int, db: Session = Depends(get_session)):
    masks = db.query(Masks).filter_by(image_id=image_id).all()
    if not masks:
        raise HTTPException(status_code=404, detail="No masks found for image.")
    return masks

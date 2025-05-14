import json
import logging
import os.path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

import config
from app.database import get_session
from app.database.mask_generation import Masks, Contours
from app.database.images import Images
from app.database.datasets import Labels
from app.schemas.segmentation_and_masks import ContourModel
from app.services.encoding import base64_decode_string, base64_encode_image
from app.services.quantifications import ContourQuantifier
from app.services.mask_generation import (generate_mask, contour_is_enclosed_by_parent,
                                          contour_overlaps_with_existing_on_parent_level, coords_to_cv_contour)

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
    mask_arr = generate_mask(mask_id).tolist()
    return {
        "success": True,
        "mask_id": mask.id,
        "image": mask_arr
    }


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
async def add_contour(mask_id: int,
                      contour_to_add: ContourModel,
                      parent_contour_id: int = None,
                      db: Session = Depends(get_session)):
    try:
        # Check if mask exists
        existing_mask = db.query(Masks).filter_by(id=mask_id).first()
        if not existing_mask:
            mask_id = await create_mask(db=db)

        contour = coords_to_cv_contour(contour_to_add.x, contour_to_add.y)
        if not contour_is_enclosed_by_parent(contour, parent_contour_id):
            return {
                "success": False,
                "message": "Contour can not be added, because it is not enclosed by its parent contour. "
                           "Child contours must be completely inside their parent contours!",
                "contour_id": None
            }
        # Check if contour overlaps with existing contours on the same level
        contours_on_same_level = db.query(Contours).filter_by(mask_id=mask_id, parent_id=parent_contour_id).all()
        contours_on_same_level = [coords_to_cv_contour(c.coords["x"], c.coords["y"]) for c in contours_on_same_level]
        if contour_overlaps_with_existing_on_parent_level(contour, contours_on_same_level):
            return {
                "success": False,
                "message": "Contour overlaps with existing contours on the same level.",
                "contour_id": None
            }

        quantifier = ContourQuantifier().from_coordinates(contour_to_add.x, contour_to_add.y)
        new_contour = Contours(
            mask_id=mask_id,
            parent_id=parent_contour_id,
            coords=json.dumps({"x": contour_to_add.x, "y": contour_to_add.y}),
            label=contour_to_add.label,
            area=quantifier.area,
            perimeter=quantifier.perimeter,
            circularity=quantifier.circularity,
            diameters=json.dumps(quantifier.diameters),
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

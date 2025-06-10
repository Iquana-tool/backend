import json
import logging
import os.path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List

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
from app.services.database_access import get_height_width_of_image

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

        # Check if contour is enclosed by its parent contour
        parent_contour = db.query(Contours).filter_by(id=parent_contour_id).first() if parent_contour_id else None
        if parent_contour:
            if not contour_is_enclosed_by_parent(contour, coords_to_cv_contour(parent_contour.coords["x"],
                                                                               parent_contour.coords["y"])):
                return {
                    "success": False,
                    "message": "Contour can not be added, because it is not enclosed by its parent contour. "
                               "Child contours must be completely inside their parent contours!",
                    "contour_id": None
                }

        # Check if contour overlaps with existing contours on the same level
        contours_on_same_level = db.query(Contours).filter_by(mask_id=mask_id, parent_id=parent_contour_id).all()
        if contours_on_same_level:
            contours_on_same_level = [coords_to_cv_contour(c.coords["x"], c.coords["y"]) for c in contours_on_same_level]
            if contour_overlaps_with_existing_on_parent_level(contour, contours_on_same_level):
                return {
                    "success": False,
                    "message": "Contour overlaps with existing contours on the same level.",
                    "contour_id": None
                }

        # Quantify contour
        quantifier = ContourQuantifier().from_coordinates(contour_to_add.x, contour_to_add.y)
        height, width = get_height_width_of_image(existing_mask.image_id)
        rescaled_x = [int(x * width) for x in contour_to_add.x]
        rescaled_y = [int(y * height) for y in contour_to_add.y]
        quantifier = ContourQuantifier().from_coordinates(rescaled_x, rescaled_y)
        new_contour = Contours(
            mask_id=mask_id,
            parent_id=parent_contour_id,
            coords=json.dumps({"x": contour_to_add.x, "y": contour_to_add.y}),
            label=contour_to_add.label,
            area=quantifier.area,
            perimeter=quantifier.perimeter,
            circularity=quantifier.circularity,
            diameters=json.dumps(quantifier.get_diameters()),
        )

        # Add contour to the database
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
        # Check if the contour has child contours
        child_contours = db.query(Contours).filter_by(parent_id=contour_id).all()
        for child_contour in child_contours:
            # Recursively delete child contours
            await delete_contour(child_contour.id, db)
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


@router.get("/get_final_mask/{image_id}")
async def get_final_mask(image_id: int, db: Session = Depends(get_session)):
    """Get the final mask for an image. Returns the first mask found for the image."""
    try:
        # Find the first mask for this image (assuming it's the final mask)
        mask = db.query(Masks).filter_by(image_id=image_id).first()
        if not mask:
            raise HTTPException(status_code=404, detail="No final mask found for this image.")
        
        # Get the image to find its dataset_id for label lookup
        image = db.query(Images).filter_by(id=image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found.")
        
        # Get all contours for this mask
        contours = db.query(Contours).filter_by(mask_id=mask.id).all()
        
        # Get all labels for this dataset to map label IDs to names
        labels = db.query(Labels).filter_by(dataset_id=image.dataset_id).all()
        label_id_to_name = {label.id: label.name for label in labels}
        label_id_to_parent = {label.id: label.parent_id for label in labels}
        
        # Helper function to get full hierarchy name
        def get_hierarchical_label_name(label_id):
            if label_id not in label_id_to_name:
                return f"Unknown Label ({label_id})"
            
            label_name = label_id_to_name[label_id]
            parent_id = label_id_to_parent.get(label_id)
            
            # If this label has a parent, prepend parent name
            if parent_id and parent_id in label_id_to_name:
                parent_name = label_id_to_name[parent_id]
                return f"{parent_name} › {label_name}"
            
            return label_name
        
        # Format contours for frontend
        formatted_contours = []
        for contour in contours:
            coords = json.loads(contour.coords) if isinstance(contour.coords, str) else contour.coords
            diameters = json.loads(contour.diameters) if isinstance(contour.diameters, str) else contour.diameters
            label_name = get_hierarchical_label_name(contour.label)
            formatted_contours.append({
                "id": contour.id,
                "x": coords["x"],
                "y": coords["y"],
                "label": contour.label,
                "label_name": label_name,
                "area": contour.area,
                "perimeter": contour.perimeter,
                "circularity": contour.circularity,
                "diameters": diameters
            })
        
        return {
            "success": True,
            "mask_id": mask.id,
            "image_id": image_id,
            "contours": formatted_contours
        }
    except Exception as e:
        logger.error(f"Error getting final mask: {e}")
        raise HTTPException(status_code=500, detail="Error getting final mask.")


@router.post("/create_final_mask/{image_id}")
async def create_final_mask(image_id: int, db: Session = Depends(get_session)):
    """Create a final mask for an image."""
    try:
        # Check if image exists
        image = db.query(Images).filter_by(id=image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found.")
        
        # Check if a mask already exists for this image
        existing_mask = db.query(Masks).filter_by(image_id=image_id).first()
        if existing_mask:
            return {
                "success": True,
                "message": "Final mask already exists.",
                "mask_id": existing_mask.id
            }
        
        # Create a new mask
        new_mask = Masks(image_id=image_id)
        db.add(new_mask)
        db.commit()
        
        return {
            "success": True,
            "message": "Final mask created successfully.",
            "mask_id": new_mask.id
        }
    except Exception as e:
        logger.error(f"Error creating final mask: {e}")
        raise HTTPException(status_code=500, detail="Error creating final mask.")


@router.post("/add_contour_to_final_mask/{image_id}")
async def add_contour_to_final_mask(image_id: int, contour_to_add: ContourModel, db: Session = Depends(get_session)):
    """Add a single contour to the final mask for an image."""
    try:
        # Get or create the final mask for this image
        mask = db.query(Masks).filter_by(image_id=image_id).first()
        if not mask:
            # Create a new mask if none exists
            mask = Masks(image_id=image_id)
            db.add(mask)
            db.commit()
        
        # Add the contour using the existing add_contour logic
        contour = coords_to_cv_contour(contour_to_add.x, contour_to_add.y)
        
        # Quantify contour
        height, width = get_height_width_of_image(image_id)
        rescaled_x = [int(x * width) for x in contour_to_add.x]
        rescaled_y = [int(y * height) for y in contour_to_add.y]
        quantifier = ContourQuantifier().from_coordinates(rescaled_x, rescaled_y)
        
        new_contour = Contours(
            mask_id=mask.id,
            parent_id=None,  # Final mask contours don't have parents
            coords=json.dumps({"x": contour_to_add.x, "y": contour_to_add.y}),
            label=contour_to_add.label,
            area=quantifier.area,
            perimeter=quantifier.perimeter,
            circularity=quantifier.circularity,
            diameters=json.dumps(quantifier.get_diameters()),
        )
        
        db.add(new_contour)
        db.commit()
        
        return {
            "success": True,
            "message": "Contour added to final mask successfully.",
            "mask_id": mask.id,
            "contour_id": new_contour.id
        }
    except Exception as e:
        logger.error(f"Error adding contour to final mask: {e}")
        raise HTTPException(status_code=500, detail="Error adding contour to final mask.")


@router.post("/add_contours_to_final_mask/{image_id}")
async def add_contours_to_final_mask(image_id: int, request_data: dict, db: Session = Depends(get_session)):
    """Add multiple contours to the final mask for an image."""
    try:
        # Extract contours from request data
        contours_data = request_data.get("contours", [])
        if not contours_data:
            raise HTTPException(status_code=400, detail="No contours provided.")
        
        # Get or create the final mask for this image
        mask = db.query(Masks).filter_by(image_id=image_id).first()
        if not mask:
            # Create a new mask if none exists
            mask = Masks(image_id=image_id)
            db.add(mask)
            db.commit()
        
        added_contour_ids = []
        height, width = get_height_width_of_image(image_id)
        
        for contour_data in contours_data:
            try:
                # Validate contour data
                if not all(key in contour_data for key in ["x", "y", "label"]):
                    logger.warning(f"Skipping invalid contour data: {contour_data}")
                    continue
                
                # Create ContourModel from data
                contour_model = ContourModel(
                    x=contour_data["x"],
                    y=contour_data["y"],
                    label=contour_data["label"]
                )
                
                # Quantify contour
                rescaled_x = [int(x * width) for x in contour_model.x]
                rescaled_y = [int(y * height) for y in contour_model.y]
                quantifier = ContourQuantifier().from_coordinates(rescaled_x, rescaled_y)
                
                new_contour = Contours(
                    mask_id=mask.id,
                    parent_id=None,  # Final mask contours don't have parents
                    coords=json.dumps({"x": contour_model.x, "y": contour_model.y}),
                    label=contour_model.label,
                    area=quantifier.area,
                    perimeter=quantifier.perimeter,
                    circularity=quantifier.circularity,
                    diameters=json.dumps(quantifier.get_diameters()),
                )
                
                db.add(new_contour)
                db.commit()
                added_contour_ids.append(new_contour.id)
                
            except Exception as contour_error:
                logger.error(f"Error adding individual contour: {contour_error}")
                continue
        
        return {
            "success": True,
            "message": f"Successfully added {len(added_contour_ids)} contours to final mask.",
            "mask_id": mask.id,
            "contour_ids": added_contour_ids
        }
    except Exception as e:
        logger.error(f"Error adding contours to final mask: {e}")
        raise HTTPException(status_code=500, detail="Error adding contours to final mask.")

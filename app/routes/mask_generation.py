import json
import logging

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.mask_generation import Masks, Contours
from app.schemas.segmentation.contours_and_quantifications import ContourModel
from app.services.quantifications import ContourQuantifier
from app.services.mask_generation import (generate_mask, contour_is_enclosed_by_parent,
                                          contour_overlaps_with_existing_on_parent_level, coords_to_cv_contour)
from app.services.database_access import get_height_width_of_image
from app.services.labels import get_hierarchical_label_name

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/masks", tags=["masks"])


@router.put("/create_mask/{image_id}")
async def create_mask(image_id: int, db: Session = Depends(get_session)):
    try:
        # Check if mask already exists for the image
        existing_mask = db.query(Masks).filter_by(image_id=image_id).first()
        if existing_mask:
            return {
                "success": False,
                "message": "Mask already exists for this image.",
                "mask_id": existing_mask.id
            }
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
        "mask_id": mask_id,
        "contours": formatted_contours
    }


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
            contours_on_same_level = [coords_to_cv_contour(c.coords["x"], c.coords["y"]) for c in
                                      contours_on_same_level]
            if contour_overlaps_with_existing_on_parent_level(contour, contours_on_same_level):
                return {
                    "success": False,
                    "message": "Contour overlaps with existing contours on the same level.",
                    "contour_id": None
                }

        # Quantify contour
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


@router.post("/add_contours")
async def add_contours(mask_id: int,
                       contours_to_add: list[ContourModel],
                       parent_contour_id: int = None,
                       db: Session = Depends(get_session)):
    failed = []
    added_ids = []
    for contour_to_add in contours_to_add:
        result = await add_contour(mask_id, contour_to_add, parent_contour_id, db)
        if not result["success"]:
            failed.append({
                "contour": contour_to_add,
                "error": result["message"]
            })
        else:
            added_ids.append(result["contour_id"])
    if failed:
        return {
            "success": False,
            "message": f"Added {len(added_ids)} contours. Failed to add {len(failed)} contours.",
            "mask_id": mask_id,
            "failed": failed,
            "added_ids": added_ids
        }
    else:
        return {
            "success": True,
            "message": "All contours added successfully.",
            "mask_id": mask_id,
            "added_ids": added_ids,
            "failed": []
        }

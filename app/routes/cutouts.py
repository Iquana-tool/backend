import logging

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.images import Images
from app.schemas.cutouts import CutoutsRequest
from app.services.cutouts import cutout_objects_on_mask_from_image
from app.services.database_access import load_image_as_array_from_disk
from app.services.database_access import save_image_to_disk_and_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cutouts", tags=["cutouts"])


@router.get("/get_cutouts")
async def get_cutouts(request: CutoutsRequest, db: Session = Depends(get_session)):
    """Get cutouts from an image"""
    try:
        cutouts = cutout_objects_on_mask_from_image(load_image_as_array_from_disk(request.image_id),
                                                    request.mask,
                                                    request.resize_factor,
                                                    request.darken_outside_contours,
                                                    request.darkening_factor)
        cutout_ids = []
        for cutout, lower_left_x, lower_left_y in cutouts:
            cutout_id = await save_image_to_disk_and_db(cutout)
            cutout_ids.append(cutout_id)
            db.add(Images(
                id=cutout_id,
                filename=f"{cutout_id}.png",
                width=cutout.shape[1],
                height=cutout.shape[0],
                parent_image_id=request.image_id,
                lower_left_x=lower_left_x,
                lower_left_y=lower_left_y))
            db.commit()
        return {
            "success": True,
            "message": "Cutouts successfully created!",
            "cutout_ids": cutout_ids
        }
    except Exception as e:
        logger.error(f"Get cutouts error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

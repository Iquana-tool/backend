from app.services.database_access import load_image
from app.database import get_session
from app.database.cutouts import Cutouts
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from sqlalchemy.orm import Session

from app.schemas.cutouts import CutoutsRequest
from app.services.database_access import save_image, load_image_as_base64
from app.services.cutouts import cutout_objects_on_mask_from_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cutouts", tags=["cutouts"])


@router.get("/get_cutouts")
async def get_cutouts(request: CutoutsRequest, db: Session = Depends(get_session)):
    """Get cutouts from an image"""
    try:
        cutouts = cutout_objects_on_mask_from_image(load_image(request.image_id),
                                                    request.mask,
                                                    request.resize_factor,
                                                    request.darken_outside_contours,
                                                    request.darkening_factor)
        for cutout, lower_left_x, lower_left_y in cutouts:
            db.add(Cutouts(image_id=request.image_id.image_id,
                           lower_left_x=lower_left_x, lower_left_y=lower_left_y))
            db.commit()
        return
    except Exception as e:
        logger.error(f"Get cutouts error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

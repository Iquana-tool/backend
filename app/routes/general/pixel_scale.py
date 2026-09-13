"""HTTP routes for reading and setting the physical pixel scale of an image.

Each handler validates the request shape, enforces the corresponding dataset
permission, calls the service, and maps domain exceptions to HTTP status codes.
"""
from logging import getLogger

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_session
from app.exceptions import DatasetNotFoundError, ImageNotFoundError, InvalidScaleError
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services.auth import get_current_user
from app.services.permissions import ensure_permission, ensure_permission_for, require
from app.services.scale_computation import (
    apply_scale_to_dataset,
    get_image_scale,
    set_image_scale,
    set_scale_from_drawn_line,
)

router = APIRouter(prefix="/scale", tags=["scale"])
logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class SetScaleRequest(BaseModel):
    """Body for the manual set-scale endpoint."""
    image_id: int
    scale_x: float = Field(..., gt=0, description="Physical size of one pixel along x.")
    scale_y: float = Field(..., gt=0, description="Physical size of one pixel along y.")
    unit: str = Field(..., min_length=1, description="Length unit, e.g. 'mm' or 'µm'.")


class DrawnLineScaleRequest(BaseModel):
    """Body for the drawn-line calibration endpoint."""
    image_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    known_distance: float = Field(..., gt=0, description="Real-world distance between the two points.")
    unit: str = Field(..., min_length=1, description="Unit of known_distance, e.g. 'mm'.")


class DatasetScaleRequest(BaseModel):
    """Body for the bulk dataset scale endpoint."""
    dataset_id: int
    scale_x: float = Field(..., gt=0)
    scale_y: float = Field(..., gt=0)
    unit: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/get_pixel_scale/{image_id}")
async def get_pixel_scale(
    image_id: int,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(require(Permission.IMAGE_READ, "image_id")),
):
    """Return the current physical scale stored for an image."""
    try:
        return get_image_scale(db, image_id)
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    # All other exceptions bubble up → FastAPI returns 500


@router.post("/set_pixel_scale")
async def set_pixel_scale(
    body: SetScaleRequest,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Set the physical scale for an image and mark scale-dependent metrics stale.

    Returns the stored scale values on success.
    """
    ensure_permission_for(user, "image_id", body.image_id, Permission.PIXEL_SCALE_SET, db)
    try:
        result = set_image_scale(db, body.image_id, body.scale_x, body.scale_y, body.unit,
                                 username=user.username)
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidScaleError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"message": "Scale set successfully.", **result}


@router.post("/set_pixel_scale_via_drawn_line")
async def set_pixel_scale_via_drawn_line(
    body: DrawnLineScaleRequest,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Compute and store the physical scale from a user-drawn calibration line.

    The user provides two pixel coordinates on the image and the known real-world
    distance between them. The service computes the scale and persists it.
    """
    ensure_permission_for(user, "image_id", body.image_id, Permission.PIXEL_SCALE_SET, db)
    try:
        result = set_scale_from_drawn_line(
            db,
            body.image_id,
            (body.x1, body.y1),
            (body.x2, body.y2),
            body.known_distance,
            body.unit,
            username=user.username,
        )
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidScaleError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"message": "Scale calibrated successfully.", **result}


@router.post("/apply_to_dataset")
async def apply_scale_to_dataset_endpoint(
    body: DatasetScaleRequest,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Apply the same physical scale to every image in a dataset.

    Useful when all images in a dataset share the same acquisition settings
    (e.g. same microscope magnification). Marks scale-dependent metrics stale
    for every affected image.
    """
    ensure_permission(user, body.dataset_id, Permission.PIXEL_SCALE_SET)
    try:
        result = apply_scale_to_dataset(
            db, body.dataset_id, body.scale_x, body.scale_y, body.unit,
            username=user.username,
        )
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidScaleError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"message": "Scale applied to dataset.", **result}

"""HTTP routes for image calibration.

Generic over calibration kinds: every handler takes the kind as a path parameter
and asks the registry what it means, so a kind added to
``app.services.calibration.registry`` is reachable here without touching this file.

Permissions are per kind, also from the registry — scale keeps the older
``pixel_scale.set`` so an existing grant of it is neither widened nor revoked,
while the newer kinds use ``calibration.set``. Both sit in the curator bundle.
"""
from logging import getLogger

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_session
from app.exceptions import (
    DatasetNotFoundError,
    ImageNotFoundError,
    InvalidCalibrationError,
    InvalidScaleError,
    UnknownCalibrationKindError,
)
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services.auth import get_current_user
from app.services.calibration import registry, service
from app.services.calibration.registry import CalibrationSource
from app.services.permissions import ensure_permission, ensure_permission_for, require

router = APIRouter(prefix="/calibration", tags=["calibration"])
logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class SetCalibrationRequest(BaseModel):
    """Kind-specific parameters, validated by the kind rather than by pydantic.

    A per-kind pydantic model would put the parameter shapes in two places and
    make adding a kind a two-file change; the registry's ``normalize`` is the
    single validator, and it is also what the dataset-wide and sampling paths use.
    """
    params: dict = Field(default_factory=dict)
    source: str = Field(
        default=CalibrationSource.MANUAL,
        description="How the calibration was obtained: manual | measured | dataset | file_metadata",
    )


class DatasetCalibrationRequest(BaseModel):
    """Apply one calibration to every image in a dataset."""
    dataset_id: int
    params: dict = Field(default_factory=dict)


class DatasetDefaultsRequest(BaseModel):
    """Set how a dataset calibrates one kind, before anything is measured."""
    defaults: dict = Field(default_factory=dict)


class SamplePatchRequest(BaseModel):
    """Average a disc of pixels to use as a calibration reference."""
    x: float = Field(..., description="Sample centre x, in image pixels.")
    y: float = Field(..., description="Sample centre y, in image pixels.")
    radius: int = Field(
        default=service.DEFAULT_SAMPLE_RADIUS, ge=1, le=service.MAX_SAMPLE_RADIUS,
        description="Radius of the averaged disc, in image pixels.",
    )
    for_kind: str | None = Field(
        default=None,
        description="Kind being calibrated. Calibration stages ordered before it are "
                    "applied first, so the sample is read in the space that kind acts on.",
    )


class SamplePatchesRequest(BaseModel):
    """Read every patch of a reference card in one request."""
    points: list[tuple[float, float]] = Field(
        ..., description="Patch centres in image pixels, in card order.",
    )
    radius: int = Field(
        default=service.DEFAULT_SAMPLE_RADIUS, ge=1, le=service.MAX_SAMPLE_RADIUS,
    )
    for_kind: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _permission_for(kind_key: str) -> Permission:
    """The permission a kind requires, or 404 for a kind that does not exist."""
    try:
        return registry.get_kind(kind_key).permission
    except UnknownCalibrationKindError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@router.get("/kinds")
async def list_calibration_kinds(user: AuthenticatedUser = Depends(get_current_user)):
    """Describe every calibration kind this server supports.

    Lets a client render (or at least name) a kind it has no purpose-built card
    for, and tells it which measurements each kind affects.
    """
    return {"kinds": service.kinds_metadata()}


# ---------------------------------------------------------------------------
# Per-image
# ---------------------------------------------------------------------------

@router.get("/image/{image_id}")
async def get_image_calibrations(
    image_id: int,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(require(Permission.IMAGE_READ, "image_id")),
):
    """Every kind's state for one image — one entry per kind, calibrated or not."""
    try:
        return service.get_calibration_state(db, image_id)
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.put("/image/{image_id}/{kind}")
async def set_image_calibration(
    image_id: int,
    kind: str,
    body: SetCalibrationRequest,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Set (or replace) one calibration on one image.

    Stores parameters only — the image file is never modified — and marks the
    measurements that depend on this kind stale so the next quantification run
    recomputes them.
    """
    ensure_permission_for(user, "image_id", image_id, _permission_for(kind), db)
    source = body.source if body.source in CalibrationSource.ALL else CalibrationSource.MANUAL
    try:
        result = service.set_calibration(
            db, image_id, kind, body.params, source=source, username=user.username,
        )
    except UnknownCalibrationKindError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (InvalidCalibrationError, InvalidScaleError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"message": "Calibration saved.", **result}


@router.delete("/image/{image_id}/{kind}")
async def clear_image_calibration(
    image_id: int,
    kind: str,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Remove one calibration, returning the image to its uncalibrated reading."""
    ensure_permission_for(user, "image_id", image_id, _permission_for(kind), db)
    try:
        result = service.clear_calibration(db, image_id, kind)
    except UnknownCalibrationKindError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"message": "Calibration cleared." if result["cleared"]
                       else "No calibration of that kind was set.", **result}


@router.post("/image/{image_id}/sample")
async def sample_reference_patch(
    image_id: int,
    body: SamplePatchRequest,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(require(Permission.IMAGE_READ, "image_id")),
):
    """Average a disc of pixels around a point, for use as a calibration reference.

    Read from the original file on disk rather than from whatever the client has
    rendered, so the numbers describe the data and not the display.
    """
    try:
        return service.sample_patch(
            db, image_id, body.x, body.y, radius=body.radius, for_kind=body.for_kind,
        )
    except UnknownCalibrationKindError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidCalibrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/image/{image_id}/sample_batch")
async def sample_reference_patches(
    image_id: int,
    body: SamplePatchesRequest,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(require(Permission.IMAGE_READ, "image_id")),
):
    """Read every patch of a reference card at once.

    One request rather than one per patch: the server decodes the image and
    applies the preceding calibration stages a single time, which for a
    twenty-patch card is the difference between one decode and twenty.
    """
    try:
        return service.sample_patches(
            db, image_id, body.points, radius=body.radius, for_kind=body.for_kind,
        )
    except UnknownCalibrationKindError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidCalibrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ---------------------------------------------------------------------------
# Dataset-wide
# ---------------------------------------------------------------------------

@router.post("/dataset/{kind}")
async def apply_calibration_to_dataset(
    kind: str,
    body: DatasetCalibrationRequest,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Apply one calibration to every image in a dataset.

    The usual case for a dataset captured in a single session: the reference is
    measured once and propagated. Propagated rows are marked ``source=dataset``,
    so a reader can tell them from calibrations measured in their own frame.
    """
    ensure_permission(user, body.dataset_id, _permission_for(kind))
    try:
        result = service.apply_to_dataset(
            db, body.dataset_id, kind, body.params, username=user.username,
        )
    except UnknownCalibrationKindError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (InvalidCalibrationError, InvalidScaleError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"message": "Calibration applied to dataset.", **result}


@router.get("/dataset/{dataset_id}/defaults")
async def get_dataset_calibration_defaults(
    dataset_id: int,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(require(Permission.DATASET_READ, "dataset_id")),
):
    """How this dataset calibrates each kind, where it has been configured."""
    return {"dataset_id": dataset_id, "defaults": service.get_dataset_defaults(db, dataset_id)}


@router.put("/dataset/{dataset_id}/defaults/{kind}")
async def set_dataset_calibration_defaults(
    dataset_id: int,
    kind: str,
    body: DatasetDefaultsRequest,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Choose the strategy (and reference card) this dataset calibrates a kind with.

    Separate from setting a calibration: the choice has to be makeable before
    anything has been measured, and changing it deliberately leaves existing
    calibrations alone rather than silently re-estimating them.
    """
    ensure_permission(user, dataset_id, _permission_for(kind))
    try:
        result = service.set_dataset_defaults(
            db, dataset_id, kind, body.defaults, username=user.username,
        )
    except UnknownCalibrationKindError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidCalibrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"message": "Calibration defaults saved.", **result}

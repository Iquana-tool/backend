"""Stacks: upload, list, inspect and delete OCT volumes and other frame stacks.

The frames of a stack are images, so everything else -- serving a frame's pixels,
annotating it, reviewing it -- goes through the image, mask and contour routes
with the frame's ``image_id``.
"""
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_session
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.exceptions import InvalidMetadataError, StackNotFoundError, UnsupportedStackFileError
from app.services.database_access import datasets as datasets_db
from app.services.database_access import image_metadata as metadata_db
from app.services.database_access import stacks as stacks_db
from app.services.permissions import require
from app.services.stack_readers import supported_extensions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stacks", tags=["stacks"])

#: Stored files never change after upload, so the browser may keep them. They are
#: per-user data, hence ``private``.
IMMUTABLE_FILE_HEADERS = {"Cache-Control": "private, max-age=31536000, immutable"}


class StackMetadataUpdate(BaseModel):
    entries: dict[str, str] = Field(default_factory=dict)
    replace: bool = False
    remove_keys: list[str] = Field(default_factory=list)


@router.get("/formats")
async def get_supported_formats():
    """File extensions the stack upload accepts."""
    return {"extensions": supported_extensions()}


@router.post("/upload")
async def upload_stacks(
        dataset_id: int,
        files: list[UploadFile] = File(...),
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_UPLOAD)),
):
    """Upload stack files (e.g. Heidelberg ``.vol`` / ``.e2e``). One file may hold several stacks.

    Files are processed one by one; a file that cannot be read is reported in
    ``failed`` and does not stop the others.
    """
    dataset = await datasets_db.get_dataset(dataset_id, db=db)
    stack_ids: list[int] = []
    failed: list[dict] = []
    for file in files:
        try:
            stack_ids += await stacks_db.process_and_save_stack_file(
                file, dataset_id, dataset.folder_path, db=db, username=user.username)
        except UnsupportedStackFileError as exc:
            failed.append({"file_name": file.filename, "reason": str(exc)})
    if not stack_ids and failed:
        raise HTTPException(status_code=400, detail=failed)
    return {
        "success": True,
        "message": f"Uploaded {len(stack_ids)} stack(s).",
        "stack_ids": stack_ids,
        "failed": failed,
    }


@router.get("/dataset/{dataset_id}")
async def list_stacks(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ)),
):
    """Every stack of a dataset with its own metadata."""
    return {"stacks": stacks_db.list_stacks_of_dataset(db, dataset_id)}


@router.get("/{stack_id}")
async def get_stack(
        stack_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ, "stack_id")),
):
    """A stack with its frames, their positions on the overview and their workflow status."""
    return stacks_db.get_stack_details(db, stack_id)


@router.get("/{stack_id}/overview")
async def get_stack_overview(
        stack_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_READ, "stack_id")),
):
    """The stack's overview image (an OCT volume's IR-SLO) as PNG."""
    stack = stacks_db.get_stack(db, stack_id)
    if stack.overview_file_path is None:
        raise HTTPException(status_code=404, detail=f"Stack {stack_id} has no overview image.")
    return FileResponse(stack.overview_file_path, media_type="image/png", headers=IMMUTABLE_FILE_HEADERS)


@router.get("/{stack_id}/thumbnail")
async def get_stack_thumbnail(
        stack_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_READ, "stack_id")),
):
    stack = stacks_db.get_stack(db, stack_id)
    return FileResponse(stack.thumbnail_file_path, media_type="image/png", headers=IMMUTABLE_FILE_HEADERS)


@router.put("/{stack_id}/metadata")
async def update_stack_metadata(
        stack_id: int,
        payload: StackMetadataUpdate,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_METADATA_WRITE, "stack_id")),
):
    """Edit the stack's own metadata, which every frame inherits.

    An empty value removes the key (as for images).
    """
    try:
        result = metadata_db.set_metadata_for_stacks(
            db, [stack_id], payload.entries, username=user.username,
            replace=payload.replace, remove_keys=payload.remove_keys)
    except StackNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InvalidMetadataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, **result, "metadata": metadata_db.get_metadata_for_stack(db, stack_id)}


@router.delete("/{stack_id}")
async def delete_stack(
        stack_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_DELETE, "stack_id")),
):
    """Delete a stack with all its frames and their annotations."""
    stacks_db.delete_stack(db, stack_id)
    return {"success": True, "message": f"Deleted stack {stack_id}."}

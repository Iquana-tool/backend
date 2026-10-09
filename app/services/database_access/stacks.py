"""Store, list and delete stacks (see ``app.database.stacks``).

Upload converts: the file is read by a reader from ``app.services.stack_readers``,
every frame is written as a PNG next to the dataset's images, and the uploaded
file itself is discarded. One file can hold several stacks.

A stack is stored in one transaction -- the stack row, its frames, their masks
and all metadata -- and the files written for it are removed again if anything
fails, so a half-imported volume never shows up in the gallery.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from logging import getLogger
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from app.database.image_metadata import ImageMetadata
from app.database.images import Frames
from app.database.masks import Masks
from app.database.stacks import Stacks
from app.exceptions import StackNotFoundError
from app.services import image_status
from app.services.database_access import image_metadata as metadata_db
from app.services.database_access.masks import create_new_mask
from app.services.metadata_types import coerce
from app.services.stack_readers import StackData, read_stack_file
from config import THUMBNAILS_DIR

logger = getLogger(__name__)

#: Longest side of a gallery thumbnail, as for images.
THUMBNAIL_SIZE = (500, 500)


async def process_and_save_stack_file(
        file: UploadFile,
        dataset_id: int,
        dataset_folder: str,
        db: Session,
        username: str | None = None,
) -> list[int]:
    """Read an uploaded stack file and store every stack in it. Returns the new stack ids.

    :raises UnsupportedStackFileError: if the format is unknown or the file holds
        no importable volume.
    """
    await file.seek(0)
    suffix = Path(file.filename or "").suffix
    # The readers need a path (eyepy seeks around the file). Windows cannot reopen
    # a NamedTemporaryFile while it is open, so write, close, read, then remove.
    handle, temp_name = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(handle, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
        stacks = await run_in_threadpool(read_stack_file, Path(temp_name), file.filename)
    finally:
        os.remove(temp_name)

    stack_ids = []
    for data in stacks:
        stack_ids.append(await save_stack(data, dataset_id, dataset_folder, db,
                                          source_file_name=file.filename, username=username))
    return stack_ids


async def save_stack(
        data: StackData,
        dataset_id: int,
        dataset_folder: str,
        db: Session,
        source_file_name: str | None = None,
        username: str | None = None,
) -> int:
    """Store one stack read by a reader: files, rows and metadata, all or nothing."""
    stack = Stacks(
        dataset_id=dataset_id,
        name=data.name,
        kind=data.kind,
        axis=data.axis,
        frame_count=len(data.frames),
        frame_spacing=data.frame_spacing,
        frame_spacing_unit=data.frame_spacing_unit,
        overview_scale_x=data.overview_scale_x,
        overview_scale_y=data.overview_scale_y,
        overview_unit=data.overview_unit,
        thumbnail_file_path="",  # set once the id names the folder
        source_format=data.source_format,
        source_file_name=source_file_name or data.name,
        created_by=username,
    )
    db.add(stack)
    db.flush()

    stack_folder = Path(dataset_folder) / "stacks" / str(stack.id)
    thumbnail_folder = Path(THUMBNAILS_DIR) / "stacks" / str(stack.id)
    try:
        frame_folder = stack_folder / "frames"
        os.makedirs(frame_folder, exist_ok=True)
        os.makedirs(thumbnail_folder, exist_ok=True)

        if data.overview is not None:
            overview_path = stack_folder / "overview.png"
            Image.fromarray(data.overview).save(overview_path)
            stack.overview_file_path = str(overview_path)
            stack.overview_height, stack.overview_width = data.overview.shape[:2]
        preview = data.overview if data.overview is not None else data.frames[len(data.frames) // 2].pixels
        stack_thumbnail = thumbnail_folder / "stack.png"
        _save_thumbnail(preview, stack_thumbnail)
        stack.thumbnail_file_path = str(stack_thumbnail)

        frames: list[Frames] = []
        for index, frame_data in enumerate(data.frames):
            file_name = f"frame_{index:04d}.png"
            frame_path = frame_folder / file_name
            frame_thumbnail = thumbnail_folder / file_name
            image = Image.fromarray(frame_data.pixels)
            image.save(frame_path)
            _save_thumbnail(frame_data.pixels, frame_thumbnail)
            frame = Frames(
                dataset_id=dataset_id,
                stack_id=stack.id,
                frame_index=index,
                frame_position=frame_data.position,
                overview_geometry=frame_data.overview_geometry,
                file_name=f"{data.name}_{file_name}",
                file_path=str(frame_path),
                thumbnail_file_path=str(frame_thumbnail),
                width=image.width,
                height=image.height,
                color_mode=image.mode,
                scale_x=data.scale_x,
                scale_y=data.scale_y,
                unit=data.unit,
            )
            db.add(frame)
            frames.append(frame)
        db.flush()

        for frame in frames:
            await create_new_mask(frame.id, dataset_folder, db)

        _add_metadata_rows(db, dataset_id, data, stack, frames, username)
        db.commit()
    except Exception:
        db.rollback()
        shutil.rmtree(stack_folder, ignore_errors=True)
        shutil.rmtree(thumbnail_folder, ignore_errors=True)
        raise

    logger.info("Stored stack %s ('%s', %s frames) in dataset %s.",
                stack.id, data.name, len(frames), dataset_id)
    return stack.id


def _save_thumbnail(pixels: np.ndarray, path: Path) -> None:
    thumbnail = Image.fromarray(pixels)
    thumbnail.thumbnail(THUMBNAIL_SIZE)
    thumbnail.save(path)


def _add_metadata_rows(db: Session, dataset_id: int, data: StackData, stack: Stacks,
                       frames: list[Frames], username: str | None) -> None:
    """Stack metadata once on the stack, frame metadata on each frame.

    Goes through the same key declaration and coercion as every other metadata
    write, but adds the rows itself so the whole stack stays one transaction
    (``set_metadata_for_*`` commit per call).
    """
    descriptors = {}

    def typed(key: str, value: str) -> tuple[str, str, float | None]:
        key = metadata_db.normalize_key(key)
        if key not in descriptors:
            descriptors[key] = metadata_db.ensure_key(
                db, dataset_id, key, value_type=data.key_types.get(key), username=username)
        descriptor = descriptors[key]
        canonical, numeric = coerce(metadata_db.normalize_value(value), descriptor.value_type,
                                    list(descriptor.options or []))
        return key, canonical, numeric

    for key, value in data.metadata.items():
        if value:
            key, canonical, numeric = typed(key, value)
            db.add(ImageMetadata(stack_id=stack.id, key=key, value=canonical,
                                 value_num=numeric, created_by=username))
    for frame, frame_data in zip(frames, data.frames):
        for key, value in frame_data.metadata.items():
            if value:
                key, canonical, numeric = typed(key, value)
                db.add(ImageMetadata(image_id=frame.id, key=key, value=canonical,
                                     value_num=numeric, created_by=username))


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_stack(db: Session, stack_id: int) -> Stacks:
    stack = db.query(Stacks).filter_by(id=stack_id).first()
    if stack is None:
        raise StackNotFoundError(f"Stack {stack_id} was not found.")
    return stack


def _summary(stack: Stacks, metadata: dict[str, str]) -> dict:
    return {
        "stack_id": stack.id,
        "dataset_id": stack.dataset_id,
        "name": stack.name,
        "description": stack.description,
        "kind": stack.kind,
        "axis": stack.axis,
        "frame_count": stack.frame_count,
        "frame_spacing": stack.frame_spacing,
        "frame_spacing_unit": stack.frame_spacing_unit,
        "has_overview": stack.overview_file_path is not None,
        "overview_width": stack.overview_width,
        "overview_height": stack.overview_height,
        "source_format": stack.source_format,
        "created_by": stack.created_by,
        "created_at": stack.created_at.isoformat() if stack.created_at else None,
        "metadata": metadata,
    }


def list_stacks_of_dataset(db: Session, dataset_id: int) -> list[dict]:
    """Every stack of a dataset with its own metadata, for the gallery."""
    stacks = db.query(Stacks).filter_by(dataset_id=dataset_id).order_by(Stacks.id).all()
    metadata = metadata_db.get_metadata_for_stacks(db, [stack.id for stack in stacks])
    return [_summary(stack, metadata[stack.id]) for stack in stacks]


def get_stack_details(db: Session, stack_id: int) -> dict:
    """A stack, its own metadata, and every frame with its workflow status.

    Frame metadata is the frame's *own* rows only; what it inherits is the
    stack's ``metadata``, sent once rather than repeated per frame.
    """
    stack = get_stack(db, stack_id)
    frames = (
        db.query(Frames)
        .filter(Frames.stack_id == stack_id)
        .order_by(Frames.frame_index)
        .all()
    )
    frame_ids = [frame.id for frame in frames]
    masks_by_image = {
        mask.image_id: mask
        for mask in db.query(Masks).filter(Masks.image_id.in_(frame_ids)).all()
    } if frame_ids else {}
    statuses = image_status.status_for_images(db, frames, masks_by_image=masks_by_image)
    own_metadata: dict[int, dict[str, str]] = {frame_id: {} for frame_id in frame_ids}
    if frame_ids:
        for row in db.query(ImageMetadata).filter(ImageMetadata.image_id.in_(frame_ids)).all():
            own_metadata[row.image_id][row.key] = row.value

    details = _summary(stack, metadata_db.get_metadata_for_stack(db, stack_id))
    details["frames"] = [
        {
            "image_id": frame.id,
            "frame_index": frame.frame_index,
            "frame_position": frame.frame_position,
            "overview_geometry": frame.overview_geometry,
            "width": frame.width,
            "height": frame.height,
            "scale_x": frame.scale_x,
            "scale_y": frame.scale_y,
            "unit": frame.unit,
            "mask_id": statuses[frame.id]["mask_id"],
            "status": statuses[frame.id]["status"],
            "phases": statuses[frame.id]["phases"],
            "metadata": own_metadata[frame.id],
        }
        for frame in frames
    ]
    return details


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def delete_stack(db: Session, stack_id: int) -> None:
    """Delete a stack, its frames (with their masks and contours) and its files."""
    stack = get_stack(db, stack_id)
    frame_paths = [
        (row.file_path, row.thumbnail_file_path)
        for row in db.query(Frames.file_path, Frames.thumbnail_file_path).filter(Frames.stack_id == stack_id)
    ]
    # frames live in <stack folder>/frames/, the overview in <stack folder>/.
    stack_folder = (Path(frame_paths[0][0]).parent.parent if frame_paths
                    else Path(stack.overview_file_path).parent if stack.overview_file_path else None)
    thumbnail_folder = Path(stack.thumbnail_file_path).parent if stack.thumbnail_file_path else None

    db.delete(stack)
    db.commit()

    for paths in frame_paths:
        for path in paths:
            if path and os.path.exists(path):
                os.remove(path)
    for folder in (stack_folder, thumbnail_folder):
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)

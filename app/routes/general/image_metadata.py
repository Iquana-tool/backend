"""HTTP routes for per-image metadata.

Laid out like the calibration router — ``/image/{id}`` for one image,
``/dataset/{id}`` for the dataset-wide view — because the two features answer the
same shape of question about an image and the clients pair them in the same UI.

Reading needs ``image.read``; writing needs ``image.metadata_write``, which sits
in the curator bundle (see ``app.schemas.permissions``).
"""
from logging import getLogger

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette import status
from starlette.responses import StreamingResponse

from app.database import get_session
from app.database.images import Images
from app.exceptions import ImageNotFoundError, InvalidMetadataError
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services import metadata_csv
from app.services.auth import get_current_user
from app.services.database_access import image_metadata as metadata_db
from app.services.database_access import datasets as datasets_db
from app.services.metadata_types import (
    GROUPABLE_TYPES,
    ORDERED_TYPES,
    MetadataValueType,
)
from app.services.permissions import ensure_permission_on_datasets, require

router = APIRouter(prefix="/metadata", tags=["image metadata"])
logger = getLogger(__name__)


class ImageMetadataRequest(BaseModel):
    """Metadata edits for a single image."""
    entries: dict[str, str] = Field(
        default_factory=dict,
        description="Key/value pairs to write. An empty value removes the key.",
    )
    replace: bool = Field(
        default=False,
        description="Treat `entries` as the image's complete metadata, deleting "
                    "any key it does not mention.",
    )


class BulkMetadataRequest(BaseModel):
    """The same metadata edits applied to a set of images.

    This is the grouping action: select the images of one site / treatment /
    collection date and give them all the same key. `remove_keys` is its inverse,
    for pulling images back out of a subgroup.
    """
    image_ids: list[int] = Field(..., min_length=1)
    entries: dict[str, str] = Field(default_factory=dict)
    remove_keys: list[str] = Field(default_factory=list)
    replace: bool = Field(
        default=False,
        description="Delete every key not named in `entries` from each image. "
                    "Off by default: a bulk tag should not wipe per-image keys.",
    )


class MetadataKeyRequest(BaseModel):
    """Declare or redeclare what a metadata key means."""
    value_type: str | None = Field(
        default=None,
        description="One of text | categorical | number | date | boolean.",
    )
    unit: str | None = Field(default=None, description="Display unit for a numeric key.")
    options: list[str] | None = Field(
        default=None,
        description="Allowed values for a categorical key. Empty leaves it open.",
    )
    description: str | None = None


class RenameKeyRequest(BaseModel):
    """Rename a key, optionally folding it into one that already exists."""
    new_key: str
    merge: bool = Field(
        default=False,
        description="Allow merging into an existing key. Without it a collision "
                    "is refused rather than silently combining two vocabularies.",
    )


def _check_image_batch(image_ids: list[int], user: AuthenticatedUser,
                       permission: Permission, db: Session) -> None:
    """Check `permission` on every dataset the requested images belong to.

    A bulk edit takes an arbitrary id list, which can span datasets the caller has
    no access to — one check on one id would not cover it.
    """
    dataset_ids = [
        row.dataset_id for row in
        db.query(Images.dataset_id).filter(Images.id.in_(image_ids)).distinct()
    ]
    if not dataset_ids:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="None of the given images exist.")
    ensure_permission_on_datasets(user, dataset_ids, permission)


# ---------------------------------------------------------------------------
# Per-image
# ---------------------------------------------------------------------------

@router.get("/image/{image_id}")
async def get_image_metadata(
        image_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_READ, "image_id")),
):
    """Every metadata pair of one image, as a flat ``{key: value}`` object."""
    return {
        "success": True,
        "image_id": image_id,
        "metadata": metadata_db.get_metadata(db, image_id),
    }


@router.put("/image/{image_id}")
async def set_image_metadata(
        image_id: int,
        body: ImageMetadataRequest,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(
            require(Permission.IMAGE_METADATA_WRITE, "image_id")),
):
    """Write metadata onto one image.

    Upserts by default and only touches the keys it is given; pass
    ``replace=true`` to make the payload authoritative for the whole image.
    Either way, a key sent with an empty value is removed rather than blanked.
    """
    try:
        result = metadata_db.set_metadata_for_images(
            db, [image_id], body.entries,
            username=user.username, replace=body.replace,
        )
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except InvalidMetadataError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return {
        "success": True,
        "message": f"Wrote {result['written']} and removed {result['removed']} "
                   f"metadata entries.",
        "metadata": metadata_db.get_metadata(db, image_id),
    }


@router.delete("/image/{image_id}/{key}")
async def delete_image_metadata_key(
        image_id: int,
        key: str,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(
            require(Permission.IMAGE_METADATA_WRITE, "image_id")),
):
    """Remove one key from one image."""
    try:
        deleted = metadata_db.delete_key(db, image_id, key)
    except InvalidMetadataError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return {
        "success": True,
        "message": f"Removed '{key}'." if deleted else f"Image had no '{key}'.",
        "deleted": deleted,
        "metadata": metadata_db.get_metadata(db, image_id),
    }


# ---------------------------------------------------------------------------
# Bulk
# ---------------------------------------------------------------------------

@router.post("/images")
async def set_metadata_for_images(
        body: BulkMetadataRequest,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(get_current_user),
):
    """Apply one set of edits to many images at once — the grouping action.

    There is no single id for the `require` dependency to key off, so the check is
    :func:`_check_image_batch` over every dataset the id list touches: a list that
    reaches into a dataset the caller cannot curate is rejected outright rather
    than partly applied.
    """
    _check_image_batch(body.image_ids, user, Permission.IMAGE_METADATA_WRITE, db)
    try:
        result = metadata_db.set_metadata_for_images(
            db, body.image_ids, body.entries,
            username=user.username, replace=body.replace,
            remove_keys=body.remove_keys,
        )
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except InvalidMetadataError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return {
        "success": True,
        "message": f"Updated metadata on {len(body.image_ids)} image(s): "
                   f"{result['written']} written, {result['removed']} removed.",
        "written": result["written"],
        "removed": result["removed"],
        "metadata": {
            str(image_id): entries for image_id, entries
            in metadata_db.get_metadata_for_images(db, body.image_ids).items()
        },
    }


# ---------------------------------------------------------------------------
# Dataset-wide
# ---------------------------------------------------------------------------

@router.get("/dataset/{dataset_id}")
async def get_dataset_metadata(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_READ)),
):
    """The dataset's metadata vocabulary: which keys it uses and their values.

    This is what the grouping UI is drawn from — the filter chips, and the key /
    value suggestions in the editor that keep a second curator from inventing
    ``Site`` next to ``site``. The per-image pairs are not repeated here; they
    ride along on ``GET /datasets/{id}/images`` with the rest of each image's row.

    ``untagged_count`` is how many images carry no metadata at all, which is the
    honest answer to "is this dataset actually grouped yet?".
    """
    per_image = metadata_db.get_metadata_for_dataset(db, dataset_id)
    total_images = db.query(Images.id).filter(Images.dataset_id == dataset_id).count()
    return {
        "success": True,
        "facets": metadata_db.get_dataset_facets(db, dataset_id),
        "total_images": total_images,
        "untagged_count": total_images - len(per_image),
    }


# ---------------------------------------------------------------------------
# Key declarations
# ---------------------------------------------------------------------------

@router.get("/types")
async def list_metadata_types(user: AuthenticatedUser = Depends(get_current_user)):
    """The value types a metadata key can have, and what each one supports.

    Dataset-independent, so the client can build its type picker and know which
    types offer a range filter (``ordered``) and which can be a grouping axis
    (``groupable``) without hardcoding the list.
    """
    return {
        "success": True,
        "types": [
            {
                "value_type": value_type.value,
                "groupable": value_type in GROUPABLE_TYPES,
                "ordered": value_type in ORDERED_TYPES,
            }
            for value_type in MetadataValueType
        ],
    }


@router.get("/dataset/{dataset_id}/keys")
async def list_dataset_metadata_keys(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_READ)),
):
    """Every declared key of the dataset with its type, unit and vocabulary."""
    return {
        "success": True,
        "keys": [row.to_dict() for row in metadata_db.list_keys(db, dataset_id)],
    }


@router.put("/dataset/{dataset_id}/keys/{key}")
async def update_dataset_metadata_key(
        dataset_id: int,
        key: str,
        body: MetadataKeyRequest,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_METADATA_WRITE)),
):
    """Change a key's type, unit, vocabulary or description.

    Retyping re-validates every value already stored under the key and is refused
    outright — naming the offending values — if any of them do not fit. Declaring
    ``depth`` numeric while one image reads "shallow" would otherwise either lose
    that value or leave one the type says cannot exist.
    """
    try:
        if metadata_db.get_key(db, dataset_id, key) is None:
            metadata_db.ensure_key(db, dataset_id, key,
                                   value_type=body.value_type, username=user.username)
            db.commit()
        descriptor = metadata_db.update_key(
            db, dataset_id, key,
            value_type=body.value_type, unit=body.unit,
            options=body.options, description=body.description,
        )
    except InvalidMetadataError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return {"success": True, "message": "Key updated.", "key": descriptor.to_dict()}


@router.post("/dataset/{dataset_id}/keys/{key}/rename")
async def rename_dataset_metadata_key(
        dataset_id: int,
        key: str,
        body: RenameKeyRequest,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_METADATA_WRITE)),
):
    """Rename a key across the dataset, or merge it into an existing one.

    The repair for the split every free-form key system grows: someone typed
    ``Site`` where everyone else typed ``site``, and the dataset reports two
    subgroups that are one.
    """
    try:
        result = metadata_db.rename_key(db, dataset_id, key, body.new_key,
                                        merge=body.merge)
    except InvalidMetadataError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return {
        "success": True,
        "message": f"Renamed '{key}' on {result['renamed']} image(s)"
                   + (f", merging {result['merged']}." if result["merged"] else "."),
        **result,
    }


@router.delete("/dataset/{dataset_id}/keys/{key}")
async def delete_dataset_metadata_key(
        dataset_id: int,
        key: str,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_METADATA_WRITE)),
):
    """Drop a key and every value of it across the dataset."""
    removed = metadata_db.delete_key_from_dataset(db, dataset_id, key)
    return {
        "success": True,
        "message": f"Removed '{key}' from {removed} image(s).",
        "removed": removed,
    }


# ---------------------------------------------------------------------------
# CSV round trip
# ---------------------------------------------------------------------------

@router.get("/dataset/{dataset_id}/csv")
async def download_dataset_metadata_csv(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_READ)),
):
    """The dataset's metadata as a CSV — and, when empty, the template to fill in.

    One row per image including untagged ones, so the downloaded file is what you
    edit and upload back. Starting from it is what makes the filenames match.
    """
    dataset = await datasets_db.get_dataset(dataset_id, db=db)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
    content = metadata_csv.export_csv(db, dataset_id)
    file_name = f"{dataset.name.replace(' ', '_')}_metadata.csv"
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
    )


@router.post("/dataset/{dataset_id}/import")
async def import_dataset_metadata_csv(
        dataset_id: int,
        file: UploadFile = File(...),
        dry_run: bool = True,
        replace: bool = False,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_METADATA_WRITE)),
):
    """Apply a metadata CSV to the dataset, or preview what it would do.

    Defaults to ``dry_run=true``: the response is the preview — rows matched,
    filenames not found, images the file omits, and each column's key and type
    (inferred from the file's own values when the key is new). The same call with
    ``dry_run=false`` applies exactly that, so what is approved is what happens.

    Matching is scoped to this dataset, so a file naming an image elsewhere counts
    as unmatched rather than reaching into another dataset.
    """
    content = await file.read()
    try:
        return {
            "success": True,
            **metadata_csv.import_csv(
                db, dataset_id, content,
                username=user.username, dry_run=dry_run, replace=replace,
            ),
        }
    except InvalidMetadataError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

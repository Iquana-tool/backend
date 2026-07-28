import json
import logging

from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.images import Images
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import Permission
from app.services.auth import get_current_user
from app.services.database_access import datasets as datasets_db
from app.services.database_access import images as images_db
from app.services.embedding_lifecycle import enqueue_embed_image
from app.services.permissions import ensure_permission_on_datasets, require

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/images", tags=["images"])


def _check_image_batch(image_ids: list[int], user: AuthenticatedUser,
                       permission: Permission, db: Session) -> None:
    """Check `permission` on every dataset the requested images belong to.

    Batch endpoints take an arbitrary list of ids, so one `require()` on a single
    id is not enough — the list can span datasets the caller has no access to.
    """
    dataset_ids = [
        row.dataset_id for row in
        db.query(Images.dataset_id).filter(Images.id.in_(image_ids)).distinct()
    ]
    ensure_permission_on_datasets(user, dataset_ids, permission)


@router.post("/upload")
async def upload_image(
        dataset_id: int,
        file: UploadFile = File(...),
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_UPLOAD)),
):
    dataset = await datasets_db.get_dataset(dataset_id, db=db)
    image_id = await images_db.process_and_save_image(file, dataset_id, dataset.folder_path, db=db)
    # Opt-in background embedding for cross-image retrieval (no-op unless enabled).
    enqueue_embed_image(image_id)
    return {
        "success": True,
        "message": f"Uploaded image {image_id}.",
        "image_id": image_id
    }


@router.post("/upload_multi")
async def upload_images(
        dataset_id: int,
        files: list[UploadFile] = File(...),
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_UPLOAD)),
):
    dataset = await datasets_db.get_dataset(dataset_id, db=db)
    image_ids = []

    for file in files:
        # Now we only query the dataset folder ONCE at the top
        image_id = await images_db.process_and_save_image(file, dataset_id, dataset.folder_path, db=db)
        image_ids.append(image_id)

    # Opt-in background embedding for cross-image retrieval (no-op unless enabled).
    for image_id in image_ids:
        enqueue_embed_image(image_id)

    return {
        "success": True,
        "message": f"Uploaded {len(image_ids)} images.",
        "image_ids": image_ids
    }


@router.post("/backfill_dimensions")
async def backfill_image_dimensions(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.DATASET_UPDATE)),
):
    """Repair image dimensions recorded by the old thumbnail-mutation ingest bug.

    Rows uploaded before the fix stored the 200px preview's dimensions instead of
    the native ones, which shrank the precomputed contour paths (annotations
    huddled in the image's top-left corner), the COCO export and the pixel-space
    geometry metrics by the same factor. This re-reads every image file of the
    dataset, fixes mismatched dimensions and recomputes the geometry metrics of
    the affected contours. Safe to re-run: images whose stored dimensions already
    match are untouched.
    """
    result = await images_db.backfill_image_dimensions(db, dataset_id=dataset_id)
    return {
        "success": True,
        "message": f"Corrected {len(result['corrected'])} images, "
                   f"recomputed metrics for {result['recomputed_contours']} contours"
                   + (f"; {len(result['missing'])} files missing on disk."
                      if result["missing"] else "."),
        **result,
    }


@router.delete("/{image_id}")
async def delete_image(
        image_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_DELETE, "image_id"))
):
    """Delete an image and its associated masks.

    Args:
        image_id: ID of the image to delete.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        A dictionary indicating success and a message.
    """
    await images_db.delete_image(image_id, db=db)
    return {"success": True,
            "message": f"Deleted image {image_id}."}


@router.get("/{image_id}/b64")
async def get_base64_image(
        image_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_READ, "image_id"))
):
    """Get images via ids.

    Args:
        image_id (int): Image ID to retrieve.
        db (Session): Database session dependency.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        A dict mapping from image ID to base64 encoded image.
    """
    return {
        "success": True,
        "message": f"Successfully retrieved image {image_id}.",
        image_id: await images_db.get_image_data(image_id, as_thumbnail=False, as_base64=True, db=db)
    }


@router.get("/{image_id}/thumbnail")
async def get_base64_thumbnail(
        image_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_READ, "image_id"))
):
    """Get images via ids.

    Args:
        image_id (int): Image ID to retrieve.
        db (Session): Database session dependency.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        A dict mapping from image ID to base64 encoded image.
    """
    return {
        "success": True,
        "message": f"Successfully retrieved image {image_id}.",
        image_id: await images_db.get_image_data(image_id, as_thumbnail=True, as_base64=True, db=db)
    }


@router.get("/ids/b64")
async def get_base64_images(
        image_ids: str,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(get_current_user)
):
    """Get images via a list of image IDs. This gets the images in batches to avoid sending too many requests at once.

    Args:
        image_ids (str): JSON string containing a list of image IDs to retrieve.
        db (Session): Database session dependency.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        A dictionary mapping from image ID to base64 encoded image.
    """
    image_ids = json.loads(image_ids)
    _check_image_batch(image_ids, user, Permission.IMAGE_READ, db)
    return {
        "success": True,
        "message": f"Successfully retrieved {len(image_ids)} images.",
        "images": await images_db.get_images_data(image_ids, as_thumbnail=False, as_base64=True, db=db)
    }


@router.get("/ids/thumbnails")
async def get_base64_thumbnails(
        image_ids: str,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(get_current_user)
):
    """Get images via a list of image IDs. This gets the images in batches to avoid sending too many requests at once.

    Args:
        image_ids (str): JSON string containing a list of image IDs to retrieve.
        db (Session): Database session dependency.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        A dictionary mapping from image ID to base64 encoded image.
    """
    image_ids = json.loads(image_ids)
    _check_image_batch(image_ids, user, Permission.IMAGE_READ, db)
    return {
        "success": True,
        "message": f"Successfully retrieved {len(image_ids)} images.",
        "images": await images_db.get_images_data(image_ids, as_thumbnail=True, as_base64=True, db=db)
    }


@router.get("/{image_id}/masks")
async def get_mask_for_image(
        image_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ, "image_id"))
):
    """ Get the mask image for a given image. """
    return {
        "success": True,
        "masks": await images_db.get_masks_of_image(image_id, db=db)
    }


@router.post("/{image_id}/masks/upload/semantic_mask")
async def post_semantic_mask_to_image(
        image_id: int,
        mask: UploadFile = File(...),
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_CREATE, "image_id"))
):
    """
    Upload a mask to a mask id. Compute the contours for each label in the mask, build the hierarchy and add
    them to the database.

    Args:
        image_id (int): The ID of the image.
        mask (UploadFile): The mask file.
        db (Session): The database session.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and result.
    """
    raise NotImplementedError

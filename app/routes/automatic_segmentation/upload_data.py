import os

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
import httpx
from paths import AUTOMATIC_SEGMENTATION_BACKEND_URL as BASE_URL
from . import router, logger
from app.database import get_session
from sqlalchemy.orm import Session

from app.database.images import Images
from app.database.mask_generation import Masks


@router.post("/upload_file_to_dataset")
async def proxy_upload_file(
    dataset_id: int = Form(...),
    is_image: bool = Form(...),
    file: UploadFile = File(...),
    filename: str = Form(None)
):
    """
    Proxies a single image/mask file upload to the segmentation training backend.
    """
    url = f"{BASE_URL}/data/upload_file_to_dataset"
    # httpx needs 'files' and 'data' for multipart forwards
    data = {
        "dataset_id": str(dataset_id),
        "is_image": str(int(is_image)),  # send as 0/1 for bool
    }
    if filename:
        data["filename"] = filename

    files = {"file": (file.filename, await file.read(), file.content_type)}

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, data=data, files=files)
        resp.raise_for_status()
        return resp.json()


@router.post("/upload_dataset")
async def proxy_upload_dataset(
    dataset_id: int,
    db: Session = Depends(get_session)
):
    """
    Finds all finished mask/image pairs for this dataset, loads them as files, and proxies to segmentation backend.
    """
    image_tuples = (
        db.query(Images.file_path)
        .join(Masks, Images.id == Masks.image_id)
        .filter(
            Masks.finished == True,
            Images.dataset_id == dataset_id
        )
        .all()
    )
    image_paths = [row[0] for row in image_tuples]
    mask_paths = []
    paired_image_paths = []

    # Only include pairs where both files exist on disk
    for img_path in image_paths:
        mask_path = get_mask_path_from_image_path(img_path)
        if not (os.path.exists(img_path) and os.path.exists(mask_path)):
            continue
        # Optionally: skip if not both files, or log
        paired_image_paths.append(img_path)
        mask_paths.append(mask_path)
    if not paired_image_paths or not mask_paths:
        raise HTTPException(status_code=400, detail="No valid image/mask pairs found.")

    url = f"{BASE_URL}/data/upload_dataset"
    data = {"dataset_id": str(dataset_id)}
    files = []

    # Open image files
    for img_path in paired_image_paths:
        filename = os.path.basename(img_path)
        files.append(
            ("images", (filename, open(img_path, "rb"), "application/octet-stream"))
        )

    # Open mask files
    for msk_path in mask_paths:
        filename = os.path.basename(msk_path)
        files.append(
            ("masks", (filename, open(msk_path, "rb"), "application/octet-stream"))
        )

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(url, data=data, files=files)
            resp.raise_for_status()
            return resp.json()
    finally:
        # Close all files!
        for tup in files:
            _, (_, f, _) = tup
            if hasattr(f, "close"):
                f.close()


def get_mask_path_from_image_path(path: str):
    parts = path.split(os.path.sep)
    parts[-2] = "masks"  # Replace the parent directory
    return os.path.sep.join(parts)
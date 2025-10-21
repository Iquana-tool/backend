import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import httpx
from paths import AUTOMATIC_SEGMENTATION_BACKEND_URL as BASE_URL
from . import logger
from app.database import get_session
from sqlalchemy.orm import Session
from logging import getLogger
from app.database.images import Images
from app.database.masks import Masks
from ...services.util import get_mask_path_from_image_path

logger = getLogger(__name__)
router = APIRouter(prefix="/semantic_segmentation", tags=["semantic_segmentation"])



@router.post("/upload_file_to_dataset")
async def proxy_upload_file(
    dataset_id: int,
    is_image: bool,
    filename: str = None,
    file: UploadFile = File(...),
):
    """
    Proxies a single image/mask file upload to the prompted_segmentation training backend.
    """
    logger.debug(f"Uploading file with name {filename} to dataset {dataset_id} as {'image' if is_image else 'mask'}")
    url = f"{BASE_URL}/data/upload_file_to_dataset"
    # httpx needs 'files' and 'data' for multipart forwards
    data = {
        "dataset_id": str(dataset_id),
        "is_image": str(is_image),  # send as 0/1 for bool
    }
    if filename:
        data["filename"] = filename

    files = {"file": (file.filename if not filename else filename, await file.read(), file.content_type)}

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, data=data, files=files)
        resp.raise_for_status()
        return resp.json()


@router.post("/upload_dataset/{dataset_id}")
async def proxy_upload_dataset(
    dataset_id: int,
    db: Session = Depends(get_session)
):
    """
    Finds all finished mask/image pairs for this dataset, and uploads them one-by-one to the prompted_segmentation backend.
    """
    image_tuples = (
        db.query(Images.file_path)
        .join(Masks, Images.id == Masks.image_id)
        .filter(Masks.finished == True, Images.dataset_id == dataset_id)
        .all()
    )
    image_paths = [row[0] for row in image_tuples]
    mask_paths = []
    paired_image_paths = []
    for img_path in image_paths:
        mask_path = get_mask_path_from_image_path(img_path)
        if not (os.path.exists(img_path) and os.path.exists(mask_path)):
            continue
        paired_image_paths.append(img_path)
        mask_paths.append(mask_path)
    if not paired_image_paths or not mask_paths:
        raise HTTPException(status_code=400, detail="No valid image/mask pairs found.")

    url = f"{BASE_URL}/data/upload_file_to_dataset"
    results = []
    async with httpx.AsyncClient(timeout=60) as client:
        for img_path in paired_image_paths:
            # --- Image upload ---
            with open(img_path, "rb") as img_file:
                files = {"file": (os.path.basename(img_path), img_file, f"image/{os.path.splitext(img_path)[1]}")}
                data = {
                    "dataset_id": str(dataset_id),
                    "is_image": "1",  # backend expects int-bool
                    "filename": os.path.splitext(os.path.basename(img_path))[0]
                }
                resp = await client.post(url, data=data, files=files)
                try:
                    resp.raise_for_status()
                except Exception as e:
                    results.append({"file": img_path, "type": "image", "status": "failed", "error": str(e)})
                    continue
                results.append({"file": img_path, "type": "image", "status": "ok"})

        for mask_path in mask_paths:
            # --- Mask upload ---
            with open(mask_path, "rb") as mask_file:
                files = {"file": (os.path.basename(mask_path), mask_file, f"image/{os.path.splitext(mask_path)[1]}")}
                data = {
                    "dataset_id": str(dataset_id),
                    "is_image": "0",  # backend expects int-bool
                    "filename": os.path.splitext(os.path.basename(mask_path))[0]
                }
                resp = await client.post(url, data=data, files=files)
                print(resp.text)
                try:
                    resp.raise_for_status()
                except Exception as e:
                    results.append({"file": mask_path, "type": "mask", "status": "failed", "error": str(e)})
                    continue
                results.append({"file": mask_path, "type": "mask", "status": "ok"})

    return {
        "success": all(r["status"] == "ok" for r in results),
        "num_uploaded_images": len(paired_image_paths),
        "num_uploaded_masks": len(mask_paths),
        "results": results,
    }

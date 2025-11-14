import os
from logging import getLogger

import httpx
from fastapi import Depends, APIRouter
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.images import Images
from app.database.masks import Masks
from paths import AUTOMATIC_SEGMENTATION_BACKEND_URL as BASE_URL

logger = getLogger(__name__)
router = APIRouter(prefix="/semantic_segmentation", tags=["semantic_segmentation"])


@router.post("/start_inference/model={model_id}&image={image_id}")
async def send_inference_job(model_registry_key: str, image_id: int, db: Session = Depends(get_session)):
    """ Sends an inference job to the semantic segmentation service.

    Args:
        model_registry_key (str): The registry key of the model you want to use.
        image_id (int): ID of the image to segment.
        db (Session): Database session dependency.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    try:
        image_path = db.query(Images.file_path).filter_by(id=image_id).first()
        mask_id = db.query(Masks.id).filter_by(image_id=image_id).first()
        return await send_inference_request(model_registry_key, image_path, mask_id)
    except Exception as e:
        logger.error(f"Batch prompted_segmentation failed: {e}")
        raise e


async def send_inference_request(model_registry_key: str, image_path: str, mask_id: int):
    """ Send a batch request to the automatic prompted_segmentation backend.

    Args:
        model_registry_key (str): The ID of the model to use for prompted_segmentation.
        image_path (list[str]): Path to image file.
        mask_id (int): The ID of the mask to add the result to.

    Returns:
        httpx.Response: The response from the prompted_segmentation backend.
    """
    url = f"{BASE_URL}/inference/start_inference/model={model_registry_key}&mask_id={mask_id}"
    files = [
        ("files",
         (os.path.basename(image_path),
                   open(image_path, "rb"),
                   f"image/{image_path.rsplit('.', maxsplit=1)[-1]}")
         )
    ]
    data = {"model_registry_key": model_registry_key, "mask_id": mask_id}
    logger.info(f"Sending request to {url} with {len(files)} files")

    try:
        async with httpx.AsyncClient(timeout=30000) as client:
            resp = await client.post(url, data=data, files=files)
            resp.raise_for_status()
            return resp
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


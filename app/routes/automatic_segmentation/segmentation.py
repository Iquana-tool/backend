from fastapi import Depends, APIRouter
from app.database import get_session
from sqlalchemy.orm import Session
import httpx
import zipfile
import numpy as np
from io import BytesIO
from PIL import Image
from app.database.images import Images
from app.database.models import Models
from paths import AUTOMATIC_SEGMENTATION_BACKEND_URL as BASE_URL
from app.routes.prompted_segmentation.util import get_masks_responses
from app.routes.mask_generation import create_masks_and_add_contours_for_images
from logging import getLogger


logger = getLogger(__name__)
router = APIRouter(prefix="/automatic_segmentation", tags=["automatic_segmentation"])


@router.post("/segment_batch")
async def segment_batch_with_backend(model_id: str, image_ids: list[int], db: Session = Depends(get_session)):
    try:
        model = db.query(Models).filter_by(id=model_id).one()
        logger.info(f"{model.name} with model_id {model_id} is being used for batch prompted_segmentation of {len(image_ids)} "
                    f"images.")
        dataset_ids = db.query(Images.dataset_id).filter(Images.id.in_(image_ids)).distinct().all()
        if len(dataset_ids) > 1:
            logger.warning("Batch prompted_segmentation is being performed on images from multiple datasets. "
                           "This may lead to unexpected results.")
        image_paths = db.query(Images.file_path).filter(Images.id.in_(image_ids)).all()
        response = await send_batch_request(model_id, image_paths)
        zip_bytes = response.content
        # Extract ZIP in-memory
        with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
            # List mask files
            names = z.namelist()
            masks = {}
            for name in names:
                # Read bytes for each image
                img_bytes = z.read(name)
                mask_img = Image.open(BytesIO(img_bytes))
                masks[name] = mask_img
        masks = list(masks.values())
        mask_responses = await get_masks_responses(masks, np.ones(len(image_ids)).tolist())
        responses = await create_masks_and_add_contours_for_images(image_ids, mask_responses)
        failed = len([response.success for response in responses if response["success"]])
        success = len(image_ids) - failed
        return {
            "success": True,
            "message": f"Successfully segmented {success} images. Failed to add masks for {failed} images.",
            "responses": responses
        }

    except Exception as e:
        logger.error(f"Batch prompted_segmentation failed: {e}")
        return {"success": False, "message": str(e)}


async def send_batch_request(model_id: str, image_paths: list[str]) -> dict:
    url = f"{BASE_URL}/segment_batch"
    files = [
        ("files", (p, open(p, "rb"), f"image/{p.split('.')[-1]}"))
        for p in image_paths
    ]
    data = {"model_id": model_id}
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, data=data, files=files)
        resp.raise_for_status()
        return resp.json()

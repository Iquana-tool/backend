import cv2
from fastapi import Depends, APIRouter
from app.database import get_session
from sqlalchemy.orm import Session
import httpx
import zipfile
import numpy as np
from io import BytesIO
from PIL import Image
from app.database.images import Images
from app.database.datasets import Labels
from paths import AUTOMATIC_SEGMENTATION_BACKEND_URL as BASE_URL
from app.routes.prompted_segmentation.util import get_masks_responses
from app.routes.mask_generation import create_masks_and_add_contours_for_images
from logging import getLogger


logger = getLogger(__name__)
router = APIRouter(prefix="/automatic_segmentation", tags=["automatic_segmentation"])


@router.post("/segment_batch")
async def segment_batch_with_backend(model_id: str, image_ids: list[int], db: Session = Depends(get_session)):
    try:
        dataset_ids = db.query(Images.dataset_id).filter(Images.id.in_(image_ids)).distinct().all()
        if len(dataset_ids) > 1:
            logger.error("Batch prompted_segmentation is being performed on images from multiple datasets. "
                           "This may lead to unexpected results.")
        image_paths = list(db.query(Images.file_path).filter(Images.id.in_(image_ids)).all())
        image_paths = [tup[0] for tup in image_paths]
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
                mask_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                masks[name] = mask_img
        masks = list(masks.values())
        labels = db.query(Labels).filter_by(dataset_id=(dataset_ids[0])[0]).all()
        label_id_to_value = {i + 1: label.id for i, label in enumerate(labels)}
        mask_responses = await get_masks_responses(masks, np.ones(len(image_ids)).tolist(), label_id_to_value)
        responses = await create_masks_and_add_contours_for_images(image_ids, mask_responses, db)
        failed = len([response["success"] for response in responses["responses"] if not response["success"]])
        success = len(image_ids) - failed
        return {
            "success": True,
            "message": f"Successfully segmented {success} images. Failed to add masks for {failed} images.",
            "responses": responses
        }

    except Exception as e:
        logger.error(f"Batch prompted_segmentation failed: {e}")
        raise e
        return {"success": False, "message": str(e)}


async def send_batch_request(model_id: str, image_paths: list[str]) -> dict:
    url = f"{BASE_URL}/segment/segment_batch"
    files = [
        ("files", (p, open(p, "rb"), "img/octet-stream"))
        for p in image_paths
    ]
    data = {"model_id": model_id}
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, data=data, files=files)
        resp.raise_for_status()
        return resp

import cv2
from fastapi import Depends, APIRouter
from app.database import get_session
from sqlalchemy.orm import Session
import httpx
import zipfile
import numpy as np
import os
from io import BytesIO
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.contours import Contours
from app.routes.masks import create_masks_and_add_contours_for_images
from paths import AUTOMATIC_SEGMENTATION_BACKEND_URL as BASE_URL
from app.routes.prompted_segmentation.util import convert_numpy_masks_to_segmentation_mask_models
from logging import getLogger


logger = getLogger(__name__)
router = APIRouter(prefix="/semantic_segmentation", tags=["semantic_segmentation"])


@router.post("/segment_batch/{model_id}")
async def segment_batch_with_backend(model_id: int, image_ids: list[int], db: Session = Depends(get_session)):
    """ Segment a batch of images using the automatic segmentation backend.

    Args:
        model_id (int): The ID of the model to use for segmentation.
        image_ids (list[int]): List of image IDs to segment.
        db (Session): Database session dependency.

    Returns:
        dict: A dictionary containing the success status, message, and responses for each image.
    """
    try:
        dataset_ids = db.query(Images.dataset_id).filter(Images.id.in_(image_ids)).distinct().all()
        if len(dataset_ids) > 1:
            logger.error("Batch segmentation is being performed on images from multiple datasets. "
                           "This may lead to unexpected results.")
        # Remove all previous contours first.
        contours = db.query(Contours).join(Masks, Contours.mask_id == Masks.id).filter(Masks.image_id.in_(image_ids)).all()
        if contours:
            logger.info(f"Deleting {len(contours)} contours from the database before batch segmentation.")
            for contour in contours:
                # Delete contours from database
                db.delete(contour)
            db.commit()
        image_paths = list(db.query(Images.file_path).filter(Images.id.in_(image_ids)).all())
        image_paths = [tup[0] for tup in image_paths]
        response = await send_batch_request(model_id, image_paths)
        zip_bytes = response.content
        # Extract ZIP in-memory
        masks = []
        os.makedirs("./temp_masks", exist_ok=True)
        with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
            # List mask files
            names = z.namelist()
            for name in names:
                # Read bytes for each image
                img_bytes = z.read(name)
                mask_arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                if False:
                    # For debugging, save the mask to a temporary directory. Set to True to enable.
                    logger.info(f"Saving mask {name} to temp_masks directory for debugging.")
                    cv2.imwrite(f"./temp_masks/{name}", mask_arr * (255 // 4))
                masks.append(mask_arr)
        labels = db.query(Labels).filter_by(dataset_id=(dataset_ids[0])[0]).all()
        label_id_to_value = {i + 1: label.id for i, label in enumerate(labels)}
        mask_responses = await convert_numpy_masks_to_segmentation_mask_models(masks,
                                                                               np.ones(len(image_ids)).tolist(),
                                                                               label_id_to_value,
                                                                               only_return_one=False)
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


async def send_batch_request(model_id: int, image_paths: list[str]):
    """ Send a batch request to the automatic segmentation backend.

    Args:
        model_id (int): The ID of the model to use for segmentation.
        image_paths (list[str]): List of image file paths to segment.

    Returns:
        httpx.Response: The response from the segmentation backend.
    """
    url = f"{BASE_URL}/segment/segment_batch"
    files = [
        ("files", (os.path.basename(p), open(p, "rb"), f"image/{p.rsplit('.', maxsplit=1)[-1]}"))
        for p in image_paths
    ]
    data = {"model_id": model_id}
    logger.info(f"Sending request to {url} with {len(files)} files")

    try:
        async with httpx.AsyncClient(timeout=30000) as client:
            resp = await client.post(url, data=data, files=files)
            # Log response details
            response_content = await resp.aread()
            resp.raise_for_status()
            return resp
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


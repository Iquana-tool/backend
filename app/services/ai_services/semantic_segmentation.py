import os
import httpx
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.schemas.contours import ContourHierarchy
from app.schemas.labels import LabelHierarchy
from app.schemas.user import System
from app.services.util import extract_mask_from_response
from app.routes import contours
from paths import SEMANTIC_SEGMENTATION_BACKEND_URL as BASE_URL
from logging import getLogger


logger = getLogger(__name__)


async def segment_image_with_semantic_model(model_registry_key, image_id, db):
    image = db.query(Images).filter_by(id=image_id).first()
    image_path = image.file_path
    dataset_id = image.dataset_id
    labels = db.query(Labels).filter_by(dataset_id=dataset_id)
    mask_id = db.query(Masks.id).filter_by(image_id=image_id).first()
    await contours.delete_all_contours_of_mask(mask_id, System(username="semantic segmentation"), db)
    response = await send_inference_request(model_registry_key,
                                            image_path,
                                            mask_id
                                            )
    mask, shape, dtype, score = extract_mask_from_response(response)
    hierarchy = ContourHierarchy.from_semantic_mask(mask_id,
                                                    mask,
                                                    LabelHierarchy.from_query(labels),
                                                    model_registry_key,
                                                    db)
    return hierarchy


async def send_inference_request(model_registry_key: str, image_path: str, mask_id: int):
    """ Send a batch request to the automatic prompted_segmentation backend.

    Args:
        model_registry_key (str): The ID of the model to use for prompted_segmentation.
        image_path (list[str]): Path to image file.
        mask_id (int): The ID of the mask to add the result to.

    Returns:
        httpx.Response: The response from the prompted_segmentation backend.
    """
    url = f"{BASE_URL}/inference/infer_image/model={model_registry_key}&mask_id={mask_id}"
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

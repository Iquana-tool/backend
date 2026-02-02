import os
from logging import getLogger

import httpx
from iquana_toolbox.schemas.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.labels import LabelHierarchy
from iquana_toolbox.schemas.service_requests import SemanticSegmentationRequest
from iquana_toolbox.schemas.training import SemanticTrainingRequest
from iquana_toolbox.schemas.user import System

from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.routes.general import contours, masks
from app.services.ai_services.base_service import BaseService
from app.services.util import extract_mask_from_response
from config import SEMANTIC_SEGMENTATION_BACKEND_URL as BASE_URL

logger = getLogger(__name__)


async def segment_image_with_semantic_model(model_registry_key, image_id, db):
    image = db.query(Images).filter_by(id=image_id).first()
    image_path = image.file_path
    dataset_id = image.dataset_id
    labels = db.query(Labels).filter_by(dataset_id=dataset_id)
    mask_id = db.query(Masks.id).filter_by(image_id=image_id).first()
    await masks.delete_all_contours_of_mask(mask_id, System(username="semantic segmentation"), db)
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


class SemanticSegmentationService(BaseService):
    def __init__(self):
        super().__init__(BASE_URL)

    async def get_models(self):
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/models/all"
            response = await client.get(url)

            response.raise_for_status()

            return response.json()

    async def get_model(self, model_registry_key: str):
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/model/{model_registry_key}"
            response = await client.get(url)

            response.raise_for_status()

            return response.json()

    async def delete_model(self, model_registry_key: str):
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/models/{model_registry_key}"
            response = await client.delete(url)

            response.raise_for_status()

            return response.json()

    async def inference(self, request: SemanticSegmentationRequest):
        """Segment an image using 2D prompts.
        Args:
            request (PromptedSegmentationWebsocketRequest): Request object.
        Returns:
            dict: A response dict
        """

        # Send the request to the backend
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/annotation_session/run"
            response = await client.post(url, json=request.model_dump())

            response.raise_for_status()

            return response.json()

    async def start_training(self, request: SemanticTrainingRequest):
        # Send the request to the backend
        async with httpx.AsyncClient(timeout=120) as client:
            url = f"{self.backend_url}/training/start"
            response = await client.post(url, json=request.model_dump())

            response.raise_for_status()

            return response.json()

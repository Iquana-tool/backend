import httpx
import numpy as np

from app.database.images import Images
from app.schemas.segmentation.segmentations import Prompts
from paths import PROMPTED_SEGMENTATION_BACKEND_URL as BASE_URL
from app.database import get_context_session
from logging import getLogger

logger = getLogger(__name__)


async def check_backend():
    """Check if the prompted segmentation backend is reachable.
    Returns:
        bool: True if the backend is reachable, False otherwise.
    """
    url = f"{BASE_URL}/health"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            return await client.get(url)
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.error(f"Error checking prompted segmentation backend: {e}")
        return False


async def upload_image(user_id: int, image_id: int):
    """Upload an image to the prompted segmentation backend.
    :param user_id: The user id.
    :param image_id: The image id.
    :returns dict: A dictionary containing the success status and message.
    """
    url = f"{BASE_URL}/annotation_session/open_image/user_uid={user_id}"
    with get_context_session() as session:
        image_path = session.query(Images.file_path).filter_by(id=image_id).first()
    with open(image_path[0], "rb") as f:
        file = {"image": f}
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(url, files=file)
            response.raise_for_status()
    return response.json()


async def select_model(user_id: int, model_identifier: str):
    """
    Preload a model into the model cache
    Args:
        user_id: The user id.
        model_identifier: The model identifier string

    Returns:
        Response message indicating success.
    """
    url = f"{BASE_URL}/models/select_model/{model_identifier}"
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(url)
        response.raise_for_status()
    return response.json()


async def get_models():
    """ List all available models."""
    url = f"{BASE_URL}/models/available"
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(url)
        response.raise_for_status()
    return response.json()


async def focus_crop(user_id: int, min_x: float, min_y: float, max_x: float, max_y: float):
    """Crop the uploaded image to the specified bounding box.
    Args:
        user_id (str): Unique identifier for the user.
        min_x (float): Minimum x-coordinate of the bounding box.
        min_y (float): Minimum y-coordinate of the bounding box.
        max_x (float): Maximum x-coordinate of the bounding box.
        max_y (float): Maximum y-coordinate of the bounding box.
    Returns:
        dict: A dictionary containing the success status and message.
    """
    url = f"{BASE_URL}/annotation_session/focus_crop/min_x={min_x}&min_y={min_y}&max_x={max_x}&max_y={max_y}&user_uid={user_id}"
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(url)
        response.raise_for_status()
    return response.json()

async def unfocus_crop(user_id: int):
    """Revert the cached image to the original uploaded image.
    Args:
        user_id (str): Unique identifier for the user.
    Returns:
        dict: A dictionary containing the success status and message.
    """
    url = f"{BASE_URL}/annotation_session/unfocus_crop/user_uid={user_id}"
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(url)
        response.raise_for_status()
    return response.json()

async def close_image(user_id: int):
    """Clear the cached image for the specified user.
    Args:
        user_id (str): Unique identifier for the user.
    Returns:
        dict: A dictionary containing the success status and message.
    """
    url = f"{BASE_URL}/annotation_session/close_image/user_uid={user_id}"
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(url)
        response.raise_for_status()
    return response.json()

async def segment_image_with_prompts(user_id: int, model_identifier: str, prompts_request: Prompts):
    """Segment an image using 2D prompts.
    Args:
        user_id (str): Unique identifier for the user.
        model_identifier (str): Identifier for the segmentation model.
        prompts_request (dict): Dictionary containing point and box prompts.
    Returns:
        dict: A response dict
    """

     # Send the request to the backend
    async with httpx.AsyncClient(timeout=120) as client:
        url = f"{BASE_URL}/annotation_session/segment_image_with_prompts/model={model_identifier}&user_uid={user_id}"
        response = await client.post(url, json=prompts_request.model_dump(exclude_none=True))

        response.raise_for_status()
        # Extract metadata from headers
        shape = tuple(map(int, response.headers["X-Mask-Shape"].split(',')))
        dtype = np.dtype(response.headers["X-Mask-Dtype"])
        score = float(response.headers.get("X-Score", 0.0))

        # Load the mask from raw bytes
        mask_bytes = response.content
        mask = np.frombuffer(mask_bytes, dtype=dtype).reshape(shape)

        return {
            "success": True,
            "message": f"Successfully computed mask from prompts with confidence of {score:.1%}",
            "mask": mask,
            "score": score,
            "shape": shape,
            "dtype": dtype,
        }

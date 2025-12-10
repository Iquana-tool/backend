import json
from abc import ABC, abstractmethod
from enum import Enum
import httpx
from logging import getLogger
from app.database import get_context_session
from app.database.contours import Contours
from app.database.images import Images

logger = getLogger(__name__)


class BaseService(ABC):
    """Base class for all service classes."""
    def __init__(self, backend_url):
        self.backend_url = backend_url
        self.enabled = True

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    async def check_backend(self):
        """Check if the backend is reachable.
        Returns:
            bool: True if the backend is reachable, False otherwise.
        """
        url = f"{self.backend_url}/health"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                return await client.get(url)
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.error(f"Error checking prompted prompted_segmentation backend: {e}")
            return False

    async def upload_image(self, user_id: str, image_id: int):
        """Upload an image to the prompted prompted_segmentation backend.
        :param user_id: The user id.
        :param image_id: The image id.
        :returns dict: A dictionary containing the success status and message.
        """
        url = f"{self.backend_url}/annotation_session/open_image/user_id={user_id}"
        with get_context_session() as session:
            image_path = session.query(Images.file_path).filter_by(id=image_id).first()
            image_path = image_path[0]

        with open(image_path, "rb") as f:
            file = {"image": f}
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(url, files=file)
                response.raise_for_status()
        return response.json()

    async def select_model(self, user_id: str, model_identifier: str):
        """
        Preload a model into the model cache
        Args:
            user_id: The user id.
            model_identifier: The model identifier string

        Returns:
            Response message indicating success.
        """
        url = f"{self.backend_url}/annotation_session/load_model/model_key={model_identifier}&user_id={user_id}"
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(url)
            response.raise_for_status()
        return response.json()

    async def get_models(self):
        """ List all available models."""
        url = f"{self.backend_url}/get_available_models"
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(url)
            response.raise_for_status()
        return response.json()

    async def focus_contour(self, user_id, contour_id):
        """Crop the uploaded image to a contour. """
        with get_context_session() as session:
            contour = session.query(Contours.x, Contours.y).filter_by(id=contour_id).first()
            x = json.loads(contour.x)
            y = json.loads(contour.y)
            min_x = min(x)
            max_x = max(x)
            min_y = min(y)
            max_y = max(y)
            return await self.focus_crop(user_id, min_x, min_y, max_x, max_y)

    async def focus_crop(self, user_id: str, min_x: float, min_y: float, max_x: float, max_y: float):
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
        url = f"{self.backend_url}/annotation_session/focus_crop/min_x={min_x}&min_y={min_y}&max_x={max_x}&max_y={max_y}&user_uid={user_id}"
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(url)
            response.raise_for_status()
        return response.json()

    async def unfocus_crop(self, user_id: str):
        """Revert the cached image to the original uploaded image.
        Args:
            user_id (str): Unique identifier for the user.
        Returns:
            dict: A dictionary containing the success status and message.
        """
        url = f"{self.backend_url}/annotation_session/unfocus_crop/user_uid={user_id}"
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(url)
            response.raise_for_status()
        return response.json()

    async def close_image(self, user_id: str):
        """Clear the cached image for the specified user.
        Args:
            user_id (str): Unique identifier for the user.
        Returns:
            dict: A dictionary containing the success status and message.
        """
        url = f"{self.backend_url}/annotation_session/close_image/user_uid={user_id}"
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(url)
            response.raise_for_status()
        return response.json()

    @abstractmethod
    async def inference(self, request):
        """ The inference endpoint of each service. Is specific to the service. Should call a Post to  API_BASE/inference with
         specific type of request. """
        pass

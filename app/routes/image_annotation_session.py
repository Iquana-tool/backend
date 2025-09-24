import json
from fastapi import APIRouter
from fastapi.websockets import WebSocket
from logging import getLogger


router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])
logger = getLogger(__name__)


@router.websocket("/ws/open_session/user={user_id}&image={image_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, image_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            # Here you can handle different types of messages based on their "type" field
            match message["type"]:
                case "prompted_segmentation":
                    # Handle prompted segmentation message
                    pass
                case "automatic_segmentation":
                    # Handle automatic segmentation message
                    pass
                case "image":
                    # Handle annotation completion message
                    pass
                case _:
                    logger.warning(f"Unknown message type received: {message['type']}")
            logger.info(f"Received data from user {user_id} for image {image_id}: {data}")
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        logger.error(f"WebSocket connection error for user {user_id} and image {image_id}: {e}")
    finally:
        await websocket.close()

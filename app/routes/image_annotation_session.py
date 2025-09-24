import json
from fastapi import APIRouter
from fastapi.websockets import WebSocket
from logging import getLogger
from app.services.ai_services import prompted_segmentation as prompted_service


router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])
logger = getLogger(__name__)


async def on_startup(image_id, user_uid: str):
    """Function to be called at the start of an annotation session. Any initialization code can be placed here.
    """
    # Check for running backends
    running = []
    if not prompted_service.check_backend():
        logger.error("Prompted segmentation backend is not reachable. Please make sure it is running.")
    else:
        running.append("prompted_segmentation")
        logger.debug("Prompted segmentation backend is reachable.")

    # Initialize backends by uploading the image etc.
    failed_initializations = []
    if not (await prompted_service.upload_image(user_uid, image_id))["success"]:
        logger.error(f"Failed to upload image {image_id} for user {user_uid} to prompted segmentation backend.")
        failed_initializations.append("prompted_segmentation")


    logger.info("Annotation session initialized.")
    return {
        "message": f"Annotation session initialized."
                   f"\nRunning backends: {running}"
                   f"\nFailed backend initializations: {failed_initializations}",
        "running_backends": running,
        "failed_initializations": failed_initializations
    }



@router.websocket("/ws/open_session/user={user_id}&image={image_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, image_id: str):
    """WebSocket endpoint to handle real-time image annotation sessions. The image annotation session takes multiple
        messages from the user as input to start tasks in the background. The messages can be for different tasks:
        - "prompted_segmentation": for prompted segmentation requests with optional prompts.
        - "automatic_segmentation": for automatic segmentation requests without prompts.
        - "contours": For editing, deleting or adding contours.
    """
    await websocket.accept()
    try:
        # Call some functions on startup
        response = await on_startup(image_id, user_id)
        await websocket.send_json(response)
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

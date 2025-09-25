import json
from fastapi import APIRouter
from fastapi.websockets import WebSocket
from logging import getLogger

from pydantic_core import ValidationError

from app.services.ai_services import prompted_segmentation as prompted_service
from uuid import UUID
from app.schemas.annotation_session import ServerMessageType, ClientMessageType, ServerMessage, ClientMessage


router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])
logger = getLogger(__name__)


async def receive_msg(websocket: WebSocket) -> ClientMessage:
    msg = await websocket.receive_json()
    try:
        msg = ClientMessage.model_validate_json(msg)
        logger.info(f"Received message: {msg}")
        return msg
    except ValidationError as e:
        # Client message couldn't be validated, send an error message
        logger.error(f"Client message couldn't be validated, sent an error message. \n{str(e)}")
        await send_msg(websocket,
                       ServerMessage(
                           id="0",
                           type=ServerMessageType.ERROR,
                           message=f"Client message could not be validated. See error here:\n{str(e)}",
                           data=None,
                           success=False
                       ))
        return ClientMessage(
            id="0",
            type=ClientMessageType.ERROR,
            success=False,
        )


async def send_msg(websocket: WebSocket, message: ServerMessage):
    logger.info(f"Sending message: {message}")
    await websocket.send_json(message.model_dump_json())


async def on_startup(image_id, user_uid: str) -> ServerMessage:
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
    return ServerMessage(
        id="test",
        type=ServerMessageType.SESSION_INITIALIZED,
        success=len(failed_initializations) == 0,
        message=f"Annotation session initialized."
                f"\nRunning backends: {running}"
                f"\n{f'Failed initializations: {failed_initializations}' if failed_initializations else ''}",
        data={
            "running": running,
            "failed": failed_initializations,
        }
    )



@router.websocket("/ws/annotation_session/user={user_id}&image={image_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, image_id: str):
    """WebSocket endpoint to handle real-time image annotation sessions. The image annotation session takes multiple
        messages from the user as input to start tasks in the background.
        Client sent messages should be structured as JSON and should look like this: \n
        { \n
        "type": "prompted_segmentation" | "automatic_segmentation" | "image", \n
        "data": { ... }  # Data specific to the message type \n
        } \n
        For info on the message types and their data structure, see the respective documentation.

        Server responses will also be structured as JSON and will contain the results of the requested tasks: \n
        { \n
        "type": "response_type",  # Type of the response, e.g., "prompted_segmentation_result" \n
        "success": True | False,  # Indicates if the task was successful \n
        "message": "Informational message about the response", \n
        "data": { ... }  # Data specific to the response type \n
        } \n
        The server may also send status updates or error messages as needed. The response types and their data structure
        will depend on the tasks performed and the results obtained.

        :param websocket: The WebSocket connection.
        :param user_id: Unique identifier for the user.
        :param image_id: Unique identifier for the image to be annotated.
        :raises WebsocketException: If the WebSocket connection fails.
    """
    await websocket.accept()
    try:
        # Call some functions on startup
        response = await on_startup(image_id, user_id)
        # Send a message about startup
        await send_msg(websocket, response)
        while True:
            client_msg = await receive_msg(websocket)
            # Here we handle different types of messages based on their "type" field
            match client_msg.type:
                case ClientMessageType.FOCUS_IMAGE:
                    handle_focus_image(client_msg)
                case ClientMessageType.UNFOCUS_IMAGE:
                    handle_unfocus_image(client_msg)
                case ClientMessageType.OBJECT_ADD:
                    handle_object_add(client_msg)
                case ClientMessageType.OBJECT_DELETE:
                    handle_object_delete(client_msg)
                case ClientMessageType.OBJECT_MODIFY:
                    handle_object_modify(client_msg)
                case ClientMessageType.AUTOMATIC_SELECT_MODEL:
                    handle_automatic_select_model(client_msg)
                case ClientMessageType.AUTOMATIC_SEGMENTATION:
                    handle_automatic_segmentation(client_msg)
                case ClientMessageType.PROMPTED_SELECT_MODEL:
                    handle_prompted_select_model(client_msg)
                case ClientMessageType.PROMPTED_SEGMENTATION:
                    handle_prompted_segmentation(client_msg)
                case ClientMessageType.COMPLETION_SELECT_MODEL:
                    handle_completion_select_model(client_msg)
                case ClientMessageType.COMPLETION_ENABLE:
                    handle_completion_enable(client_msg)
                case ClientMessageType.COMPLETION_DISABLE:
                    handle_completion_disable(client_msg)
                case ClientMessageType.FINISH_ANNOTATION:
                    handle_finish_annotation(client_msg)
                case ClientMessageType.OBJECT_CONFLICT_RESOLUTION:
                    handle_object_conflict_resolve(client_msg)
                case _:
                    # Ignore erroneous messages from the client
                    pass
    except Exception as e:
        logger.error(f"WebSocket connection error for user {user_id} and image {image_id}: {e}")
    finally:
        await websocket.close()



def handle_focus_image(websocket, data):
    """ Handle the client sending a focus image request"""
    pass


def handle_unfocus_image(websocket, data):
    """ Handle the client unfocussing."""
    pass


def handle_object_add(websocket, data):
    """ Handle adding an object to the mask."""
    pass


def handle_object_delete(websocket, data):
    """ Handle removing an object from the mask. """
    pass


def handle_object_modify(websocket, data):
    """ Handle Modifying an object. """
    pass


def handle_automatic_select_model(websocket, data):
    """ Handle the selection of an automatic model. """
    pass


def handle_automatic_segmentation(websocket, data):
    """ Handle segmentation using an automatic model. """
    pass


def handle_prompted_select_model(websocket, data):
    """ Handle the selection of a prompted model. """
    pass


def handle_prompted_segmentation(websocket, data):
    """ Handle segmentation using a prompted model. """
    pass


def handle_completion_select_model(websocket, data):
    """ Handle the selection of a completion model. """
    pass


def handle_completion_enable(websocket, data):
    """ Handle enabling of completion model. Leads to a state change. """
    pass


def handle_completion_disable(websocket, data):
    """ Handle disabling of completion model. Leads to a state change. """
    pass


def handle_finish_annotation(websocket, data):
    """ Handle marking a mask as finished. """
    pass


def handle_object_conflict_resolve(websocket, data):
    """ Handle how an object conflict should be resolved. """
    pass

from logging import getLogger

from fastapi import APIRouter
from fastapi.websockets import WebSocket
from iquana_toolbox.schemas.networking.websockets.annotation_session import (
    ServerMessageType,
    ClientMessageType,
    ServerMessage,
)
from starlette.websockets import WebSocketDisconnect

from app.routes.websockets import annotation_handlers as handlers
from app.routes.websockets.messaging import receive_msg, send_msg
from app.services.annotation_session.state import AnnotationSessionState

router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])
logger = getLogger(__name__)


@router.websocket("/ws/{user_id}/{image_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, image_id: int):
    """WebSocket endpoint to handle real-time image annotation sessions. The image annotation session takes multiple
        messages from the user as input to start tasks in the background.
        Client sent messages should be structured as JSON and should look like this: \n
        { \n
        "type": "prompted_segmentation" | "semantic_segmentation" | "image", \n
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
    logger.info(f"WebSocket connection accepted for user {user_id} and image {image_id}")
    state = AnnotationSessionState(
        image_id=image_id,
        mask_id=None,
        user_id=user_id,
    )
    try:
        # Call some functions on startup
        logger.info(f"Calling on startup for user {user_id} and image {image_id}")
        await handlers.startup(websocket, state)
        while True:
            client_msg = await receive_msg(websocket)
            # Here we handle different types of messages based on their "type" field
            try:
                match client_msg.type:
                    case ClientMessageType.FOCUS_IMAGE:
                        await handlers.handle_focus_image(websocket, client_msg, state)
                    case ClientMessageType.UNFOCUS_IMAGE:
                        await handlers.handle_unfocus_image(websocket, client_msg, state)
                    case ClientMessageType.SELECT_REFINEMENT_OBJECT:
                        await handlers.handle_select_refinement_object(websocket, client_msg, state)
                    case ClientMessageType.UNSELECT_REFINEMENT_OBJECT:
                        await handlers.handle_unselect_refinement_object(websocket, client_msg, state)
                    case ClientMessageType.OBJECT_ADD_MANUAL:
                        await handlers.handle_object_add(websocket, client_msg, state)
                    case ClientMessageType.OBJECT_FINALISE:
                        await handlers.handle_object_finalise(websocket, client_msg, state)
                    case ClientMessageType.OBJECT_DELETE:
                        await handlers.handle_object_delete(websocket, client_msg, state)
                    case ClientMessageType.OBJECT_MODIFY:
                        await handlers.handle_object_modify(websocket, client_msg, state)
                    case ClientMessageType.PROMPTED_SELECT_MODEL:
                        await handlers.handle_prompted_select_model(websocket, client_msg, state)
                    case ClientMessageType.PROMPTED_INFERENCE:
                        await handlers.handle_prompted_segmentation(websocket, client_msg, state)
                    case ClientMessageType.COMPLETION_SELECT_MODEL:
                        await handlers.handle_suggestion_select_model(websocket, client_msg, state)
                    case ClientMessageType.COMPLETION_ENABLE:
                        await handlers.handle_suggestion_enable(websocket, client_msg, state)
                    case ClientMessageType.COMPLETION_DISABLE:
                        await handlers.handle_suggestion_disable(websocket, client_msg, state)
                    case ClientMessageType.COMPLETION_INFERENCE:
                        await handlers.handle_suggestion(websocket, client_msg, state)
                    case ClientMessageType.INSTANCE_SELECT_MODEL:
                        await handlers.handle_instance_select_model(websocket, client_msg, state)
                    case ClientMessageType.INSTANCE_INFERENCE:
                        await handlers.handle_instance_segmentation(websocket, client_msg, state)
                    case ClientMessageType.FINISH_ANNOTATION:
                        await handlers.handle_finish_annotation(websocket, client_msg, state)
                    case ClientMessageType.OBJECT_CONFLICT_RESOLUTION:
                        await handlers.handle_object_conflict_resolve(websocket, client_msg, state)
                    case _:
                        # Ignore erroneous messages from the client
                        pass
            except Exception as e:
                # A single message failing should not tear down the session. Report the error
                # back to the client and keep the connection open for further messages.
                logger.error(f"Ran into an error handling message: {e} \n Message: {client_msg}")
                await send_msg(websocket, ServerMessage(
                    id=client_msg.id,
                    type=ServerMessageType.ERROR,
                    message=f"An error occurred: {str(e)}",
                    success=False,
                    data=None
                ))
                # Loop continues; the websocket stays connected.
    except WebSocketDisconnect:
        # Client disconnected normally, just log and exit
        logger.info(f"WebSocket disconnected for user {user_id} and image {image_id}")
        print(f"WebSocket disconnected for user {user_id} and image {image_id}")
    except Exception as e:
        # Fallback
        logger.error(f"WebSocket connection error for user {user_id} and image {image_id}: {e}")
        print(f"Error: {e}")
        # Try to send error message if websocket is still open
        try:
            await send_msg(websocket, ServerMessage(
                id="error",
                type=ServerMessageType.ERROR,
                message=f"An error occurred: {str(e)}",
                success=False,
                data=None
            ))
        except Exception:
            # Websocket might already be closed, ignore
            pass
        finally:
            # This will throw an error, which is better for debugging, but should be removed when deployed.
            raise e
    finally:
        # Only close if websocket is still open
        try:
            if websocket.client_state.name != "DISCONNECTED":
                await websocket.close()
        except Exception:
            # Websocket might already be closed, ignore
            pass
        # Save the session state and the mask to disk
        # TODO save the session state
        pass

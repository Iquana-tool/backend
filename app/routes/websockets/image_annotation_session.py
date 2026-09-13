from logging import getLogger

from fastapi import APIRouter
from fastapi.websockets import WebSocket
from iquana_toolbox.schemas.networking.websockets.annotation_session import (
    ServerMessageType,
    ClientMessageType,
    ServerMessage,
)
from starlette.websockets import WebSocketDisconnect

from app.database import get_context_session
from app.routes.websockets import annotation_handlers as handlers
from app.routes.websockets.messaging import receive_msg, send_msg
from app.schemas.permissions import Permission
from app.services.annotation_session.state import AnnotationSessionState
from app.services.auth import authenticate_websocket
from app.services.permissions import dataset_id_for_image

router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])
logger = getLogger(__name__)

#: Sent before the socket is closed on an auth failure (RFC 6455 policy violation).
_POLICY_VIOLATION = 1008


@router.websocket("/ws/{user_id}")
@router.websocket("/ws/{user_id}/{image_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, image_id: int | None = None):
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

        The socket belongs to a *user*, not to an image. ``image_id`` is optional: a
        client may connect once and then re-target the session with a ``switch_image``
        message for every image it shows. Passing an image in the URL is still supported
        and simply pre-selects the first one.

        The connection must carry a bearer token (``?token=...``, or an
        Authorization header for non-browser clients). The caller's identity comes
        from that token: the ``user_id`` in the path is not trusted, because
        anything else would let a client annotate as any user simply by editing the
        URL. Annotating also requires `annotation.create` on the image's dataset --
        re-checked on every switch, since images may come from different datasets.

        :param websocket: The WebSocket connection.
        :param user_id: Display identifier from the URL. Ignored for authorisation.
        :param image_id: Unique identifier for the image to be annotated. Optional.
        :raises WebsocketException: If the WebSocket connection fails.
    """
    await websocket.accept()

    dataset_id = None
    with get_context_session() as db:
        user = await authenticate_websocket(websocket, db)
        if user is None:
            logger.warning("Rejecting unauthenticated annotation session for image %s.", image_id)
            await websocket.close(code=_POLICY_VIOLATION, reason="Authentication required.")
            return

        if image_id is not None:
            dataset_id = dataset_id_for_image(image_id, db)
            if dataset_id is None:
                await websocket.close(code=_POLICY_VIOLATION, reason="Unknown image.")
                return
            if not user.has_permission(dataset_id, Permission.ANNOTATION_CREATE):
                logger.warning("%s lacks annotation rights on dataset %s.", user.username, dataset_id)
                await websocket.close(code=_POLICY_VIOLATION,
                                      reason="You do not have permission to annotate this dataset.")
                return

    if user_id != user.username:
        logger.info("Ignoring URL user_id %r; session belongs to %s.", user_id, user.username)

    logger.info(f"WebSocket connection accepted for user {user.username} and image {image_id}")
    state = AnnotationSessionState(
        image_id=image_id,
        mask_id=None,
        user_id=user.username,
        dataset_id=dataset_id,
    )
    try:
        # Call some functions on startup
        logger.info(f"Calling on startup for user {user_id} and image {image_id}")
        await handlers.startup(websocket, state)
        while True:
            client_msg = await receive_msg(websocket)
            if client_msg is None:
                # Unparseable message. The client has already been told what was wrong;
                # dropping the connection over it would cost the user their session.
                continue

            # Everything except switching needs an image to act on. A session opened
            # without one (the per-user socket) is idle until the client picks one.
            if state.image_id is None and client_msg.type != ClientMessageType.SWITCH_IMAGE:
                await send_msg(websocket, ServerMessage(
                    id=client_msg.id,
                    type=ServerMessageType.ERROR,
                    message="No image selected. Send a switch_image message first.",
                    success=False,
                    data=None,
                ))
                continue

            # Here we handle different types of messages based on their "type" field
            try:
                match client_msg.type:
                    case ClientMessageType.SWITCH_IMAGE:
                        await handlers.handle_switch_image(websocket, client_msg, state)
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
                    case ClientMessageType.SUGGESTION_SELECT_MODEL:
                        await handlers.handle_suggestion_select_model(websocket, client_msg, state)
                    case ClientMessageType.SUGGESTION_ENABLE:
                        await handlers.handle_suggestion_enable(websocket, client_msg, state)
                    case ClientMessageType.SUGGESTION_DISABLE:
                        await handlers.handle_suggestion_disable(websocket, client_msg, state)
                    case ClientMessageType.SUGGESTION_INFERENCE:
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
            except WebSocketDisconnect:
                # The client went away mid-handler. Not an error, and there is nobody
                # left to report it to - let the outer handler close the session.
                raise
            except Exception as e:
                # A single message failing should not tear down the session. Report the error
                # back to the client and keep the connection open for further messages.
                # The traceback is logged, not just the message: swallowing an exception
                # here used to leave nothing to debug from beyond its str().
                logger.exception(f"Ran into an error handling message: {e} \n Message: {client_msg}")
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
        logger.info(f"WebSocket disconnected for user {user.username} and image {state.image_id}")
    except Exception as e:
        # Fallback: anything the per-message handling above did not catch, e.g. a failure
        # in startup() or while sending on a socket the client has already dropped.
        logger.exception(
            f"WebSocket connection error for user {user.username} and image {state.image_id}: {e}"
        )
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
        # Deliberately not re-raised. Starlette turns an exception escaping a WebSocket
        # endpoint into a 1011 close and an unhandled-error log; the traceback is already
        # recorded above, and swallowing it keeps a single bad session from being reported
        # as a server fault.
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

from logging import getLogger
from time import perf_counter

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
from app.services.telemetry.emit import emit_api, emit_navigation

router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])
logger = getLogger(__name__)

#: Sent before the socket is closed on an auth failure (RFC 6455 policy violation).
_POLICY_VIOLATION = 1008


@router.websocket("/ws/{user_id}/{image_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, image_id: int,
                             telemetry_session: str | None = None):
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

        The connection must carry a bearer token (``?token=...``, or an
        Authorization header for non-browser clients). The caller's identity comes
        from that token: the ``user_id`` in the path is not trusted, because
        anything else would let a client annotate as any user simply by editing the
        URL. Annotating also requires `annotation.create` on the image's dataset.

        :param websocket: The WebSocket connection.
        :param user_id: Display identifier from the URL. Ignored for authorisation.
        :param image_id: Unique identifier for the image to be annotated.
        :param telemetry_session: Study session id, so the events this socket emits
            join the participant's timeline. A handshake cannot carry the
            ``X-Telemetry-Session`` header the HTTP routes use, hence the query
            parameter. Purely for grouping captured events; never used for access.
        :raises WebsocketException: If the WebSocket connection fails.
    """
    await websocket.accept()

    with get_context_session() as db:
        user = await authenticate_websocket(websocket, db)
        if user is None:
            logger.warning("Rejecting unauthenticated annotation session for image %s.", image_id)
            await websocket.close(code=_POLICY_VIOLATION, reason="Authentication required.")
            return

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
        telemetry_session=telemetry_session,
    )
    try:
        # Call some functions on startup
        logger.info(f"Calling on startup for user {user_id} and image {image_id}")
        await handlers.startup(websocket, state)
        emit_navigation("session.ws_open", username=state.user_id, session_id=telemetry_session,
                        dataset_id=dataset_id, image_id=image_id)
        while True:
            client_msg = await receive_msg(websocket)
            # Here we handle different types of messages based on their "type" field
            # One `api.request`-equivalent event per message, timed around the whole
            # dispatch. Doing it here rather than in each handler means every current
            # and future message type is covered by one call site.
            handler_started = perf_counter()
            handler_error: Exception | None = None
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
            except Exception as e:
                # A single message failing should not tear down the session. Report the error
                # back to the client and keep the connection open for further messages.
                handler_error = e
                logger.error(f"Ran into an error handling message: {e} \n Message: {client_msg}")
                await send_msg(websocket, ServerMessage(
                    id=client_msg.id,
                    type=ServerMessageType.ERROR,
                    message=f"An error occurred: {str(e)}",
                    success=False,
                    data=None
                ))
                # Loop continues; the websocket stays connected.
            finally:
                payload = {
                    "type": getattr(client_msg.type, "value", str(client_msg.type)),
                    "ok": handler_error is None,
                }
                if handler_error is not None:
                    payload["error"] = type(handler_error).__name__
                emit_api("ws.message",
                         username=state.user_id,
                         session_id=telemetry_session,
                         dataset_id=state.dataset_id,
                         image_id=state.image_id,
                         duration_ms=int((perf_counter() - handler_started) * 1000),
                         payload=payload)
    except WebSocketDisconnect:
        # Client disconnected normally, just log and exit
        logger.info(f"WebSocket disconnected for user {user.username} and image {image_id}")
        emit_navigation("session.ws_close", username=user.username, session_id=telemetry_session,
                        dataset_id=dataset_id, image_id=image_id)
    except Exception as e:
        # Fallback
        logger.error(f"WebSocket connection error for user {user.username} and image {image_id}: {e}")
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

"""WebSocket transport helpers for the annotation session.

Kept separate from both the dispatch endpoint and the handlers so that handlers can send
messages without importing the endpoint module (avoids a circular import).
"""

from json import JSONDecodeError
from logging import getLogger

from fastapi.websockets import WebSocket
from iquana_toolbox.schemas.networking.websockets.annotation_session import (
    ServerMessageType,
    ServerMessage,
    ClientMessage,
)
from pydantic_core import ValidationError

logger = getLogger(__name__)


async def receive_msg(websocket: WebSocket) -> ClientMessage | None:
    """Read one client message, or None if it could not be understood.

    A malformed message is the client's problem, not the connection's: it is reported
    back and ``None`` is returned so the caller can carry on reading. This used to raise,
    which unwound the whole session loop and disconnected the user -- losing their AI
    backends and forcing a full reconnect over a single bad payload.

    A dropped connection still raises (``WebSocketDisconnect``), because there is nothing
    left to read.
    """
    try:
        raw = await websocket.receive_json()
    except JSONDecodeError as e:
        logger.error(f"Client sent a payload that is not JSON: {e}")
        await _try_send_error(websocket, "0", f"Message is not valid JSON: {e}")
        return None

    try:
        msg = ClientMessage.model_validate(raw)
        logger.info(f"Received message: {msg}")
        return msg
    except ValidationError as e:
        # Client message couldn't be validated, send an error message
        logger.error(f"Client message couldn't be validated, sent an error message. \n{str(e)}")
        # Echo the id when the payload had a usable one, so the client can fail the
        # request that caused this instead of letting it hang until its timeout.
        message_id = str(raw.get("id")) if isinstance(raw, dict) and raw.get("id") else "0"
        await _try_send_error(
            websocket, message_id,
            f"Client message could not be validated. See error here:\n{str(e)}",
        )
        return None


async def send_msg(websocket: WebSocket, message: ServerMessage):
    logger.info(f"Sending message: {message}")
    await websocket.send_json(message.model_dump_json())


async def _try_send_error(websocket: WebSocket, message_id: str, text: str) -> None:
    """Best-effort error reply. Never raises: the socket may already be gone."""
    try:
        await send_msg(websocket, ServerMessage(
            id=message_id,
            type=ServerMessageType.ERROR,
            message=text,
            data=None,
            success=False,
        ))
    except Exception:
        logger.debug("Could not deliver error to the client; the socket is likely closed.")

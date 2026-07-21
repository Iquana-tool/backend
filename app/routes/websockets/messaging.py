"""WebSocket transport helpers for the annotation session.

Kept separate from both the dispatch endpoint and the handlers so that handlers can send
messages without importing the endpoint module (avoids a circular import).
"""

from logging import getLogger

from fastapi.websockets import WebSocket
from iquana_toolbox.schemas.networking.websockets.annotation_session import (
    ServerMessageType,
    ServerMessage,
    ClientMessage,
)
from pydantic_core import ValidationError

logger = getLogger(__name__)


async def receive_msg(websocket: WebSocket) -> ClientMessage:
    msg = await websocket.receive_json()
    print("Received message JSON:", msg)
    try:
        msg = ClientMessage.model_validate(msg)
        logger.info(f"Received message: {msg}")
        return msg
    except ValidationError as e:
        # Client message couldn't be validated, send an error message
        logger.error(f"Client message couldn't be validated, sent an error message. \n{str(e)}")
        try:
            await send_msg(websocket,
                           ServerMessage(
                               id="0",
                               type=ServerMessageType.ERROR,
                               message=f"Client message could not be validated. See error here:\n{str(e)}",
                               data=None,
                               success=False
                           ))
        except Exception:
            # Websocket might already be closed, ignore
            pass
        raise e


async def send_msg(websocket: WebSocket, message: ServerMessage):
    logger.info(f"Sending message: {message}")
    await websocket.send_json(message.model_dump_json())

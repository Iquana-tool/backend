import json
from typing import List

from fastapi import APIRouter
from fastapi.websockets import WebSocket
from logging import getLogger

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import ValidationError

from app.database import get_context_session
from app.database.images import Images
from app.database.masks import Masks
from app.routes.contours import add_contour, get_contours_of_mask, finalise, delete_contour, modify_contour
from app.routes.masks import create_mask, finish_mask
from app.schemas.prompted_segmentation.prompts import Prompts
from app.services.ai_services import prompted_segmentation as prompted_service
from app.schemas.annotation_session import ServerMessageType, ClientMessageType, ServerMessage, ClientMessage
from app.schemas.contours import Contour
from app.services.ai_services.prompted_segmentation import select_model, segment_image_with_prompts, focus_contour, \
    unfocus_crop
from app.services.contours import get_contours_from_binary_mask

router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])
logger = getLogger(__name__)


class AnnotationSessionState(BaseModel):
    """ A class to track the state of the annotation session. """
    image_id: int = Field(..., title="Image ID", description="The id of the image whose annotations are being done.")
    mask_id: int = Field(default=None,
                         init=False,  # Isnt initialized in the init
                         title="Mask ID",
                         description="The id of the mask who annotations are being added to.")
    user_id: int = Field(...,
                         title="User ID",
                         description="The id of the user who is annotating.")
    annotation_completion_enabled: bool = Field(...,
                                                title="Annotation completion",
                                                description="Whether the annotation completion is enabled.")
    awaiting_response: dict[str, ClientMessageType] = Field(default_factory=dict,
                                                            title="Messages await responses",
                                                            description="A dict mapping from message ids to the expected"
                                                                        " client message. This is needed for prompting"
                                                                        " user input and continuing the loop.")
    conflicts: dict[int, List[int]] = Field(default_factory=dict,
                                            title="Object conflicts",
                                            description="A dict mapping from object ids to a list of "
                                                    "object ids that are conflicting with it. This means"
                                                    " that the object for example overlaps with another.")
    focussed_contour_id: int | None = Field(default=None,
                                            description="The id of the focussed contour.")


    @field_validator("image_id", mode="before")
    def validate_image_id(cls, value, values):
        with get_context_session() as session:
            if not session.query(Images).filter_by(image_id=value).exists():
                raise ValidationError(f"Image ID {value} does not exist.")
        return value

    @model_validator(mode="after")
    def validate_model(self):
        """ Validate the model and fill fields that were not initialized yet."""
        with get_context_session() as session:
            if not session.query(Masks).filter_by(image_id=self.image_id).exists():
                create_mask(self.image_id)
            mask_id = session.query(Masks.id).filter_by(image_id=self.image_id).first()
            self.mask_id = mask_id


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


async def on_startup(state: AnnotationSessionState) -> ServerMessage:
    """Function to be called at the start of an annotation session. Any initialization code can be placed here.
    """
    user_id = state.user_id
    image_id = state.image_id
    # Check for running backends
    running = []
    if not prompted_service.check_backend():
        logger.error("Prompted prompted_segmentation backend is not reachable. Please make sure it is running.")
    else:
        running.append("prompted_segmentation")
        logger.debug("Prompted prompted_segmentation backend is reachable.")

    # Initialize backends by uploading the image etc.
    failed_initializations = []
    if not (await prompted_service.upload_image(user_id, image_id))["success"]:
        logger.error(f"Failed to upload image {image_id} for user {user_id} to prompted prompted_segmentation backend.")
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
async def websocket_endpoint(websocket: WebSocket, user_id: int, image_id: int):
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
    state = AnnotationSessionState(
        image_id=image_id,
        user_id=user_id,
        annotation_completion_enabled=False,
    )
    try:
        # Call some functions on startup
        response = await on_startup(state)
        # Send a message about startup
        await send_msg(websocket, response)
        while True:
            client_msg = await receive_msg(websocket)
            # Here we handle different types of messages based on their "type" field
            match client_msg.type:
                case ClientMessageType.FOCUS_IMAGE:
                    await handle_focus_image(websocket, client_msg, state)
                case ClientMessageType.UNFOCUS_IMAGE:
                    await handle_unfocus_image(websocket, client_msg, state)
                case ClientMessageType.OBJECT_ADD_MANUAL:
                    await handle_object_add(websocket, client_msg, state)
                case ClientMessageType.OBJECT_FINALISE:
                    await handle_object_finalise(websocket, client_msg, state)
                case ClientMessageType.OBJECT_DELETE:
                    await handle_object_delete(websocket, client_msg, state)
                case ClientMessageType.OBJECT_MODIFY:
                    await handle_object_modify(websocket, client_msg, state)
                case ClientMessageType.AUTOMATIC_SELECT_MODEL:
                    await handle_automatic_select_model(websocket, client_msg, state)
                case ClientMessageType.AUTOMATIC_SEGMENTATION:
                    await handle_automatic_segmentation(websocket, client_msg, state)
                case ClientMessageType.PROMPTED_SELECT_MODEL:
                    await handle_prompted_select_model(websocket, client_msg, state)
                case ClientMessageType.PROMPTED_SEGMENTATION:
                    await handle_prompted_segmentation(websocket, client_msg, state)
                case ClientMessageType.COMPLETION_SELECT_MODEL:
                    await handle_completion_select_model(websocket, client_msg, state)
                case ClientMessageType.COMPLETION_ENABLE:
                    await handle_completion_enable(websocket, client_msg, state)
                case ClientMessageType.COMPLETION_DISABLE:
                    await handle_completion_disable(websocket, client_msg, state)
                case ClientMessageType.FINISH_ANNOTATION:
                    await handle_finish_annotation(websocket, client_msg, state)
                case ClientMessageType.OBJECT_CONFLICT_RESOLUTION:
                    await handle_object_conflict_resolve(websocket, client_msg, state)
                case _:
                    # Ignore erroneous messages from the client
                    pass
    except Exception as e:
        logger.error(f"WebSocket connection error for user {user_id} and image {image_id}: {e}")
    finally:
        await websocket.close()



async def handle_focus_image(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle the client sending a focus image request"""
    focussed_contour_id = client_msg.data.get("focussed_contour_id")
    state.focussed_contour_id = focussed_contour_id
    response = await focus_contour(state.user_id, focussed_contour_id)
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.SUCCESS if response["success"] else ServerMessageType.ERROR,
        success=response["success"],
        message=response["message"],
        data=None
    ))


async def handle_unfocus_image(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle the client unfocussing."""
    state.focussed_contour_id = None
    response = await unfocus_crop(state.user_id)
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.SUCCESS if response["success"] else ServerMessageType.ERROR,
        success=response["success"],
        message=response["message"],
        data=None
    ))

async def handle_object_add(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle adding an object to the mask."""
    contour = Contour.model_validate_json(client_msg.data)
    response = await add_contour(state.mask_id, contour, f"User {state.user_id}")
    if not response["success"]:
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.ERROR,
            message=response["message"],
            success=False,
            data=None,
        ))
    else:
        updated_objects = await get_contours_of_mask(state.mask_id)
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.OBJECT_ADDED,
            message=response["message"],
            success=True,
            data=updated_objects
        ))


async def handle_object_finalise(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Turn a temporary object into a non temporary one. For example: Temporary objects are added by AI models, if you
        make them non temporary, they will be added to the mask and can be used for training.
    """
    contour_id = client_msg.data.get("contour_id")
    response = await finalise(contour_id)
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.OBJECT_MODIFIED if response["success"] else ServerMessageType.ERROR,
        message=response["message"],
        success=response["success"],
        data={
            "contour_id": contour_id,
            "fields_to_be_updated": {
                "temporary": False
            } if response["success"] else None,
        }
    ))


async def handle_object_delete(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle removing an object from the mask. """
    contour_id = client_msg.data.get("contour_id")
    response = await delete_contour(contour_id)
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.OBJECT_REMOVED if response["success"] else ServerMessageType.ERROR,
        success=response["success"],
        message=response["message"],
        data={
            "deleted_contours": response["deleted_contours"],
        } if response["success"] else None,
    ))


async def handle_object_modify(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle Modifying an object. """
    contour_id = client_msg.data.get("contour_id")
    fields_to_be_updated = client_msg.data.get("fields_to_be_updated")
    if "label" in fields_to_be_updated:
        fields_to_be_updated.pop("label")
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.WARNING,
            message=f"You are trying to update the label of an object, this is not supported yet."
                    f"{'\nUpdating remaining fields.' if fields_to_be_updated else 'Nothing else to update.'}",
            success=False,
            data=None
        ))
    if fields_to_be_updated:
        response = await modify_contour(contour_id, **fields_to_be_updated)
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.OBJECT_MODIFIED if response["success"] else ServerMessageType.ERROR,
            message=response["message"],
            success=response["success"],
            data={
                "contour_id": contour_id,
                "fields_to_be_updated": fields_to_be_updated,
            } if response["success"] else None,
        ))


async def handle_automatic_select_model(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle the selection of an automatic model. """
    raise NotImplementedError("Method not implemented yet!")


async def handle_automatic_segmentation(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle prompted_segmentation using an automatic model. """
    raise NotImplementedError("Method not implemented yet!")


async def handle_prompted_select_model(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle the selection of a prompted model. """
    selected_model = client_msg.data.get("selected_model")
    response = await select_model(state.user_id, selected_model)
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.SUCCESS if response["success"] else ServerMessageType.ERROR,
        success=response["success"],
        message=response["message"],
        data=None
    ))


async def handle_prompted_segmentation(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle prompted_segmentation using a prompted model. """
    model_identifier = client_msg.data.get("model_identifier")
    prompts_data = client_msg.data.get("prompts")
    prompts_model = Prompts.model_value(prompts_data)
    response_seg = await segment_image_with_prompts(state.user_id, model_identifier, prompts_model)
    contour = get_contours_from_binary_mask(response_seg["mask"], only_return_biggest=True)
    contour_model = Contour(x=contour[..., 1].tolist(),
                            y=contour[..., 0].tolist(),
                            label=None,
                            parent_id=state.focussed_contour_id,
                            confidence=response_seg['score']
                            )
    response = await add_contour(
        state.mask_id,
        contour_to_add=contour_model,
        added_by=model_identifier,
        temporary=True,
    )
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.OBJECT_ADDED if response["success"] else ServerMessageType.ERROR,
        success=response["success"],
        message=f"Successfully segmented object with confidence score {response_seg['score']:.1%}" if response["success"] else response["message"],
        data=response["added_contour"] if response["success"] else None,
    ))



async def handle_completion_select_model(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle the selection of a completion model. """
    raise NotImplementedError("Method not implemented yet!")


async def handle_completion_enable(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle enabling of completion model. Leads to a state change. """
    state.annotation_completion_enabled = True
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.SUCCESS,
        success=True,
        message="Enabled annotation completion",
        data=None
    ))


async def handle_completion_disable(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle disabling of completion model. Leads to a state change. """
    state.annotation_completion_enabled = False
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.SUCCESS,
        success=True,
        message="Disabled annotation completion",
        data=None
    ))


async def handle_finish_annotation(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle marking a mask as finished. """
    response = await finish_mask(state.mask_id)
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.SUCCESS if response["success"] else ServerMessageType.ERROR,
        success=response["success"],
        message=response["message"],
        data=None
    ))


async def handle_object_conflict_resolve(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle how an object conflict should be resolved. """
    raise NotImplementedError("Method not implemented yet!")

from enum import StrEnum
from functools import cached_property
from logging import getLogger

from fastapi import APIRouter
from fastapi.websockets import WebSocket
from pydantic import field_validator, BaseModel, Field, PrivateAttr
from pydantic_core import ValidationError
from iquana_toolbox.schemas.annotation_session import ServerMessageType, ClientMessageType, ServerMessage, ClientMessage
from iquana_toolbox.schemas.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.contours import Contour
from iquana_toolbox.schemas.labels import Label
from iquana_toolbox.schemas.prompts import Prompts
from iquana_toolbox.schemas.service_requests import CompletionRequest as CompletionServiceRequest
from iquana_toolbox.schemas.service_requests import PromptedSegmentationRequest, SemanticSegmentationRequest
from starlette.websockets import WebSocketDisconnect

from app.database import get_context_session
from app.database.contours import Contours, save_contour_tree
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.routes.general.contours import add_contour_review, delete_contour, \
    modify_contour, replace_contour
from app.routes.general.images import create_new_mask_for_image
from app.routes.general.masks import mark_as_fully_annotated, get_contours_of_mask, add_contour
from app.services.ai_services.base_service import BaseService
from app.services.ai_services.completion_segmentation import CompletionService
from app.services.ai_services.prompted_segmentation import PromptedSegmentationService
from app.services.ai_services.semantic_segmentation import SemanticSegmentationService
from app.services.database_access import get_height_width_of_image

router = APIRouter(prefix="/annotation_session", tags=["annotation_session"])
logger = getLogger(__name__)


class Backends(StrEnum):
    PROMPTED_SEGMENTATION = "prompted_segmentation"
    COMPLETION_SEGMENTATION = "completion_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"


class AnnotationSessionState(BaseModel):
    """ A class to track the state of the annotation session. """
    image_id: int = Field(..., title="Image ID")
    user_id: str = Field(..., title="User ID")
    focussed_contour_id: int | None = Field(default=None, title="Contour ID")
    refinement_contour_id: int | None = Field(default=None, title="Contour ID")
    _running_backends: dict[str, BaseService] = PrivateAttr(
        default_factory=dict  # This should not be serialized
    )
    _failed_backends: dict[str, BaseService] = PrivateAttr(
        default_factory=dict  # This should not be serialized
    )

    @cached_property
    def image_db(self) -> Images:
        with get_context_session() as session:
            image_db = session.query(Images).filter_by(id=self.image_id).one()
        return image_db

    @field_validator("image_id", mode="before")
    def validate_image_id(cls, value, values):
        with get_context_session() as session:
            if session.query(Images).filter_by(id=value).one() is None:
                raise ValidationError(f"Image ID {value} does not exist.")
        return value

    async def mask_id(self):
        """ Validate the model and fill fields that were not initialized yet."""
        with get_context_session() as session:
            if session.query(Masks).filter_by(image_id=self.image_id).first() is None:
                await create_new_mask_for_image(self.image_id)
            return session.query(Masks.id).filter_by(image_id=self.image_id).first().id

    async def upload_image(self):
        for key, service in self._running_backends.items():
            response = await service.upload_image(self.user_id, self.image_id)
            if not response["success"]:
                self._running_backends.pop(key, None)
                self._failed_backends[key] = service

    async def check_and_register_backend(self, service: BaseService, key):
        if not await service.check_backend():
            logger.error(f"{key} is not reachable. Please make sure it is running.")
            self._failed_backends[key] = service
        else:
            logger.debug(f"{key} is reachable.")
            self._running_backends[key] = service

    async def focus_contour(self, contour_id: int):
        self.focussed_contour_id = contour_id
        successful = []
        unsuccessful = []
        for key, service in self._running_backends.items():
            response = await service.focus_contour(self.user_id, self.focussed_contour_id)
            if not response["success"]:
                logger.error(f"{key} ran into an error. Focussing might not have work.")
                unsuccessful.append(key)
            else:
                successful.append(key)
        return successful, unsuccessful

    async def unfocus_contour(self):
        self.focussed_contour_id = None
        successful = []
        unsuccessful = []
        for key, service in self._running_backends.items():
            response = await service.unfocus_crop(self.user_id)
            if not response["success"]:
                logger.error(f"{key} ran into an error. Unfocussing might not have worked.")
                unsuccessful.append(key)
            else:
                successful.append(key)
        return successful, unsuccessful


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


async def on_startup(state: AnnotationSessionState) -> ServerMessage:
    """Function to be called at the start of an annotation session. Any initialization code can be placed here.
    """
    # Check for running backends
    # await state.check_and_register_backend(S, "semantic_service")
    await state.check_and_register_backend(PromptedSegmentationService(), Backends.PROMPTED_SEGMENTATION.value)
    await state.check_and_register_backend(CompletionService(), Backends.COMPLETION_SEGMENTATION.value)
    await state.check_and_register_backend(SemanticSegmentationService(), Backends.SEMANTIC_SEGMENTATION.value)

    # Upload the image to all running backends
    await state.upload_image()

    logger.info("Annotation session initialized.")
    with get_context_session() as session:
        contours_response = await get_contours_of_mask(await state.mask_id(),
                                                       flattened=False,
                                                       db=session)
        objects = contours_response.get("contours", [])
    return ServerMessage(
        id="test",
        type=ServerMessageType.SESSION_INITIALIZED,
        success=len(state._failed_backends) == 0,
        message=f"Annotation session initialized."
                f"\nRunning backends: {list(state._running_backends.keys())}"
                f"\nFailed initializations: {list(state._failed_backends.keys())}",
        data={
            "running": list(state._running_backends.keys()),
            "failed": list(state._failed_backends.keys()),
            "objects": objects,
        }
    )


@router.websocket("/ws/user={user_id}&image={image_id}")
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
    print(f"WebSocket connection accepted for user {user_id} and image {image_id}")
    state = AnnotationSessionState(
        image_id=image_id,
        user_id=user_id,
    )
    try:
        # Call some functions on startup
        print(f"Calling on startup for user {user_id} and image {image_id}")
        response = await on_startup(state)
        print(f"Startup response: {response.message}")
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
                case ClientMessageType.SELECT_REFINEMENT_OBJECT:
                    await handle_select_refinement_object(websocket, client_msg, state)
                case ClientMessageType.UNSELECT_REFINEMENT_OBJECT:
                    await handle_unselect_refinement_object(websocket, client_msg, state)
                case ClientMessageType.OBJECT_ADD_MANUAL:
                    await handle_object_add(websocket, client_msg, state)
                case ClientMessageType.OBJECT_FINALISE:
                    await handle_object_finalise(websocket, client_msg, state)
                case ClientMessageType.OBJECT_DELETE:
                    await handle_object_delete(websocket, client_msg, state)
                case ClientMessageType.OBJECT_MODIFY:
                    await handle_object_modify(websocket, client_msg, state)
                case ClientMessageType.SEMANTIC_SELECT_MODEL:
                    await handle_semantic_select_model(websocket, client_msg, state)
                case ClientMessageType.SEMANTIC_SEGMENTATION:
                    await handle_semantic_segmentation(websocket, client_msg, state)
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
                case ClientMessageType.COMPLETION_INFERENCE:
                    await handle_completion(websocket, client_msg, state)
                case ClientMessageType.FINISH_ANNOTATION:
                    await handle_finish_annotation(websocket, client_msg, state)
                case ClientMessageType.OBJECT_CONFLICT_RESOLUTION:
                    await handle_object_conflict_resolve(websocket, client_msg, state)
                case _:
                    # Ignore erroneous messages from the client
                    pass
    except WebSocketDisconnect:
        # Client disconnected normally, just log and exit
        logger.info(f"WebSocket disconnected for user {user_id} and image {image_id}")
        print(f"WebSocket disconnected for user {user_id} and image {image_id}")
    except Exception as e:
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


async def handle_focus_image(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle the client sending a focus image request"""
    focussed_contour_id = client_msg.data.get("focussed_contour_id")
    successful, unsuccessful = await state.focus_contour(focussed_contour_id)
    if len(unsuccessful) == 0:
        message_type = ServerMessageType.SUCCESS
        message = "All services focussed!"
    elif len(successful) == 0:
        message_type = ServerMessageType.ERROR
        message = f"Failed to focus any service!"
    else:
        message_type = ServerMessageType.WARNING
        message = f"Failed to focus some services! Failed services: {unsuccessful}"
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=message_type,
        success=len(unsuccessful) == 0,
        message=message,
        data=None
    ))


async def handle_unfocus_image(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle the client unfocussing."""
    successful, unsuccessful = await state.unfocus_contour()
    if len(unsuccessful) == 0:
        message_type = ServerMessageType.SUCCESS
        message = "All services unfocussed!"
    elif len(successful) == 0:
        message_type = ServerMessageType.ERROR
        message = f"Failed to unfocus any service!"
    else:
        message_type = ServerMessageType.WARNING
        message = f"Failed to unfocus some services! Failed services: {unsuccessful}"
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=message_type,
        success=len(unsuccessful) == 0,
        message=message,
        data=None
    ))


async def handle_select_refinement_object(websocket: WebSocket, client_msg: ClientMessage,
                                          state: AnnotationSessionState):
    """ Handle the client selecting an object for refinement."""
    refinement_contour_id = client_msg.data.get("contour_id")
    state.refinement_contour_id = refinement_contour_id
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.SUCCESS,
        success=True,
        message=f"Selected contour {refinement_contour_id} for refinement.",
        data=None
    ))


async def handle_unselect_refinement_object(websocket: WebSocket, client_msg: ClientMessage,
                                            state: AnnotationSessionState):
    """ Handle the client unselecting an object for refinement."""
    state.refinement_contour_id = None
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.SUCCESS,
        success=True,
        message=f"Unselected contour for refinement.",
        data=None
    ))


async def handle_object_add(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle adding an object to the mask."""
    contour = Contour.model_validate_json(client_msg.data)
    with get_context_session() as session:
        response = await add_contour(await state.mask_id(), contour, db=session)
    if not response["success"]:
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.ERROR,
            message=response["message"],
            success=False,
            data=None,
        ))
    else:
        with get_context_session() as session:
            contours_response = await get_contours_of_mask(await state.mask_id(), db=session)
            updated_objects = contours_response.get("contours", [])
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
    with get_context_session() as session:
        response = await add_contour_review(contour_id, db=session)
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.OBJECT_MODIFIED if response["success"] else ServerMessageType.ERROR,
        message=response["message"],
        success=response["success"],
        data={
            "contour_id": contour_id,
            "reviewed_by": state.user_id,
        }
    ))


async def handle_object_delete(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle removing an object from the mask. """
    contour_id = client_msg.data.get("contour_id")
    with get_context_session() as session:
        response = await delete_contour(contour_id, db=session)
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.OBJECT_REMOVED if response["success"] else ServerMessageType.ERROR,
        success=response["success"],
        message=response["message"],
        data=None,
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

    #  actual authenticated username
    if "reviewed_by" in fields_to_be_updated and fields_to_be_updated["reviewed_by"]:
        fields_to_be_updated["reviewed_by"] = [
            state.user_id if username == "current_user" else username
            for username in fields_to_be_updated["reviewed_by"]
        ]

    if fields_to_be_updated:
        with get_context_session() as session:
            response = await modify_contour(contour_id, db=session, **fields_to_be_updated)
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


async def handle_semantic_select_model(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle the selection of an automatic model. """
    selected_model = client_msg.data.get("model_registry_key")
    response = await state._running_backends[Backends.SEMANTIC_SEGMENTATION.value].select_model(state.user_id,
                                                                                                selected_model)
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.SUCCESS if response["success"] else ServerMessageType.ERROR,
        success=response["success"],
        message=response["message"],
        data=None
    ))


async def handle_semantic_segmentation(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle prompted_segmentation using an automatic model. """
    inference_req = SemanticSegmentationRequest.model_validate(client_msg.data)
    response = await state._running_backends[Backends.SEMANTIC_SEGMENTATION.value].inference(inference_req)
    contour_hierarchy = ContourHierarchy.model_validate(response["result"])
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.SUCCESS if response["success"] else ServerMessageType.ERROR,
        success=response["success"],
        message=response["message"],
        data=contour_hierarchy
    ))
    for root_contours in contour_hierarchy.root_contours:
        with get_context_session() as session:
            save_contour_tree(session, root_contours, await state.mask_id())



async def handle_prompted_select_model(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle the selection of a prompted model. """
    selected_model = client_msg.data.get("selected_model")
    response = await state._running_backends[Backends.PROMPTED_SEGMENTATION.value].select_model(state.user_id,
                                                                                                selected_model)
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.SUCCESS if response["success"] else ServerMessageType.ERROR,
        success=response["success"],
        message=response["message"],
        data=None
    ))


async def handle_prompted_segmentation(
        websocket: WebSocket,
        client_msg: ClientMessage,
        state: AnnotationSessionState,
        override_completion_disable=False,
):
    """ Handle prompted_segmentation using a prompted model. """
    model_identifier = client_msg.data.get("model_key")
    prompts_data = client_msg.data.get("prompts")
    prompts_model = Prompts.model_validate(prompts_data)
    using_refinement = state.refinement_contour_id is not None
    if using_refinement:
        # Get the contour to refine
        with get_context_session() as session:
            contour = session.query(Contours).filter_by(id=state.refinement_contour_id).first()
            contour_model = Contour.from_db(contour)
        # The height and width are hardcoded here, which is not the preferred solution
        # It should resize to the og image shape, but for now this works.
        # SAM accepts previous masks in this format anyway.
        previous_mask = contour_model.to_binary_mask_model(250, 250)
        logger.debug(f"Using contour {state.refinement_contour_id} as previous mask for refinement.")
    else:
        previous_mask = None

    prompted_request = PromptedSegmentationRequest(
        user_id=str(state.user_id),
        image_url=state.image_db.file_path,
        model_registry_key=model_identifier,
        previous_mask=previous_mask,
        prompts=prompts_model,
    )

    response_seg = await state._running_backends[Backends.PROMPTED_SEGMENTATION.value].inference(prompted_request)
    contour_model = Contour.model_validate(response_seg["result"])

    # Compute SVG path before sending
    contour_model.compute_path(
        image_width=state.image_db.width,
        image_height=state.image_db.height
    )

    if using_refinement:
        await replace_object(state.refinement_contour_id, contour_model, websocket, client_msg, state)
    else:
        await add_object(contour_model, websocket, client_msg, state)


async def handle_completion_select_model(websocket: WebSocket, client_msg: ClientMessage,
                                         state: AnnotationSessionState):
    """ Handle the selection of a completion model. """
    if Backends.COMPLETION_SEGMENTATION.value in state._running_backends:
        model_identifier = client_msg.data.get("model_identifier")
        await state._running_backends[Backends.COMPLETION_SEGMENTATION.value].select_model(state.user_id,
                                                                                           model_identifier)
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.SUCCESS,
            success=True,
            message="Enabled annotation completion",
            data=None
        ))
    else:
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.ERROR,
            success=False,
            message="Failed to enable annotation completion. Backend is not running.",
            data=None
        ))


async def handle_completion_enable(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle enabling of completion model. Leads to a state change. """
    if Backends.COMPLETION_SEGMENTATION.value in state._running_backends:
        state._running_backends[Backends.COMPLETION_SEGMENTATION.value].enable()
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.SUCCESS,
            success=True,
            message="Enabled annotation completion",
            data=None
        ))
    else:
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.ERROR,
            success=False,
            message="Failed to enable annotation completion. Backend is not running.",
            data=None
        ))


async def handle_completion_disable(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle disabling of completion model. Leads to a state change. """
    if Backends.COMPLETION_SEGMENTATION.value in state._running_backends:
        state._running_backends[Backends.COMPLETION_SEGMENTATION.value].disable()
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.SUCCESS,
            success=True,
            message="Disabled annotation completion",
            data=None
        ))
    else:
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.ERROR,
            success=False,
            message="Failed to disable annotation completion. Backend is not running.",
            data=None
        ))


async def handle_completion(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle the completion of a completion model. """
    seed_contour_ids = client_msg.data.get("seed_contour_ids")
    with get_context_session() as session:
        contours_db = session.query(Contours).filter(Contours.id.in_(seed_contour_ids)).all()
        contours = [Contour.from_db(contour_db) for contour_db in contours_db]
    height, width = get_height_width_of_image(state.image_id)
    positive_exemplars = [contour.to_binary_mask_model(height, width) for contour in contours]

    # Find out the concept
    contour_labels = {contour.label_id for contour in contours if
                      contour.label_id is not None}  # Creates a set of contours
    label_id = contour_labels.pop() if len(
        contour_labels) == 1 else None  # If only one label is present we take it as a concept, otherwise we ignore it
    if label_id is not None:
        with get_context_session() as session:
            label_db = session.query(Labels).filter_by(id=label_id).one_or_none()
            concept = Label.from_db(label_db)
    else: concept = None
    service_request = CompletionServiceRequest(
        image_url=state.image_db.file_path,
        model_registry_key=client_msg.data.get('model_key'),
        user_id=state.user_id,
        positive_exemplars=positive_exemplars,
        negative_exemplars=None,
        concept=concept,
    )
    response = await state._running_backends[Backends.COMPLETION_SEGMENTATION.value].inference(service_request)
    await send_msg(websocket, ServerMessage(
        success=response["success"],
        id=client_msg.id,
        type=ServerMessageType.SUCCESS,
        message=response["message"],
        data=None
    ))
    response_data = response["result"]
    for contour_json in response_data:
        contour = Contour.model_validate(contour_json)
        await add_object(contour, websocket, client_msg, state)


async def add_object(object_to_add: Contour, websocket: WebSocket, client_msg: ClientMessage,
                     state: AnnotationSessionState):
    with get_context_session() as session:
        response = await add_contour(
            mask_id=await state.mask_id(),
            contour_to_add=object_to_add,
            db=session,
        )
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.OBJECT_ADDED if response["success"] else ServerMessageType.ERROR,
        success=response["success"],
        message=f"Successfully added object with confidence score {object_to_add.confidence:.1%}" if response[
            "success"] else response["message"],
        data=object_to_add.model_dump() if response["success"] else None,
    ))
    return response


async def replace_object(old_object_id, new_object: Contour, websocket: WebSocket, client_msg: ClientMessage,
                         state: AnnotationSessionState):
    with get_context_session() as session:
        response = await replace_contour(
            contour_id=old_object_id,
            new_contour=new_object,
            db=session,
        )
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.OBJECT_MODIFIED if response["success"] else ServerMessageType.ERROR,
        success=response["success"],
        message=f"Successfully modified object." if response[
            "success"] else response["message"],
        data=new_object.model_dump() if response["success"] else None,
    ))
    return response


async def handle_finish_annotation(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle marking a mask as finished. """
    with get_context_session() as session:
        response = await mark_as_fully_annotated(await state.mask_id(), db=session)
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.SUCCESS if response["success"] else ServerMessageType.ERROR,
        success=response["success"],
        message=response["message"],
        data=None
    ))


async def handle_object_conflict_resolve(websocket: WebSocket, client_msg: ClientMessage,
                                         state: AnnotationSessionState):
    """ Handle how an object conflict should be resolved. """
    raise NotImplementedError("Method not implemented yet!")

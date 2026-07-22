"""WebSocket message handlers for the annotation session.

Each handler is a thin adapter: it parses ``client_msg.data``, resolves session context
from ``state``, delegates the actual work to the reusable operations in
``app.services.annotation_session.operations`` (for AI inference) or to the ``*_db``
service modules (for persistence), and sends a ``ServerMessage`` back to the client.
"""

from logging import getLogger

from fastapi.websockets import WebSocket
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.websockets.annotation_session import (
    ServerMessageType,
    ServerMessage,
    ClientMessage,
)
from iquana_toolbox.schemas.prompts import Prompts

from app.database import get_context_session
from app.database.contours import Contours
from app.database.masks import Masks
from app.routes.websockets.messaging import send_msg
from app.services.ai_services.instance_suggestion import SuggestionService
from app.services.ai_services.instance_segmentation import InstanceSegmentationService
from app.services.ai_services.prompted_segmentation import PromptedSegmentationService
from app.services.annotation_session.operations import (
    assign_hierarchy_parents,
    filter_exemplar_overlaps,
    run_suggestion_segmentation,
    run_instance_segmentation,
    run_prompted_segmentation,
)
from app.schemas.permissions import Permission
from app.services.annotation_session.state import AnnotationSessionState, Backends
from app.services.auth import load_user
from app.services.database_access import contours as contours_db
from app.services.database_access import labels as labels_db
from app.services.database_access import masks as masks_db

logger = getLogger(__name__)


def session_user(state: AnnotationSessionState, db):
    """Reload the session's authenticated user, with their current permissions.

    Permissions are re-read per message rather than cached on the session, so
    revoking someone's access takes effect on their next action instead of only
    when they next reconnect.
    """
    return load_user(state.user_id, db)


async def _deny(websocket: WebSocket, client_msg: ClientMessage, permission: Permission):
    """Refuse one action without tearing down the session."""
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.ERROR,
        success=False,
        message=f"You do not have permission to do this ({permission.value}).",
        data=None,
    ))


async def startup(websocket: WebSocket, state: AnnotationSessionState):
    """Function to be called at the start of an annotation session. Any initialization code can be placed here.
    """
    print(f"Annotation session initialized: {state.model_dump()}")
    # Check for running backends
    await state.check_and_register_backend(PromptedSegmentationService(), Backends.PROMPTED_SEGMENTATION.value)
    await state.check_and_register_backend(SuggestionService(), Backends.SUGGESTION_SEGMENTATION.value)
    await state.check_and_register_backend(InstanceSegmentationService(), Backends.INSTANCE_SEGMENTATION.value)

    with get_context_session() as db:
        mask_db = db.query(Masks).filter_by(id=state.mask_id).first()
        mask_status = mask_db.status if mask_db else None

    await send_msg(
        websocket,
        ServerMessage(
            id="0",
            type=ServerMessageType.SESSION_INITIALIZED,
            success=len(state._failed_backends) == 0,
            message=f"Annotation session initialized."
                    f"\nRunning backends: {list(state._running_backends.keys())}"
                    f"\nFailed initializations: {list(state._failed_backends.keys())}",
            data={
                "running": list(state._running_backends.keys()),
                "failed": list(state._failed_backends.keys()),
                "mask_id": state.mask_id,
                "mask_status": mask_status,
            }
        )
    )

    logger.info("Annotation session initialized.")
    with get_context_session() as db:
        hierarchy = await masks_db.get_contour_hierarchy_of_mask(state.mask_id, db)
    state.contour_hierarchy = hierarchy
    await send_msg(
        websocket,
        ServerMessage(
            id="1",
            type=ServerMessageType.OBJECTS,
            success=True,
            message=f"Retrieved annotations",
            data=hierarchy.model_dump()
        )
    )


async def handle_focus_image(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle the client sending a focus image request"""
    focussed_contour_id = client_msg.data.get("focussed_contour_id")
    successful = await state.focus_contour(focussed_contour_id)
    if successful:
        message_type = ServerMessageType.SUCCESS
        message = "All services focussed!"
    else:
        message_type = ServerMessageType.ERROR
        message = f"Failed to focus any service!"
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=message_type,
        success=successful,
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
    """ Handle adding a manually drawn object to the mask."""
    # client_msg.data is already a parsed dict (the WebSocket layer decodes the
    # JSON envelope), so validate it as an object — not as a JSON string.
    contour = Contour.model_validate(client_msg.data)
    # Persist via the shared helper, exactly like the AI add flow: it inserts the
    # contour into the DB first (which assigns the real integer id and computes
    # its SVG path) and then broadcasts the single new object.
    #
    # The previous implementation broadcast the id-less session hierarchy *before*
    # the DB insert, which made the client invent a bogus (decimal) id for the new
    # object and rebuild from the full hierarchy — blanking every other object's
    # label until the next page reload.
    await add_object(contour, websocket, client_msg, state)


async def handle_object_finalise(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Turn a temporary object into a non temporary one. For example: Temporary objects are added by AI models, if you
        make them non temporary, they will be added to the mask and can be used for training.
    """
    contour_id = client_msg.data.get("contour_id")
    with get_context_session() as db:
        # This used to call get_current_user() outside FastAPI's dependency
        # injection, which handed it Depends() sentinels instead of a token and a
        # session. Resolve the session's user directly instead.
        user = session_user(state, db)
        if user is None or not user.has_permission(state.dataset_id, Permission.REVIEW_APPROVE):
            await _deny(websocket, client_msg, Permission.REVIEW_APPROVE)
            return
        try:
            reviewed = await contours_db.review_contour(contour_id, user=user, db=db, strict=True)
            message = "Object finalised." if reviewed else "Object could not be finalised."
        except (PermissionError, KeyError) as exc:
            reviewed, message = False, str(exc)
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.OBJECT_MODIFIED if reviewed else ServerMessageType.ERROR,
        message=message,
        success=reviewed,
        data={
            "contour_id": contour_id,
            "reviewed_by": state.user_id,
        }
    ))


async def handle_object_delete(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle removing an object from the mask. """
    contour_id = client_msg.data.get("contour_id")
    with get_context_session() as db:
        response = await contours_db.delete_contour(contour_id, db)
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.OBJECT_REMOVED,
        success=True,
        message="Object removed from mask.",
        data={"deleted_contours": [contour_id]},
    ))


async def handle_object_modify(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle Modifying an object. Supports updating label_id and reviewed_by via WebSocket.
        label_id changes are validated against the dataset's label hierarchy.
    """
    contour_id = client_msg.data.get("contour_id")
    fields_to_be_updated = dict(client_msg.data.get("fields_to_be_updated") or {})

    # `reviewed_by` is never taken from the client: accepting it let a client mark
    # a contour as reviewed by arbitrary users. Approvals go through
    # review_contour(), which enforces the dataset's review policy.
    wants_review = bool(fields_to_be_updated.pop("reviewed_by", None))
    fields_to_be_updated.pop("author_username", None)
    assigns_label = "label_id" in fields_to_be_updated

    with get_context_session() as db:
        user = session_user(state, db)
        if user is None or not user.has_permission(state.dataset_id, Permission.ANNOTATION_EDIT_OWN):
            await _deny(websocket, client_msg, Permission.ANNOTATION_EDIT_OWN)
            return

        existing = db.query(Contours).filter_by(id=contour_id).first()
        authored_by_someone_else = (existing is not None
                                    and existing.author_username not in (None, state.user_id))
        if authored_by_someone_else and not user.has_permission(state.dataset_id,
                                                                Permission.ANNOTATION_EDIT_ANY):
            await _deny(websocket, client_msg, Permission.ANNOTATION_EDIT_ANY)
            return

        if fields_to_be_updated:
            await contours_db.modify_contour(contour_id, db=db, **fields_to_be_updated)

        # Assigning a label counts as a review only for callers entitled to give
        # one; for everyone else the label change simply stands on its own.
        if (wants_review or assigns_label) and user.has_permission(state.dataset_id,
                                                                   Permission.REVIEW_APPROVE):
            await contours_db.review_contour(contour_id, user=user, db=db, strict=False)

    if fields_to_be_updated or wants_review:
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.OBJECT_MODIFIED,
            message="Modified object",
            success=True,
            data={
                "contour_id": contour_id,
                "fields_to_be_updated": fields_to_be_updated,
            },
        ))


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
        override_suggestion_disable=False,
):
    """ Handle prompted_segmentation using a prompted model. """
    model_identifier = client_msg.data.get("model_key")
    prompts_data = client_msg.data.get("prompts")
    prompts_model = Prompts.model_validate(prompts_data)
    using_refinement = state.refinement_contour_id is not None
    if using_refinement:
        # Get the contour to refine
        with get_context_session() as db:
            contour_model = await contours_db.get_contour(contour_id=state.refinement_contour_id, db=db)
        # The height and width are hardcoded here, which is not the preferred solution
        # It should resize to the og image shape, but for now this works.
        # SAM accepts previous masks in this format anyway.
        previous_mask = contour_model.to_binary_mask_model(250, 250)
        logger.debug(f"Using contour {state.refinement_contour_id} as previous mask for refinement.")
    else:
        previous_mask = None

    # Nested segmentation is currently ignored!

    # When annotating inside a focussed object (e.g. patches inside a football), the model
    # may return the parent itself as a candidate. Pass the focussed contour so the
    # operation can discard candidates that just re-segment it.
    focus_contour = None
    if state.focussed_contour_id is not None and state.contour_hierarchy is not None:
        focus_contour = state.contour_hierarchy.id_to_contour.get(state.focussed_contour_id)

    result = await run_prompted_segmentation(
        service=state._running_backends[Backends.PROMPTED_SEGMENTATION.value],
        image_url=state.image_db.file_path,
        image_width=state.image_db.width,
        image_height=state.image_db.height,
        model_key=model_identifier,
        prompts=prompts_model,
        user_id=state.user_id,
        previous_mask=previous_mask,
        parent_id=state.focussed_contour_id,
        focus_contour=focus_contour,
    )
    contour_model = result.contour

    if contour_model is None:
        # Every candidate duplicated the focussed object -> nothing new to add.
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.SUCCESS,
            success=True,
            message="No new object added: all candidates duplicated the focussed object.",
            data={"candidate_count": len(result.candidates)},
        ))
        return

    if using_refinement:
        # Make sure the label stays after refinement
        old_contour = state.contour_hierarchy.id_to_contour[state.refinement_contour_id]
        if old_contour.label_id is not None:
            contour_model.label_id = old_contour.label_id

        # Replace it in our session state
        state.contour_hierarchy.id_to_contour[state.refinement_contour_id] = contour_model

        # Replace in the db
        await replace_object(state.refinement_contour_id, contour_model, websocket, client_msg, state)
    else:
        await add_object(contour_model, websocket, client_msg, state)


async def handle_suggestion_select_model(websocket: WebSocket, client_msg: ClientMessage,
                                         state: AnnotationSessionState):
    """ Handle the selection of a suggestion model. """
    if Backends.SUGGESTION_SEGMENTATION.value in state._running_backends:
        model_identifier = client_msg.data.get("model_identifier")
        response = await state._running_backends[Backends.SUGGESTION_SEGMENTATION.value].select_model(state.user_id,
                                                                                           model_identifier)
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.SUCCESS if response["success"] else ServerMessageType.ERROR,
            success=response["success"],
            message=response["message"],
            data=None
        ))
    else:
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.ERROR,
            success=False,
            message="Failed to enable annotation suggestion. Backend is not running.",
            data=None
        ))


async def handle_suggestion_enable(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle enabling of suggestion model. Leads to a state change. """
    if Backends.SUGGESTION_SEGMENTATION.value in state._running_backends:
        state._running_backends[Backends.SUGGESTION_SEGMENTATION.value].enable()
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.SUCCESS,
            success=True,
            message="Enabled annotation suggestion",
            data=None
        ))
    else:
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.ERROR,
            success=False,
            message="Failed to enable annotation suggestion. Backend is not running.",
            data=None
        ))


async def handle_suggestion_disable(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle disabling of suggestion model. Leads to a state change. """
    if Backends.SUGGESTION_SEGMENTATION.value in state._running_backends:
        state._running_backends[Backends.SUGGESTION_SEGMENTATION.value].disable()
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.SUCCESS,
            success=True,
            message="Disabled annotation suggestion",
            data=None
        ))
    else:
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.ERROR,
            success=False,
            message="Failed to disable annotation suggestion. Backend is not running.",
            data=None
        ))


async def handle_suggestion(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle the suggestion of a suggestion model. """
    seed_contour_ids = client_msg.data.get("seed_contour_ids")
    with get_context_session() as db:
        contours = await contours_db.get_contours(seed_contour_ids, db)
    height, width = state.image_db.height, state.image_db.width
    positive_exemplars = [contour.to_binary_mask_model(height, width) for contour in contours]

    # Find out the concept
    contour_labels = {contour.label_id for contour in contours if
                      contour.label_id is not None}  # Creates a set of contours
    label_id = contour_labels.pop() if len(
        contour_labels) == 1 else None  # If only one label is present we take it as a concept, otherwise we ignore it
    if label_id is not None:
        with get_context_session() as db:
            concept = await labels_db.get_label(label_id, db)
    else:
        concept = None

    result = await run_suggestion_segmentation(
        service=state._running_backends[Backends.SUGGESTION_SEGMENTATION.value],
        image_url=state.image_db.file_path,
        model_key=client_msg.data.get('model_key'),
        user_id=state.user_id,
        positive_exemplars=positive_exemplars,
        concept=concept,
    )

    # Instance suggestion may re-detect the seed exemplars themselves; drop those.
    suggested = filter_exemplar_overlaps(result.contours, contours)

    # Place the suggested instances in the hierarchy: tag them with the concept label and
    # nest each one under the existing contour (of the correct parent label) that contains
    # it. Contours without a valid parent stay at root level.
    with get_context_session() as db:
        hierarchy = await masks_db.get_contour_hierarchy_of_mask(state.mask_id, db)
        label_hierarchy = await labels_db.get_label_hierarchy(state.image_db.dataset_id, db)
    suggested = assign_hierarchy_parents(suggested, hierarchy, label_hierarchy, label_id)

    # Report how many new instances were found so the client can tell the user
    # when a model returned nothing (objects themselves follow as OBJECT_ADDED).
    await send_msg(websocket, ServerMessage(
        success=result.success,
        id=client_msg.id,
        type=ServerMessageType.SUCCESS,
        message=result.message,
        data={"added_count": len(suggested)},
    ))
    for contour in suggested:
        await add_object(contour, websocket, client_msg, state)


async def handle_instance_select_model(websocket: WebSocket, client_msg: ClientMessage,
                                       state: AnnotationSessionState):
    """ Handle the selection of an instance segmentation model. """
    if Backends.INSTANCE_SEGMENTATION.value in state._running_backends:
        selected_model = client_msg.data.get("selected_model")
        response = await state._running_backends[Backends.INSTANCE_SEGMENTATION.value].select_model(state.user_id,
                                                                                                   selected_model)
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.SUCCESS if response["success"] else ServerMessageType.ERROR,
            success=response["success"],
            message=response["message"],
            data=None
        ))
    else:
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.ERROR,
            success=False,
            message="Failed to select instance segmentation model. Backend is not running.",
            data=None
        ))


async def handle_instance_segmentation(websocket: WebSocket, client_msg: ClientMessage,
                                       state: AnnotationSessionState):
    """ Handle instance segmentation inference.

        Instance segmentation re-segments the whole image, so the detected instances
        replace every contour currently on the mask (the client warns the user about
        this before requesting it).
    """
    if Backends.INSTANCE_SEGMENTATION.value not in state._running_backends:
        await send_msg(websocket, ServerMessage(
            id=client_msg.id,
            type=ServerMessageType.ERROR,
            success=False,
            message="Failed to run instance segmentation. Backend is not running.",
            data=None
        ))
        return

    model_registry_key = client_msg.data.get("model_registry_key")
    result = await run_instance_segmentation(
        service=state._running_backends[Backends.INSTANCE_SEGMENTATION.value],
        image_url=state.image_db.file_path,
        image_width=state.image_db.width,
        image_height=state.image_db.height,
        model_registry_key=model_registry_key,
        user_id=state.user_id,
    )

    # Replace the existing contours with the freshly detected instances.
    with get_context_session() as db:
        await masks_db.delete_all_contours_of_mask(state.mask_id, db=db)
        for contour in result.contours:
            await masks_db.add_contour_to_mask(state.mask_id, contour, db=db)
        hierarchy = await masks_db.get_contour_hierarchy_of_mask(state.mask_id, db)
    state.contour_hierarchy = hierarchy

    # Send the full hierarchy so the client refreshes its object list in one go.
    # Use OBJECTS (not OBJECT_ADDED): the client resolves label names against the
    # dataset's label map on the OBJECTS path, so the detected instances keep their
    # labels. The matching message id also resolves the caller's pending request.
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.OBJECTS,
        success=result.success,
        message=result.message or f"Instance segmentation detected {len(result.contours)} objects.",
        data=hierarchy.model_dump(),
    ))


async def add_object(object_to_add: Contour, websocket: WebSocket, client_msg: ClientMessage,
                     state: AnnotationSessionState):
    with get_context_session() as db:
        response = await masks_db.add_contour_to_mask(
            mask_id=state.mask_id,
            contour_to_add=object_to_add,
            db=db,
            author_username=state.user_id,
        )
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.OBJECT_ADDED,
        success=True,
        message=f"Successfully added object with confidence score {object_to_add.confidence:.1%}",
        data=response,
    ))
    return response


async def replace_object(old_object_id, new_object: Contour, websocket: WebSocket, client_msg: ClientMessage,
                         state: AnnotationSessionState):
    with get_context_session() as db:
        success = await contours_db.replace_contour(old_object_id, new_object, db,
                                                   author_username=state.user_id)
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.OBJECT_MODIFIED if success else ServerMessageType.ERROR,
        success=success,
        message="Successfully modified object." if success else f"Failed to replace contour {old_object_id}.",
        data=new_object.model_dump() if success else None,
    ))
    return success


async def handle_finish_annotation(websocket: WebSocket, client_msg: ClientMessage, state: AnnotationSessionState):
    """ Handle submitting a mask for review, i.e. marking it fully annotated. """
    with get_context_session() as db:
        user = session_user(state, db)
        if user is None or not user.has_permission(state.dataset_id, Permission.MASK_SUBMIT):
            await _deny(websocket, client_msg, Permission.MASK_SUBMIT)
            return
        # mark_mask_as_complete returns None; the old code subscripted it as a dict
        # and raised a TypeError on every finish.
        await masks_db.mark_mask_as_complete(state.mask_id, db)
        status = db.query(Masks).filter_by(id=state.mask_id).one().status
    await send_msg(websocket, ServerMessage(
        id=client_msg.id,
        type=ServerMessageType.SUCCESS,
        success=True,
        message="Mask submitted for review.",
        data={"mask_id": state.mask_id, "status": status},
    ))


async def handle_object_conflict_resolve(websocket: WebSocket, client_msg: ClientMessage,
                                         state: AnnotationSessionState):
    """ Handle how an object conflict should be resolved. """
    raise NotImplementedError("Method not implemented yet!")

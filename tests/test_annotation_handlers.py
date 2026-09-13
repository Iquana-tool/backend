"""Focused tests for interactive instance-segmentation write modes."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

from iquana_toolbox.schemas.database.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.labels import LabelHierarchy
from iquana_toolbox.schemas.networking.websockets.annotation_session import ClientMessage

from app.routes.websockets import annotation_handlers as handlers
from app.services.annotation_session.operations import InstanceSegmentationResult
from app.services.annotation_session.state import Backends


def box(x0, y0, x1, y1, *, contour_id=None, confidence=1.0):
    return Contour(
        id=contour_id,
        x=[x0, x1, x1, x0],
        y=[y0, y0, y1, y1],
        confidence=confidence,
        added_by="model",
    )


def _setup(monkeypatch, *, write_mode=None, predictions=(), hierarchies=()):
    state = SimpleNamespace(
        _running_backends={Backends.INSTANCE_SEGMENTATION.value: object()},
        image_db=SimpleNamespace(file_path="image.png", width=100, height=100),
        user_id="alice",
        mask_id=7,
        contour_hierarchy=None,
    )
    data = {"model_registry_key": "model"}
    if write_mode is not None:
        data["write_mode"] = write_mode
    client_msg = ClientMessage(id="request-1", type="instance_inference", data=data)
    sent = []

    async def send(_websocket, message):
        sent.append(message)

    monkeypatch.setattr(handlers, "send_msg", send)
    monkeypatch.setattr(
        handlers,
        "run_instance_segmentation",
        AsyncMock(return_value=InstanceSegmentationResult(
            contours=list(predictions), success=True, message="model finished"
        )),
    )

    db = object()

    @contextmanager
    def context_session():
        yield db

    monkeypatch.setattr(handlers, "get_context_session", context_session)

    delete = AsyncMock()
    add = AsyncMock()
    hierarchy_queue = list(hierarchies)

    async def get_hierarchy(_mask_id, _db):
        return hierarchy_queue.pop(0)

    async def get_cached_hierarchy(_mask_id, _db):
        hierarchy = hierarchy_queue.pop(0)
        return hierarchy, hierarchy.model_dump()

    monkeypatch.setattr(handlers.masks_db, "delete_all_contours_of_mask", delete)
    monkeypatch.setattr(handlers.masks_db, "add_contour_to_mask", add)
    monkeypatch.setattr(handlers.masks_db, "get_contour_hierarchy_of_mask", get_hierarchy)
    monkeypatch.setattr(handlers.masks_db, "get_cached_contour_hierarchy_of_mask", get_cached_hierarchy)

    return state, client_msg, sent, delete, add


def test_patch_preserves_existing_and_suppresses_overlaps(monkeypatch):
    existing = box(0.0, 0.0, 0.3, 0.3, contour_id=1)
    duplicate_existing = box(0.0, 0.0, 0.3, 0.3, confidence=0.9)
    kept = box(0.5, 0.5, 0.7, 0.7, confidence=0.8)
    duplicate_prediction = box(0.5, 0.5, 0.7, 0.7, confidence=0.1)
    existing_hierarchy = ContourHierarchy(
        root_contours=[existing], id_to_contour={1: existing}, label_id_to_contours={None: [existing]}
    )
    final_hierarchy = ContourHierarchy(root_contours=[existing, kept])
    state, message, sent, delete, add = _setup(
        monkeypatch,
        write_mode="patch",
        predictions=[duplicate_existing, kept, duplicate_prediction],
        hierarchies=[existing_hierarchy, final_hierarchy],
    )

    asyncio.run(handlers.handle_instance_segmentation(object(), message, state))

    delete.assert_not_awaited()
    assert add.await_count == 1
    assert add.await_args.kwargs["author_username"] == "alice"
    assert add.await_args.kwargs["check_hierarchy"] is False
    assert add.await_args.args[1] == kept
    response = sent[-1]
    assert response.type == "objects"
    assert response.data["root_contours"]
    assert response.data["added_count"] == 1
    assert response.data["suppressed_count"] == 2


def test_override_deletes_existing_contours_before_adding_predictions(monkeypatch):
    predictions = [box(0.1, 0.1, 0.2, 0.2), box(0.5, 0.5, 0.6, 0.6)]
    final_hierarchy = ContourHierarchy(root_contours=predictions)
    state, message, sent, delete, add = _setup(
        monkeypatch, write_mode="override", predictions=predictions, hierarchies=[final_hierarchy]
    )

    asyncio.run(handlers.handle_instance_segmentation(object(), message, state))

    delete.assert_awaited_once()
    assert delete.await_args.args == (7,)
    assert add.await_count == 2
    assert all(call.kwargs["author_username"] == "alice" for call in add.await_args_list)
    assert all(call.kwargs["check_hierarchy"] is True for call in add.await_args_list)
    assert sent[-1].data["added_count"] == 2
    assert sent[-1].data["suppressed_count"] == 0


def test_omitted_write_mode_keeps_destructive_override_compatibility(monkeypatch):
    predictions = [box(0.1, 0.1, 0.2, 0.2)]
    final_hierarchy = ContourHierarchy(root_contours=predictions)
    state, message, _sent, delete, add = _setup(
        monkeypatch, predictions=predictions, hierarchies=[final_hierarchy]
    )

    asyncio.run(handlers.handle_instance_segmentation(object(), message, state))

    delete.assert_awaited_once()
    assert add.await_count == 1
    assert add.await_args.kwargs["check_hierarchy"] is True


def test_invalid_write_mode_returns_error_without_inference_or_db_mutation(monkeypatch):
    state, message, sent, delete, add = _setup(monkeypatch, write_mode="merge")

    asyncio.run(handlers.handle_instance_segmentation(object(), message, state))

    assert sent[-1].type == "error"
    assert sent[-1].success is False
    assert "write_mode" in sent[-1].message
    handlers.run_instance_segmentation.assert_not_awaited()
    delete.assert_not_awaited()
    add.assert_not_awaited()


def test_handle_suggestion_reports_accurate_added_count_when_fitting_skips_prediction(monkeypatch):
    seed = box(0.1, 0.1, 0.2, 0.2, contour_id=10)
    seed.label_id = 1
    pred = box(0.3, 0.3, 0.4, 0.4)
    pred.label_id = 1

    state = SimpleNamespace(
        _running_backends={Backends.SUGGESTION_SEGMENTATION.value: object()},
        image_db=SimpleNamespace(file_path="image.png", width=100, height=100, dataset_id=1),
        user_id="alice",
        mask_id=7,
    )
    client_msg = ClientMessage(
        id="req-sugg-1",
        type="suggestion_inference",
        data={"seed_contour_ids": [10], "model_key": "sam3-intra"},
    )
    sent = []

    async def send(_websocket, message):
        sent.append(message)

    monkeypatch.setattr(handlers, "send_msg", send)
    monkeypatch.setattr(
        handlers,
        "run_suggestion_segmentation",
        AsyncMock(return_value=SimpleNamespace(
            contours=[pred], success=True, message="Suggestion finished"
        )),
    )

    db = object()

    @contextmanager
    def context_session():
        yield db

    monkeypatch.setattr(handlers, "get_context_session", context_session)
    monkeypatch.setattr(handlers.contours_db, "get_contours", AsyncMock(return_value=[seed]))
    monkeypatch.setattr(handlers.labels_db, "get_label", AsyncMock(return_value="cell"))
    monkeypatch.setattr(handlers.labels_db, "get_label_hierarchy", AsyncMock(return_value=LabelHierarchy(root_level_labels=[], id_to_label_object={}, value_to_label_object={})))
    monkeypatch.setattr(handlers.masks_db, "get_contour_hierarchy_of_mask", AsyncMock(return_value=ContourHierarchy()))
    # Simulate hierarchy fitting skipping the contour (returns None)
    monkeypatch.setattr(handlers.masks_db, "add_contour_to_mask", AsyncMock(return_value=None))

    asyncio.run(handlers.handle_suggestion(object(), client_msg, state))

    # The FIRST response sharing the client's request ID must be the SUCCESS ack
    matching_messages = [m for m in sent if m.id == "req-sugg-1"]
    assert len(matching_messages) == 1
    assert matching_messages[0].type == "success"
    assert matching_messages[0].data["added_count"] == 0


def test_handle_suggestion_reports_accurate_added_count_when_prediction_persists(monkeypatch):
    seed = box(0.1, 0.1, 0.2, 0.2, contour_id=10)
    seed.label_id = 1
    pred = box(0.3, 0.3, 0.4, 0.4)
    pred.label_id = 1
    persisted = box(0.3, 0.3, 0.4, 0.4, contour_id=20)
    persisted.label_id = 1

    state = SimpleNamespace(
        _running_backends={Backends.SUGGESTION_SEGMENTATION.value: object()},
        image_db=SimpleNamespace(file_path="image.png", width=100, height=100, dataset_id=1),
        user_id="alice",
        mask_id=7,
    )
    client_msg = ClientMessage(
        id="req-sugg-2",
        type="suggestion_inference",
        data={"seed_contour_ids": [10], "model_key": "sam3-intra"},
    )
    sent = []

    async def send(_websocket, message):
        sent.append(message)

    monkeypatch.setattr(handlers, "send_msg", send)
    monkeypatch.setattr(
        handlers,
        "run_suggestion_segmentation",
        AsyncMock(return_value=SimpleNamespace(
            contours=[pred], success=True, message="Suggestion finished"
        )),
    )

    db = object()

    @contextmanager
    def context_session():
        yield db

    monkeypatch.setattr(handlers, "get_context_session", context_session)
    monkeypatch.setattr(handlers.contours_db, "get_contours", AsyncMock(return_value=[seed]))
    monkeypatch.setattr(handlers.labels_db, "get_label", AsyncMock(return_value="cell"))
    monkeypatch.setattr(handlers.labels_db, "get_label_hierarchy", AsyncMock(return_value=LabelHierarchy(root_level_labels=[], id_to_label_object={}, value_to_label_object={})))
    monkeypatch.setattr(handlers.masks_db, "get_contour_hierarchy_of_mask", AsyncMock(return_value=ContourHierarchy()))
    monkeypatch.setattr(handlers.masks_db, "add_contour_to_mask", AsyncMock(return_value=persisted))
    monkeypatch.setattr(handlers.history_db, "record_create", lambda *args, **kwargs: None)

    asyncio.run(handlers.handle_suggestion(object(), client_msg, state))

    # 1. The FIRST message sharing the request ID is the SUCCESS acknowledgement with accurate added_count
    matching_messages = [m for m in sent if m.id == "req-sugg-2"]
    assert len(matching_messages) == 1
    assert matching_messages[0] == sent[0]
    assert matching_messages[0].type == "success"
    assert matching_messages[0].data["added_count"] == 1

    # 2. Subsequent message is the OBJECT_ADDED event (which does not carry the request ID)
    assert sent[1].type == "object_added"
    assert sent[1].id != "req-sugg-2"
    persisted_id = sent[1].data["id"] if isinstance(sent[1].data, dict) else (dict(sent[1].data).get("id") if isinstance(sent[1].data, list) else sent[1].data.id)
    assert persisted_id == persisted.id


def test_handle_suggestion_forwards_parameters_from_inputs(monkeypatch):
    seed = box(0.1, 0.1, 0.2, 0.2, contour_id=10)
    seed.label_id = 1

    state = SimpleNamespace(
        _running_backends={Backends.SUGGESTION_SEGMENTATION.value: object()},
        image_db=SimpleNamespace(file_path="image.png", width=100, height=100, dataset_id=1),
        user_id="alice",
        mask_id=7,
    )
    client_msg = ClientMessage(
        id="req-sugg-params",
        type="suggestion_inference",
        data={
            "seed_contour_ids": [10],
            "model_key": "sam3-intra",
            "inputs": {
                "parameters": {"mask_threshold": 0.85, "min_mask_area": 50},
                "conditioning": {"count": 4},
            },
        },
    )
    sent = []

    async def send(_websocket, message):
        sent.append(message)

    mock_run_sugg = AsyncMock(return_value=SimpleNamespace(
        contours=[], success=True, message="Suggestion finished"
    ))

    monkeypatch.setattr(handlers, "send_msg", send)
    monkeypatch.setattr(handlers, "run_suggestion_segmentation", mock_run_sugg)

    db = object()

    @contextmanager
    def context_session():
        yield db

    monkeypatch.setattr(handlers, "get_context_session", context_session)
    monkeypatch.setattr(handlers.contours_db, "get_contours", AsyncMock(return_value=[seed]))
    monkeypatch.setattr(handlers.labels_db, "get_label", AsyncMock(return_value="cell"))
    monkeypatch.setattr(handlers.labels_db, "get_label_hierarchy", AsyncMock(return_value=LabelHierarchy(root_level_labels=[], id_to_label_object={}, value_to_label_object={})))
    monkeypatch.setattr(handlers.masks_db, "get_contour_hierarchy_of_mask", AsyncMock(return_value=ContourHierarchy()))

    asyncio.run(handlers.handle_suggestion(object(), client_msg, state))

    assert mock_run_sugg.await_count == 1
    call_kwargs = mock_run_sugg.await_args.kwargs
    assert call_kwargs["parameters"] == {"mask_threshold": 0.85, "min_mask_area": 50}
    assert call_kwargs["model_key"] == "sam3-intra"


def test_handle_instance_segmentation_forwards_parameters(monkeypatch):
    state = SimpleNamespace(
        _running_backends={Backends.INSTANCE_SEGMENTATION.value: object()},
        image_db=SimpleNamespace(file_path="image.png", width=100, height=100),
        user_id="alice",
        mask_id=7,
        contour_hierarchy=None,
    )
    client_msg = ClientMessage(
        id="req-inst-params",
        type="instance_inference",
        data={
            "model_registry_key": "m2f-model",
            "write_mode": "patch",
            "inputs": {
                "parameters": {"threshold": 0.7},
            },
        },
    )
    sent = []

    async def send(_websocket, message):
        sent.append(message)

    mock_run_inst = AsyncMock(return_value=InstanceSegmentationResult(
        contours=[], success=True, message="model finished"
    ))

    monkeypatch.setattr(handlers, "send_msg", send)
    monkeypatch.setattr(handlers, "run_instance_segmentation", mock_run_inst)

    db = object()

    @contextmanager
    def context_session():
        yield db

    monkeypatch.setattr(handlers, "get_context_session", context_session)
    monkeypatch.setattr(handlers.masks_db, "get_contour_hierarchy_of_mask", AsyncMock(return_value=ContourHierarchy()))
    monkeypatch.setattr(handlers.masks_db, "get_cached_contour_hierarchy_of_mask", AsyncMock(return_value=(ContourHierarchy(), {})))

    asyncio.run(handlers.handle_instance_segmentation(object(), client_msg, state))

    assert mock_run_inst.await_count == 1
    call_kwargs = mock_run_inst.await_args.kwargs
    assert call_kwargs["parameters"] == {"threshold": 0.7}
    assert call_kwargs["model_registry_key"] == "m2f-model"


def test_handle_suggestion_slices_exemplars_to_conditioning_count(monkeypatch):
    seed1 = box(0.1, 0.1, 0.2, 0.2, contour_id=1)
    seed2 = box(0.3, 0.3, 0.4, 0.4, contour_id=2)
    seed3 = box(0.5, 0.5, 0.6, 0.6, contour_id=3)
    seed1.label_id = 1
    seed2.label_id = 1
    seed3.label_id = 1

    state = SimpleNamespace(
        _running_backends={Backends.SUGGESTION_SEGMENTATION.value: object()},
        image_db=SimpleNamespace(file_path="image.png", width=100, height=100, dataset_id=1),
        user_id="alice",
        mask_id=7,
    )
    client_msg = ClientMessage(
        id="req-sugg-count",
        type="suggestion_inference",
        data={
            "seed_contour_ids": [1, 2, 3],
            "model_key": "sam3-intra",
            "inputs": {
                "parameters": {"mask_threshold": 0.8},
                "conditioning": {"count": 2},
            },
        },
    )
    sent = []

    async def send(_websocket, message):
        sent.append(message)

    mock_run_sugg = AsyncMock(return_value=SimpleNamespace(
        contours=[], success=True, message="Suggestion finished"
    ))

    monkeypatch.setattr(handlers, "send_msg", send)
    monkeypatch.setattr(handlers, "run_suggestion_segmentation", mock_run_sugg)

    db = object()

    @contextmanager
    def context_session():
        yield db

    monkeypatch.setattr(handlers, "get_context_session", context_session)
    monkeypatch.setattr(handlers.contours_db, "get_contours", AsyncMock(return_value=[seed1, seed2]))
    monkeypatch.setattr(handlers.labels_db, "get_label", AsyncMock(return_value="cell"))
    monkeypatch.setattr(handlers.labels_db, "get_label_hierarchy", AsyncMock(return_value=LabelHierarchy(root_level_labels=[], id_to_label_object={}, value_to_label_object={})))
    monkeypatch.setattr(handlers.masks_db, "get_contour_hierarchy_of_mask", AsyncMock(return_value=ContourHierarchy()))

    asyncio.run(handlers.handle_suggestion(object(), client_msg, state))

    assert mock_run_sugg.await_count == 1
    call_kwargs = mock_run_sugg.await_args.kwargs
    # Only 2 positive exemplars should be passed
    assert len(call_kwargs["positive_exemplars"]) == 2
    assert call_kwargs["parameters"] == {"mask_threshold": 0.8}


def test_handle_prompted_segmentation_forwards_parameters(monkeypatch):
    state = SimpleNamespace(
        _running_backends={Backends.PROMPTED_SEGMENTATION.value: object()},
        image_db=SimpleNamespace(file_path="image.png", width=100, height=100),
        user_id="alice",
        refinement_contour_id=None,
        focussed_contour_id=None,
        contour_hierarchy=None,
    )
    client_msg = ClientMessage(
        id="req-prompt-params",
        type="prompted_inference",
        data={
            "model_key": "sam3-prompted",
            "prompts": {
                "point_prompts": [{"x": 0.5, "y": 0.5, "label": True}],
            },
            "inputs": {
                "parameters": {"threshold": 0.9, "multimask_output": False},
            },
        },
    )
    sent = []

    async def send(_websocket, message):
        sent.append(message)

    res_contour = box(0.4, 0.4, 0.6, 0.6)
    mock_run_prompt = AsyncMock(return_value=SimpleNamespace(
        contour=res_contour, candidates=[res_contour], success=True, message="Prompted finished"
    ))

    monkeypatch.setattr(handlers, "send_msg", send)
    monkeypatch.setattr(handlers, "run_prompted_segmentation", mock_run_prompt)
    monkeypatch.setattr(handlers, "add_object", AsyncMock())

    asyncio.run(handlers.handle_prompted_segmentation(object(), client_msg, state))

    assert mock_run_prompt.await_count == 1
    call_kwargs = mock_run_prompt.await_args.kwargs
    assert call_kwargs["parameters"] == {"threshold": 0.9, "multimask_output": False}
    assert call_kwargs["model_key"] == "sam3-prompted"

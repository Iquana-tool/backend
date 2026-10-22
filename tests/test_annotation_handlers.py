"""Focused tests for interactive instance-segmentation write modes."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
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


def test_handle_object_add_preserves_parent_and_provenance(monkeypatch):
    parent_frag = Contour(
        id=21060,
        x=[0.0, 1.0, 1.0, 0.0],
        y=[0.0, 0.0, 1.0, 1.0],
        label_id=10,
    )
    state = SimpleNamespace(
        focussed_contour_id=999,
        mask_id=7,
        user_id="alice",
        label_parents={43: 10, 10: None},
        contour_hierarchy=SimpleNamespace(
            id_to_contour={21060: parent_frag},
            root_contours=[parent_frag],
        ),
    )
    client_msg = ClientMessage(
        id="req-add-1",
        type="object_add_manual",
        data={
            "x": [0.1, 0.2, 0.2, 0.1],
            "y": [0.1, 0.1, 0.2, 0.2],
            "label_id": 43,
            "parent_id": 21060,
            "confidence": 0.95,
            "added_by": "cpsam_polyp_v3_b_seed17",
            "source": "instance_suggestion",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )
    mock_add = AsyncMock()
    monkeypatch.setattr(handlers, "add_object", mock_add)

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    assert mock_add.await_count == 1
    contour_arg = mock_add.await_args.args[0]
    assert contour_arg.parent_id == 21060
    assert contour_arg.label_id == 43
    assert contour_arg.added_by == "cpsam_polyp_v3_b_seed17"
    assert mock_add.await_args.kwargs["source"] == "instance_suggestion"
    assert mock_add.await_args.kwargs["model_key"] == "cpsam_polyp_v3_b_seed17"


def test_handle_object_add_falls_back_to_focussed_parent(monkeypatch):
    state = SimpleNamespace(
        focussed_contour_id=21060,
        mask_id=7,
        user_id="alice",
    )
    client_msg = ClientMessage(
        id="req-add-2",
        type="object_add_manual",
        data={
            "x": [0.1, 0.2, 0.2, 0.1],
            "y": [0.1, 0.1, 0.2, 0.2],
            "label_id": 43,
            "parent_id": None,
            "confidence": 0.95,
        },
    )
    mock_add = AsyncMock()
    monkeypatch.setattr(handlers, "add_object", mock_add)

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    assert mock_add.await_count == 1
    contour_arg = mock_add.await_args.args[0]
    assert contour_arg.parent_id == 21060


def test_handle_object_add_model_prediction_outside_focused_does_not_adopt_parent(monkeypatch):
    """Whole-image model results outside the focused contour must not inherit it."""
    focused_parent = Contour(
        id=21060,
        x=[0.1, 0.3, 0.3, 0.1],
        y=[0.1, 0.1, 0.3, 0.3],
        label_id=10,
    )
    state = SimpleNamespace(
        focussed_contour_id=21060,
        mask_id=7,
        user_id="alice",
        label_parents={43: 10, 10: None},
        contour_hierarchy=SimpleNamespace(
            id_to_contour={21060: focused_parent},
            root_contours=[focused_parent],
        ),
    )
    # Prediction clearly outside [0.1, 0.3] box
    client_msg = ClientMessage(
        id="req-add-outside",
        type="object_add_manual",
        data={
            "x": [0.7, 0.9, 0.9, 0.7],
            "y": [0.7, 0.7, 0.9, 0.9],
            "label_id": 43,
            "parent_id": None,
            "confidence": 0.95,
            "source": "instance_segmentation",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )
    mock_add = AsyncMock()
    monkeypatch.setattr(handlers, "add_object", mock_add)

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    assert mock_add.await_count == 1
    contour_arg = mock_add.await_args.args[0]
    assert contour_arg.parent_id is None


@pytest.mark.parametrize("model_key", ["cpsam_polyp_v3_b_seed17", "yolo_polyp_v3_c_seed1"])
@pytest.mark.parametrize(
    ("candidate_x", "candidate_y"),
    [
        pytest.param([0.22, 0.28, 0.28, 0.22], [0.22, 0.22, 0.28, 0.28], id="contained-smaller"),
        pytest.param([0.245, 0.345, 0.345, 0.245], [0.20, 0.20, 0.30, 0.30], id="shifted"),
    ],
)
def test_polyp_instance_add_suppresses_duplicate_across_parent_branches(
        monkeypatch, model_key, candidate_x, candidate_y
):
    """A rerun must not save a rim when the existing polyp is under another ancestor."""
    parent_a = box(0.0, 0.0, 0.5, 0.5, contour_id=10)
    parent_a.label_id = 10
    parent_b = box(0.0, 0.0, 0.5, 0.5, contour_id=20)
    parent_b.label_id = 10
    existing = box(0.20, 0.20, 0.30, 0.30, contour_id=30, confidence=0.6)
    existing.label_id = 43
    existing.parent_id = parent_a.id
    hierarchy = SimpleNamespace(
        id_to_contour={10: parent_a, 20: parent_b, 30: existing},
        root_contours=[parent_a, parent_b],
        label_id_to_contours={10: [parent_a, parent_b], 43: [existing]},
    )
    state = SimpleNamespace(
        focussed_contour_id=None,
        mask_id=7,
        user_id="alice",
        label_parents={43: 10, 10: None},
        contour_hierarchy=None,
    )
    client_msg = ClientMessage(
        id="req-polyp-rerun",
        type="object_add_manual",
        data={
            "x": candidate_x,
            "y": candidate_y,
            "label_id": 43,
            "parent_id": parent_b.id,
            "confidence": 0.99,
            "source": "instance_segmentation",
            "model_key": model_key,
        },
    )
    add = AsyncMock()
    sent = []

    @contextmanager
    def context_session():
        yield object()

    async def get_hierarchy(_mask_id, _db):
        return hierarchy

    async def send(_websocket, message):
        sent.append(message)

    monkeypatch.setattr(handlers, "get_context_session", context_session)
    monkeypatch.setattr(handlers.masks_db, "get_contour_hierarchy_of_mask", get_hierarchy)
    monkeypatch.setattr(handlers, "add_object", add)
    monkeypatch.setattr(handlers, "send_msg", send)

    candidate = Contour(x=candidate_x, y=candidate_y, label_id=43)
    assert handlers.nms([candidate], iou_threshold=0.5, existing=[existing]).kept == [0]
    assert handlers._polyp_intersection_over_smaller(candidate, existing) >= 0.3

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    assert state.contour_hierarchy is hierarchy
    add.assert_not_awaited()
    assert sent[-1].data == {"skipped": True, "duplicate": True}


def test_polyp_instance_add_keeps_overlapping_contour_with_different_label(monkeypatch):
    existing = box(0.20, 0.20, 0.30, 0.30, contour_id=30)
    existing.label_id = 44
    hierarchy = SimpleNamespace(
        id_to_contour={30: existing},
        root_contours=[existing],
        label_id_to_contours={44: [existing]},
    )
    state = SimpleNamespace(
        focussed_contour_id=None,
        mask_id=7,
        user_id="alice",
        label_parents={43: 10, 44: 10, 10: None},
        contour_hierarchy=None,
    )
    client_msg = ClientMessage(
        id="req-polyp-different-label",
        type="object_add_manual",
        data={
            "x": [0.205, 0.305, 0.305, 0.205],
            "y": [0.205, 0.205, 0.305, 0.305],
            "label_id": 43,
            "confidence": 0.9,
            "source": "instance_segmentation",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )
    add = AsyncMock()

    @contextmanager
    def context_session():
        yield object()

    async def get_hierarchy(_mask_id, _db):
        return hierarchy

    monkeypatch.setattr(handlers, "get_context_session", context_session)
    monkeypatch.setattr(handlers.masks_db, "get_contour_hierarchy_of_mask", get_hierarchy)
    monkeypatch.setattr(handlers, "add_object", add)

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    add.assert_awaited_once()


def test_polyp_instance_add_keeps_slight_same_label_edge_overlap(monkeypatch):
    existing = box(0.20, 0.20, 0.30, 0.30, contour_id=30)
    existing.label_id = 43
    hierarchy = SimpleNamespace(
        id_to_contour={30: existing},
        root_contours=[existing],
        label_id_to_contours={43: [existing]},
    )
    state = SimpleNamespace(
        focussed_contour_id=None,
        mask_id=7,
        user_id="alice",
        label_parents={43: 10, 10: None},
        contour_hierarchy=None,
    )
    client_msg = ClientMessage(
        id="req-polyp-distinct",
        type="object_add_manual",
        data={
            "x": [0.275, 0.375, 0.375, 0.275],
            "y": [0.20, 0.20, 0.30, 0.30],
            "label_id": 43,
            "confidence": 0.9,
            "source": "instance_segmentation",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )
    add = AsyncMock()

    @contextmanager
    def context_session():
        yield object()

    async def get_hierarchy(_mask_id, _db):
        return hierarchy

    candidate = Contour(x=[0.275, 0.375, 0.375, 0.275], y=[0.20, 0.20, 0.30, 0.30], label_id=43)
    assert handlers._polyp_intersection_over_smaller(candidate, existing) < 0.3

    monkeypatch.setattr(handlers, "get_context_session", context_session)
    monkeypatch.setattr(handlers.masks_db, "get_contour_hierarchy_of_mask", get_hierarchy)
    monkeypatch.setattr(handlers, "add_object", add)

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    add.assert_awaited_once()


def test_handle_object_add_model_prediction_inside_focused_adopts_parent_via_containment(monkeypatch):
    """Whole-image model results inside the focused contour adopt it when it is an allowed ancestor."""
    focused_parent = Contour(
        id=21060,
        x=[0.1, 0.5, 0.5, 0.1],
        y=[0.1, 0.1, 0.5, 0.5],
        label_id=10,
    )
    state = SimpleNamespace(
        focussed_contour_id=21060,
        mask_id=7,
        user_id="alice",
        label_parents={43: 10, 10: None},
        contour_hierarchy=SimpleNamespace(
            id_to_contour={21060: focused_parent},
            root_contours=[focused_parent],
        ),
    )
    # Prediction clearly inside [0.1, 0.5] box
    client_msg = ClientMessage(
        id="req-add-inside",
        type="object_add_manual",
        data={
            "x": [0.2, 0.3, 0.3, 0.2],
            "y": [0.2, 0.2, 0.3, 0.3],
            "label_id": 43,
            "parent_id": None,
            "confidence": 0.95,
            "source": "instance_segmentation",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )
    mock_add = AsyncMock()
    monkeypatch.setattr(handlers, "add_object", mock_add)

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    assert mock_add.await_count == 1
    contour_arg = mock_add.await_args.args[0]
    assert contour_arg.parent_id == 21060


def test_handle_object_add_model_prediction_does_not_adopt_focused_same_label_contour(monkeypatch):
    """A new prediction must NOT be parented beneath an existing contour with the same label."""
    existing_same_label_polyp = Contour(
        id=21065,
        x=[0.1, 0.5, 0.5, 0.1],
        y=[0.1, 0.1, 0.5, 0.5],
        label_id=43,
    )
    state = SimpleNamespace(
        focussed_contour_id=21065,
        mask_id=7,
        user_id="alice",
        label_parents={43: 10, 10: None},
        contour_hierarchy=SimpleNamespace(
            id_to_contour={21065: existing_same_label_polyp},
            root_contours=[existing_same_label_polyp],
        ),
    )
    client_msg = ClientMessage(
        id="req-add-same-label",
        type="object_add_manual",
        data={
            "x": [0.2, 0.3, 0.3, 0.2],
            "y": [0.2, 0.2, 0.3, 0.3],
            "label_id": 43,
            "parent_id": None,
            "confidence": 0.95,
            "source": "instance_segmentation",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )
    mock_add = AsyncMock()
    monkeypatch.setattr(handlers, "add_object", mock_add)

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    assert mock_add.await_count == 1
    contour_arg = mock_add.await_args.args[0]
    # Same label cannot be a parent -> must remain None
    assert contour_arg.parent_id is None


def test_handle_object_add_model_prediction_fallback_to_valid_ancestor_and_ignores_unrelated(monkeypatch):
    """Fallback selects a valid ancestor and ignores containing unrelated or same-label objects."""
    focused_polyp = Contour(
        id=21065,
        x=[0.1, 0.5, 0.5, 0.1],
        y=[0.1, 0.1, 0.5, 0.5],
        label_id=43,
    )
    unrelated_debris = Contour(
        id=21070,
        x=[0.05, 0.6, 0.6, 0.05],
        y=[0.05, 0.05, 0.6, 0.6],
        label_id=99,
    )
    valid_fragment = Contour(
        id=21060,
        x=[0.0, 0.8, 0.8, 0.0],
        y=[0.0, 0.0, 0.8, 0.8],
        label_id=10,
    )
    state = SimpleNamespace(
        focussed_contour_id=21065,
        mask_id=7,
        user_id="alice",
        label_parents={43: 10, 10: None, 99: None},
        contour_hierarchy=SimpleNamespace(
            id_to_contour={
                21065: focused_polyp,
                21070: unrelated_debris,
                21060: valid_fragment,
            },
            root_contours=[valid_fragment],
        ),
    )
    client_msg = ClientMessage(
        id="req-add-fallback-ancestor",
        type="object_add_manual",
        data={
            "x": [0.2, 0.3, 0.3, 0.2],
            "y": [0.2, 0.2, 0.3, 0.3],
            "label_id": 43,
            "parent_id": None,
            "confidence": 0.95,
            "source": "instance_segmentation",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )
    mock_add = AsyncMock()
    monkeypatch.setattr(handlers, "add_object", mock_add)

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    assert mock_add.await_count == 1
    contour_arg = mock_add.await_args.args[0]
    # Focused polyp (43) and debris (99) are rejected; valid fragment (10) is adopted!
    assert contour_arg.parent_id == 21060


def test_handle_object_add_model_prediction_no_focused_does_not_infer_parent(monkeypatch):
    """When no contour is focused, model prediction parent is not inferred (remains None)."""
    valid_fragment = Contour(
        id=3001,
        x=[0.0, 0.8, 0.8, 0.0],
        y=[0.0, 0.0, 0.8, 0.8],
        label_id=10,
    )
    state = SimpleNamespace(
        focussed_contour_id=None,
        mask_id=7,
        user_id="alice",
        label_parents={43: 10, 10: None},
        contour_hierarchy=SimpleNamespace(
            id_to_contour={3001: valid_fragment},
            root_contours=[valid_fragment],
        ),
    )
    client_msg = ClientMessage(
        id="req-add-no-focused",
        type="object_add_manual",
        data={
            "x": [0.2, 0.3, 0.3, 0.2],
            "y": [0.2, 0.2, 0.3, 0.3],
            "label_id": 43,
            "parent_id": None,
            "confidence": 0.95,
            "source": "instance_segmentation",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )
    mock_add = AsyncMock()
    monkeypatch.setattr(handlers, "add_object", mock_add)

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    assert mock_add.await_count == 1
    contour_arg = mock_add.await_args.args[0]
    assert contour_arg.parent_id is None


def test_handle_object_add_model_prediction_closest_ancestor_preferred_over_distant(monkeypatch):
    """Nearest taxonomy ancestor (immediate parent) is chosen over distant grandparent ancestor in fallback."""
    unrelated_focused = Contour(
        id=9999,
        x=[0.9, 0.95, 0.95, 0.9],
        y=[0.9, 0.9, 0.95, 0.95],
        label_id=88,  # Unrelated label outside prediction
    )
    colony = Contour(
        id=100,
        x=[0.0, 1.0, 1.0, 0.0],
        y=[0.0, 0.0, 1.0, 1.0],
        label_id=5,  # Grandparent (colony)
    )
    fragment = Contour(
        id=200,
        x=[0.1, 0.9, 0.9, 0.1],
        y=[0.1, 0.1, 0.9, 0.9],
        label_id=10,  # Immediate parent (fragment)
    )
    # 43 (polyp) -> 10 (fragment) -> 5 (colony) -> None
    state = SimpleNamespace(
        focussed_contour_id=9999,
        mask_id=7,
        user_id="alice",
        label_parents={43: 10, 10: 5, 5: None, 88: None},
        contour_hierarchy=SimpleNamespace(
            id_to_contour={9999: unrelated_focused, 100: colony, 200: fragment},
            root_contours=[colony],
        ),
    )
    client_msg = ClientMessage(
        id="req-add-nearest-ancestor",
        type="object_add_manual",
        data={
            "x": [0.2, 0.3, 0.3, 0.2],
            "y": [0.2, 0.2, 0.3, 0.3],
            "label_id": 43,
            "parent_id": None,
            "confidence": 0.95,
            "source": "instance_segmentation",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )
    mock_add = AsyncMock()
    monkeypatch.setattr(handlers, "add_object", mock_add)

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    assert mock_add.await_count == 1
    contour_arg = mock_add.await_args.args[0]
    # Fragment (label 10, distance 0) is preferred over Colony (label 5, distance 1)
    assert contour_arg.parent_id == 200


def test_handle_object_add_model_prediction_explicit_parent_invalid_ancestor_cleared(monkeypatch):
    """Explicit model parent is cleared when its label is not in dataset label ancestry (e.g. root polyp)."""
    fragment = Contour(
        id=21060,
        x=[0.0, 1.0, 1.0, 0.0],
        y=[0.0, 0.0, 1.0, 1.0],
        label_id=10,
    )
    state = SimpleNamespace(
        focussed_contour_id=None,
        mask_id=7,
        user_id="alice",
        label_parents={43: None, 10: None},  # Root-level polyp (no ancestors)
        contour_hierarchy=SimpleNamespace(
            id_to_contour={21060: fragment},
            root_contours=[fragment],
        ),
    )
    # Client sends explicit parent_id pointing to fragment 21060
    client_msg = ClientMessage(
        id="req-add-explicit-invalid-ancestor",
        type="object_add_manual",
        data={
            "x": [0.2, 0.3, 0.3, 0.2],
            "y": [0.2, 0.2, 0.3, 0.3],
            "label_id": 43,
            "parent_id": 21060,
            "confidence": 0.95,
            "source": "instance_segmentation",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )
    mock_add = AsyncMock()
    monkeypatch.setattr(handlers, "add_object", mock_add)

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    assert mock_add.await_count == 1
    contour_arg = mock_add.await_args.args[0]
    # Incompatible explicit parent must be stripped -> remains None
    assert contour_arg.parent_id is None


def test_handle_object_add_model_prediction_explicit_parent_valid_ancestor_preserved(monkeypatch):
    """Explicit model parent is preserved when its label is a valid dataset ancestor and contains the child."""
    fragment = Contour(
        id=21060,
        x=[0.0, 1.0, 1.0, 0.0],
        y=[0.0, 0.0, 1.0, 1.0],
        label_id=10,
    )
    state = SimpleNamespace(
        focussed_contour_id=None,
        mask_id=7,
        user_id="alice",
        label_parents={43: 10, 10: None},  # Polyp is child of Fragment 10
        contour_hierarchy=SimpleNamespace(
            id_to_contour={21060: fragment},
            root_contours=[fragment],
        ),
    )
    client_msg = ClientMessage(
        id="req-add-explicit-valid-ancestor",
        type="object_add_manual",
        data={
            "x": [0.2, 0.3, 0.3, 0.2],
            "y": [0.2, 0.2, 0.3, 0.3],
            "label_id": 43,
            "parent_id": 21060,
            "confidence": 0.95,
            "source": "instance_segmentation",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )
    mock_add = AsyncMock()
    monkeypatch.setattr(handlers, "add_object", mock_add)

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    assert mock_add.await_count == 1
    contour_arg = mock_add.await_args.args[0]
    # Valid explicit parent is accepted
    assert contour_arg.parent_id == 21060


def test_handle_object_add_model_prediction_explicit_parent_same_label_cleared(monkeypatch):
    """Explicit model parent pointing to another contour with the same label is cleared."""
    polyp1 = Contour(
        id=21065,
        x=[0.1, 0.8, 0.8, 0.1],
        y=[0.1, 0.1, 0.8, 0.8],
        label_id=43,
    )
    state = SimpleNamespace(
        focussed_contour_id=None,
        mask_id=7,
        user_id="alice",
        label_parents={43: 10, 10: None},
        contour_hierarchy=SimpleNamespace(
            id_to_contour={21065: polyp1},
            root_contours=[polyp1],
        ),
    )
    client_msg = ClientMessage(
        id="req-add-explicit-same-label",
        type="object_add_manual",
        data={
            "x": [0.2, 0.3, 0.3, 0.2],
            "y": [0.2, 0.2, 0.3, 0.3],
            "label_id": 43,
            "parent_id": 21065,
            "confidence": 0.95,
            "source": "instance_segmentation",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )
    mock_add = AsyncMock()
    monkeypatch.setattr(handlers, "add_object", mock_add)

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    assert mock_add.await_count == 1
    contour_arg = mock_add.await_args.args[0]
    assert contour_arg.parent_id is None


def test_handle_object_add_refreshes_hierarchy_from_storage_when_state_is_stale(monkeypatch):
    """When state.contour_hierarchy has a stale snapshot (missing newly added parent),
    handle_object_add queries masks_db.get_contour_hierarchy_of_mask, updates state.contour_hierarchy,
    and validates the parent against the refreshed storage hierarchy.
    """
    stale_hierarchy = SimpleNamespace(
        id_to_contour={},
        root_contours=[],
    )
    fresh_parent = Contour(
        id=21060,
        x=[0.0, 1.0, 1.0, 0.0],
        y=[0.0, 0.0, 1.0, 1.0],
        label_id=10,
    )
    fresh_hierarchy = SimpleNamespace(
        id_to_contour={21060: fresh_parent},
        root_contours=[fresh_parent],
    )
    state = SimpleNamespace(
        focussed_contour_id=None,
        mask_id=7,
        user_id="alice",
        label_parents={43: 10, 10: None},
        contour_hierarchy=stale_hierarchy,
    )
    client_msg = ClientMessage(
        id="req-add-stale-refresh",
        type="object_add_manual",
        data={
            "x": [0.1, 0.2, 0.2, 0.1],
            "y": [0.1, 0.1, 0.2, 0.2],
            "label_id": 43,
            "parent_id": 21060,
            "confidence": 0.95,
            "source": "instance_suggestion",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )
    mock_add = AsyncMock()
    monkeypatch.setattr(handlers, "add_object", mock_add)

    # Mock get_contour_hierarchy_of_mask to return fresh_hierarchy
    mock_get_hierarchy = AsyncMock(return_value=fresh_hierarchy)
    monkeypatch.setattr(handlers.masks_db, "get_contour_hierarchy_of_mask", mock_get_hierarchy)

    asyncio.run(handlers.handle_object_add(object(), client_msg, state))

    assert mock_get_hierarchy.await_count == 1
    assert state.contour_hierarchy is fresh_hierarchy
    assert mock_add.await_count == 1
    contour_arg = mock_add.await_args.args[0]
    assert contour_arg.parent_id == 21060



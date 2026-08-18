"""Focused tests for interactive instance-segmentation write modes."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

from iquana_toolbox.schemas.database.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.database.contours import Contour
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

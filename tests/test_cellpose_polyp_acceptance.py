"""Integration and regression test for Cellpose-SAM polyp acceptance workflow.

Demonstrates:
1. Candidate outline with parent_id or focussed_contour_id persists inside a parent fragment.
2. The newly persisted contour receives a canonical database ID and correct parent_id / label_id.
3. Reloading the mask hierarchy from the database restores the nested polyp contour.
4. Editing an accepted polyp modifies the geometry and survives reload.
5. Deleting an accepted polyp removes it from persistence while leaving the parent fragment intact.
"""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database
from app.database.users import Users
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.masks import Masks
from app.database.labels import Labels
from app.database.contours import Contours
from app.database.ai_suggestions import AiSuggestions, SuggestionSource
from app.routes.websockets import annotation_handlers as handlers
from app.services.database_access import masks as masks_db
from app.services.database_access import contours as contours_db
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.networking.websockets.annotation_session import (
    ClientMessage,
    ServerMessageType,
)


@pytest.fixture
def scratch_ctx(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'acceptance_scratch.db'}")
    database.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    db = session_factory()

    # User
    user = Users(username="researcher", hashed_password="pw")
    db.add(user)
    # Dataset
    ds = Datasets(name="Corals", description="Test Dataset", dataset_type="image", folder_path="/tmp/corals", created_by="researcher")
    db.add(ds)
    db.flush()

    # Labels hierarchy: Coral Fragment (41) -> Coral Surface (42) -> Coral Polyp (43)
    l_frag = Labels(id=41, dataset_id=ds.id, parent_id=None, name="Coral Fragment", value=1)
    l_surf = Labels(id=42, dataset_id=ds.id, parent_id=41, name="Coral Surface", value=2)
    l_polyp = Labels(id=43, dataset_id=ds.id, parent_id=42, name="Coral Polyp", value=3)
    db.add_all([l_frag, l_surf, l_polyp])

    # Image (100x100) and Mask
    img = Images(
        id=385,
        dataset_id=ds.id,
        file_name="coral_385.png",
        file_path="/tmp/coral_385.png",
        thumbnail_file_path="/tmp/t_385.png",
        width=100,
        height=100,
        color_mode="RGB",
    )
    db.add(img)
    db.flush()

    mask = Masks(id=385, image_id=img.id, fully_annotated=False, file_path="/tmp/m_385.png")
    db.add(mask)
    db.flush()

    # Root Coral Fragment contour (id=21060) spanning normalized [0.1, 0.9]
    frag_contour = Contours(
        id=21060,
        mask_id=mask.id,
        parent_id=None,
        label_id=41,
        added_by="User",
        author_username="researcher",
        confidence_score=1.0,
        area=6400.0,
        perimeter=320.0,
        circularity=0.8,
        diameter=80.0,
        x=[0.1, 0.9, 0.9, 0.1],
        y=[0.1, 0.1, 0.9, 0.9],
    )
    db.add(frag_contour)
    db.commit()

    ds_id = ds.id
    mask_id = mask.id

    # Mock metrics invalidations that require separate tables not in minimal scope
    for name in (
        "mark_appearance_stale",
        "mark_contextual_stale_for_group",
        "mark_relational_stale_for_parent",
        "_invalidate_metrics_for_modified_contour",
    ):
        monkeypatch.setattr(contours_db, name, lambda *a, **k: None, raising=False)

    return session_factory, ds_id, img, mask_id, frag_contour


def test_cellpose_polyp_acceptance_persistence_and_reload(scratch_ctx, monkeypatch):
    session_factory, ds_id, img, mask_id, frag_contour = scratch_ctx

    # 1. Simulate websocket connection and state
    state = SimpleNamespace(
        mask_id=mask_id,
        dataset_id=ds_id,
        user_id="researcher",
        focussed_contour_id=21060,
        image_db=img,
    )

    sent_messages = []
    async def fake_send_msg(_websocket, message):
        sent_messages.append(message)

    monkeypatch.setattr(handlers, "send_msg", fake_send_msg)

    # Wrap context_session to use our test session
    @contextmanager
    def test_session():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(handlers, "get_context_session", test_session)

    # 2. Acceptance message from frontend: outline inside the fragment
    polyp_client_msg = ClientMessage(
        id="msg-accept-polyp-1",
        type="object_add_manual",
        data={
            "x": [0.3, 0.4, 0.4, 0.3],
            "y": [0.3, 0.3, 0.4, 0.4],
            "label_id": 43,
            "parent_id": 21060,
            "confidence": 0.94,
            "added_by": "cpsam_polyp_v3_b_seed17",
            "source": "instance_suggestion",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )

    asyncio.run(handlers.handle_object_add(object(), polyp_client_msg, state))

    # Verify server acknowledgement
    assert len(sent_messages) == 1
    reply = sent_messages[0]
    assert reply.type == ServerMessageType.OBJECT_ADDED
    assert reply.success is True
    added_data = reply.data
    saved_id = added_data.id if hasattr(added_data, "id") else added_data.get("id")
    assert saved_id is not None
    assert saved_id > 0
    assert saved_id != 21060

    # Verify provenance was recorded in AiSuggestions
    with test_session() as db:
        suggestion = db.query(AiSuggestions).filter_by(contour_id=saved_id).first()
        assert suggestion is not None
        assert suggestion.model_key == "cpsam_polyp_v3_b_seed17"
        assert suggestion.source == SuggestionSource.INSTANCE_SUGGESTION

    # 3. Simulate image reload: fetch hierarchy from database
    with test_session() as db:
        hierarchy = asyncio.run(masks_db.get_contour_hierarchy_of_mask(mask_id, db))
        assert len(hierarchy.root_contours) == 1
        root = hierarchy.root_contours[0]
        assert root.id == 21060
        assert len(root.children) == 1
        child = root.children[0]
        assert child.id == saved_id
        assert child.parent_id == 21060
        assert child.label_id == 43
        assert child.added_by == "cpsam_polyp_v3_b_seed17"
        assert len(child.x) == 4

    # 4. Modify accepted object (e.g. edit vertices) and verify persistence survives reload
    new_geometry = Contour(
        id=saved_id,
        mask_id=mask_id,
        parent_id=21060,
        label_id=43,
        x=[0.31, 0.42, 0.42, 0.31],
        y=[0.31, 0.31, 0.42, 0.42],
        added_by="cpsam_polyp_v3_b_seed17",
    )
    replace_msg = ClientMessage(id="msg-mod-1", type="object_modify", data={"contour_id": saved_id})
    asyncio.run(handlers.replace_object(saved_id, new_geometry, object(), replace_msg, state))

    with test_session() as db:
        reloaded_hierarchy = asyncio.run(masks_db.get_contour_hierarchy_of_mask(mask_id, db))
        reloaded_child = reloaded_hierarchy.root_contours[0].children[0]
        assert reloaded_child.id == saved_id
        assert reloaded_child.x[0] == pytest.approx(0.31)

    # 5. Delete accepted object and verify removal
    delete_msg = ClientMessage(id="msg-del-1", type="object_delete", data={"contour_id": saved_id})
    asyncio.run(handlers.handle_object_delete(object(), delete_msg, state))

    with test_session() as db:
        after_delete = asyncio.run(masks_db.get_contour_hierarchy_of_mask(mask_id, db))
        assert len(after_delete.root_contours) == 1
        assert len(after_delete.root_contours[0].children) == 0


def test_focus_mode_parent_fallback_and_unparented_clipping(scratch_ctx, monkeypatch):
    session_factory, ds_id, img, mask_id, frag_contour = scratch_ctx

    sent_messages = []
    async def fake_send_msg(_websocket, message):
        sent_messages.append(message)

    monkeypatch.setattr(handlers, "send_msg", fake_send_msg)

    @contextmanager
    def test_session():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(handlers, "get_context_session", test_session)

    # Case A: Candidate with parent_id=None but state.focussed_contour_id=21060
    # Must fall back to 21060 and successfully persist inside the fragment.
    state_focused = SimpleNamespace(
        mask_id=mask_id,
        dataset_id=ds_id,
        user_id="researcher",
        focussed_contour_id=21060,
        image_db=img,
    )
    msg_focused = ClientMessage(
        id="msg-focused-polyp",
        type="object_add_manual",
        data={
            "x": [0.4, 0.5, 0.5, 0.4],
            "y": [0.4, 0.4, 0.5, 0.5],
            "label_id": 43,
            "parent_id": None,
            "confidence": 0.9,
            "added_by": "cpsam_polyp_v3_b_seed17",
        },
    )
    asyncio.run(handlers.handle_object_add(object(), msg_focused, state_focused))
    assert len(sent_messages) == 1
    assert sent_messages[0].type == ServerMessageType.OBJECT_ADDED
    assert sent_messages[0].success is True
    saved_focused_id = sent_messages[0].data["id"]

    # Verify reloaded contour has parent_id == 21060
    with test_session() as db:
        h = asyncio.run(masks_db.get_contour_hierarchy_of_mask(mask_id, db))
        assert any(c.id == saved_focused_id and c.parent_id == 21060 for c in h.root_contours[0].children)

    # Case B: Candidate completely inside fragment, but unparented (parent_id=None and focussed_contour_id=None)
    # The hierarchy fitting algorithm clips 100% of forbidden overlap with root fragment,
    # returning skipped: true.
    sent_messages.clear()
    state_unfocused = SimpleNamespace(
        mask_id=mask_id,
        dataset_id=ds_id,
        user_id="researcher",
        focussed_contour_id=None,
        image_db=img,
    )
    msg_unparented = ClientMessage(
        id="msg-unparented-polyp",
        type="object_add_manual",
        data={
            "x": [0.6, 0.7, 0.7, 0.6],
            "y": [0.6, 0.6, 0.7, 0.7],
            "label_id": 43,
            "parent_id": None,
            "confidence": 0.9,
            "added_by": "cpsam_polyp_v3_b_seed17",
        },
    )
    asyncio.run(handlers.handle_object_add(object(), msg_unparented, state_unfocused))
    assert len(sent_messages) == 1
    # Returns ServerMessageType.SUCCESS with skipped: True
    assert sent_messages[0].type == ServerMessageType.SUCCESS
    assert sent_messages[0].data.get("skipped") is True


def test_open_image_draw_fragment_run_cellpose_without_reload(scratch_ctx, monkeypatch):
    """Test the complete flow:
    1. Open image (loads existing/empty hierarchy snapshot).
    2. Draw a new fragment interactively.
    3. Run Cellpose on the newly saved fragment without reloading the session.
    4. Verify the polyp is accepted (not skipped) and hierarchy fitting nests it under the new fragment.
    """
    session_factory, ds_id, _img, _mask_id, _frag_contour = scratch_ctx

    sent_messages = []
    async def fake_send_msg(_websocket, message):
        sent_messages.append(message)

    monkeypatch.setattr(handlers, "send_msg", fake_send_msg)

    @contextmanager
    def test_session():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(handlers, "get_context_session", test_session)

    # 1. Create a fresh image and mask with 0 contours to simulate a newly opened image
    with test_session() as db:
        new_img = Images(
            id=386,
            dataset_id=ds_id,
            file_name="coral_386.png",
            file_path="/tmp/coral_386.png",
            thumbnail_file_path="/tmp/t_386.png",
            width=100,
            height=100,
            color_mode="RGB",
        )
        db.add(new_img)
        db.flush()
        new_mask = Masks(id=386, image_id=new_img.id, fully_annotated=False, file_path="/tmp/m_386.png")
        db.add(new_mask)
        db.commit()

    # Step 1: "open image" — state is initialized and initial hierarchy is loaded into state.contour_hierarchy
    with test_session() as db:
        initial_hierarchy = asyncio.run(masks_db.get_contour_hierarchy_of_mask(new_mask.id, db))

    state = SimpleNamespace(
        mask_id=new_mask.id,
        dataset_id=ds_id,
        user_id="researcher",
        focussed_contour_id=None,
        image_db=new_img,
        contour_hierarchy=initial_hierarchy,
    )
    assert len(state.contour_hierarchy.id_to_contour) == 0

    # Step 2: "draw fragment" — user manually draws a Coral Fragment (label 41)
    frag_draw_msg = ClientMessage(
        id="msg-draw-fragment",
        type="object_add_manual",
        data={
            "x": [0.1, 0.9, 0.9, 0.1],
            "y": [0.1, 0.1, 0.9, 0.9],
            "label_id": 41,
            "confidence": 1.0,
            "source": "manual",
            "added_by": "User",
        },
    )
    sent_messages.clear()
    asyncio.run(handlers.handle_object_add(object(), frag_draw_msg, state))

    assert len(sent_messages) == 1
    assert sent_messages[0].type == ServerMessageType.OBJECT_ADDED
    assert sent_messages[0].success is True
    drawn_frag = sent_messages[0].data
    drawn_frag_id = drawn_frag.id if hasattr(drawn_frag, "id") else drawn_frag.get("id")
    assert drawn_frag_id is not None

    # Step 3: "run Cellpose without reloading"
    # Even if state.contour_hierarchy held a stale snapshot before validation,
    # the handler must refresh the hierarchy from current storage.
    state.focussed_contour_id = drawn_frag_id

    polyp_model_msg = ClientMessage(
        id="msg-cellpose-polyp-1",
        type="object_add_manual",
        data={
            "x": [0.3, 0.4, 0.4, 0.3],
            "y": [0.3, 0.3, 0.4, 0.4],
            "label_id": 43,
            "parent_id": drawn_frag_id,
            "confidence": 0.93,
            "added_by": "cpsam_polyp_v3_b_seed17",
            "source": "instance_suggestion",
            "model_key": "cpsam_polyp_v3_b_seed17",
        },
    )
    sent_messages.clear()
    asyncio.run(handlers.handle_object_add(object(), polyp_model_msg, state))

    # Assert that the polyp is accepted and NOT skipped/dropped
    assert len(sent_messages) == 1
    reply = sent_messages[0]
    assert reply.type == ServerMessageType.OBJECT_ADDED
    assert reply.success is True
    assert reply.data.get("skipped") is not True

    polyp_data = reply.data
    saved_polyp_id = polyp_data.id if hasattr(polyp_data, "id") else polyp_data.get("id")
    assert saved_polyp_id is not None
    assert saved_polyp_id != drawn_frag_id

    # Verify that reloading from the database confirms the nested hierarchy
    with test_session() as db:
        reloaded = asyncio.run(masks_db.get_contour_hierarchy_of_mask(new_mask.id, db))
        assert len(reloaded.root_contours) == 1
        root_frag = reloaded.root_contours[0]
        assert root_frag.id == drawn_frag_id
        assert len(root_frag.children) == 1
        child_polyp = root_frag.children[0]
        assert child_polyp.id == saved_polyp_id
        assert child_polyp.parent_id == drawn_frag_id
        assert child_polyp.label_id == 43
        assert child_polyp.added_by == "cpsam_polyp_v3_b_seed17"



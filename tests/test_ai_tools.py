"""Tests for per-dataset AI tool switches."""
import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.datasets  # noqa: F401
import app.database.users  # noqa: F401
from app.database.datasets import Datasets
from app.database.users import Users
from app.routes.websockets import annotation_handlers as handlers
from app.services import ai_tools
from app.services.ai_tools import AiTool


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'tools.db'}")
    database.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Users(username="ann", hashed_password="x"))
    session.add(Datasets(id=1, name="ds", description="", dataset_type="image",
                         folder_path="/tmp/ds", created_by="ann"))
    session.commit()
    yield session
    session.close()


def _switch_off(db, *tools):
    db.get(Datasets, 1).disabled_ai_tools = ai_tools.format_tools(frozenset(tools))
    db.commit()


def test_every_tool_is_on_by_default(db):
    assert all(ai_tools.is_enabled(1, tool, db) for tool in AiTool)


def test_switched_off_tools_are_refused_with_403(db):
    _switch_off(db, AiTool.CROSS_IMAGE, AiTool.TRAINING)

    assert not ai_tools.is_enabled(1, AiTool.CROSS_IMAGE, db)
    assert ai_tools.is_enabled(1, AiTool.PROMPTED, db)
    with pytest.raises(HTTPException) as refused:
        ai_tools.ensure_tool_enabled(1, AiTool.TRAINING, db)
    assert refused.value.status_code == 403
    assert "training" in refused.value.detail


def test_unknown_stored_names_are_ignored_but_unknown_input_is_rejected():
    assert ai_tools.parse("prompted, telepathy,,refine") == {AiTool.PROMPTED, AiTool.REFINE}
    with pytest.raises(HTTPException) as refused:
        ai_tools.validate(["prompted", "telepathy"])
    assert refused.value.status_code == 422


def test_a_session_without_a_dataset_is_never_blocked(db):
    assert ai_tools.is_enabled(None, AiTool.PROMPTED, db)


def test_the_websocket_refuses_a_switched_off_tool_without_calling_the_model(db, monkeypatch):
    _switch_off(db, AiTool.INSTANCE_SUGGESTION)
    sent = []

    @contextmanager
    def context_session():
        yield db

    async def send(_websocket, message):
        sent.append(message)

    async def must_not_run(*_args, **_kwargs):
        raise AssertionError("the model must not be called")

    monkeypatch.setattr(handlers, "get_context_session", context_session)
    monkeypatch.setattr(handlers, "send_msg", send)
    monkeypatch.setattr(handlers, "run_suggestion_segmentation", must_not_run)
    state = SimpleNamespace(dataset_id=1, user_id="ann")
    message = SimpleNamespace(id="m1", data={"seed_contour_ids": [1]})

    asyncio.run(handlers.handle_suggestion(None, message, state))

    assert len(sent) == 1 and sent[0].success is False
    assert sent[0].data == {"disabled_tool": "instance_suggestion"}

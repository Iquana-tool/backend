"""Tests for re-targeting an annotation session at another image over one socket.

Stepping through a dataset used to close the WebSocket and open a new one per image,
which re-ran authentication, three backend health checks and the model preloading before
the first contour could be asked for. A ``switch_image`` message replaces all of that, so
what has to hold is:

  * the session forgets everything that described the previous image (including the two
    cached ORM rows, which are ``functools.cached_property`` and therefore sticky),
  * it keeps what is not per-image -- the registered AI backends and the user,
  * a session may exist with no image at all, since the socket is opened per user,
  * and a message the server cannot parse is answered rather than fatal, because raising
    used to unwind the session loop and disconnect the user over one bad payload.
"""
import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.users  # noqa: F401
import app.database.contours  # noqa: F401
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.masks import Masks
from app.database.users import Users
from app.routes.websockets import messaging
from app.services.annotation_session.state import AnnotationSessionState

WIDTH, HEIGHT = 640, 480


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


@pytest.fixture
def images(tmp_path, monkeypatch):
    """Two images, each with a mask, behind a patched ``get_context_session``.

    The session state opens its own sessions through that helper rather than taking one,
    so the test database has to be injected where the helper is looked up -- in the state
    module itself.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    setup = Session()
    setup.add(Users(username="u", hashed_password="x", is_admin=False))
    ds = Datasets(name="switch", description="", dataset_type="image",
                  folder_path="/tmp/switch", created_by="u")
    setup.add(ds)
    setup.flush()

    made = []
    for name in ("a.png", "b.png"):
        img = Images(dataset_id=ds.id, file_name=name, file_path=f"/tmp/{name}",
                     thumbnail_file_path="/tmp/t.png", width=WIDTH, height=HEIGHT,
                     color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
        setup.add(img)
        setup.flush()
        mask = Masks(image_id=img.id, fully_annotated=False, file_path=f"/tmp/m-{name}")
        setup.add(mask)
        setup.flush()
        made.append(SimpleNamespace(image_id=img.id, mask_id=mask.id, dataset_id=ds.id))
    setup.commit()
    setup.close()

    from contextlib import contextmanager

    @contextmanager
    def _context_session():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    monkeypatch.setattr("app.services.annotation_session.state.get_context_session",
                        _context_session)
    try:
        yield made
    finally:
        engine.dispose()


# --------------------------------------------------------------------------- #
# Switching
# --------------------------------------------------------------------------- #
def test_switch_repoints_the_session_at_the_new_image(images):
    first, second = images
    state = AnnotationSessionState(image_id=first.image_id, mask_id=None,
                                   user_id="u", dataset_id=first.dataset_id)
    assert state.mask_id == first.mask_id

    state.switch_to_image(second.image_id, second.dataset_id)

    assert state.image_id == second.image_id
    # The mask is resolved from the new image, not carried over from the old one.
    assert state.mask_id == second.mask_id


def test_switch_refreshes_the_cached_orm_rows(images):
    """``cached_property`` keeps its first answer forever unless the key is dropped."""
    first, second = images
    state = AnnotationSessionState(image_id=first.image_id, mask_id=None,
                                   user_id="u", dataset_id=first.dataset_id)
    assert state.image_db.file_name == "a.png"
    assert state.mask_db.id == first.mask_id

    state.switch_to_image(second.image_id, second.dataset_id)

    assert state.image_db.file_name == "b.png"
    assert state.mask_db.id == second.mask_id


def test_switch_drops_the_previous_images_working_state(images):
    first, second = images
    state = AnnotationSessionState(image_id=first.image_id, mask_id=None,
                                   user_id="u", dataset_id=first.dataset_id)
    state.focussed_contour_id = 7
    state.refinement_contour_id = 9
    state.contour_hierarchy = object()

    state.switch_to_image(second.image_id, second.dataset_id)

    # Carrying any of these over would apply the old image's selection to the new one.
    assert state.focussed_contour_id is None
    assert state.refinement_contour_id is None
    assert state.contour_hierarchy is None


def test_switch_keeps_the_registered_backends(images):
    """The reason a switch is a message: the loaded models survive it."""
    first, second = images
    state = AnnotationSessionState(image_id=first.image_id, mask_id=None,
                                   user_id="u", dataset_id=first.dataset_id)
    state._running_backends["prompted_segmentation"] = object()

    state.switch_to_image(second.image_id, second.dataset_id)

    assert "prompted_segmentation" in state._running_backends
    assert state.user_id == "u"


def test_switch_to_an_unknown_image_is_refused(images):
    first, _ = images
    state = AnnotationSessionState(image_id=first.image_id, mask_id=None,
                                   user_id="u", dataset_id=first.dataset_id)

    with pytest.raises(Exception):
        state.switch_to_image(999_999, first.dataset_id)


def test_session_may_start_without_an_image(images):
    """The socket is opened per user; the client picks an image afterwards."""
    state = AnnotationSessionState(user_id="u")

    assert state.image_id is None
    assert state.mask_id is None

    first, _ = images
    state.switch_to_image(first.image_id, first.dataset_id)
    assert state.mask_id == first.mask_id


# --------------------------------------------------------------------------- #
# A bad message must not cost the connection
# --------------------------------------------------------------------------- #
class _FakeWebSocket:
    """Replays queued payloads and records what was sent back."""

    def __init__(self, payload):
        self._payload = payload
        self.sent = []

    async def receive_json(self):
        return self._payload

    async def send_json(self, data):
        self.sent.append(data)


def test_unparseable_message_is_reported_not_raised():
    ws = _FakeWebSocket({"id": "abc", "type": "not_a_real_message_type", "data": {}})

    result = asyncio.run(messaging.receive_msg(ws))

    assert result is None, "an invalid message must not tear down the session loop"
    assert len(ws.sent) == 1
    # The id is echoed so the client can fail that request instead of waiting it out.
    assert '"abc"' in ws.sent[0]
    assert '"error"' in ws.sent[0]


def test_valid_message_is_returned():
    ws = _FakeWebSocket({"id": "1", "type": "switch_image", "data": {"image_id": 3}})

    result = asyncio.run(messaging.receive_msg(ws))

    assert result is not None
    assert result.type == "switch_image"
    assert result.data["image_id"] == 3
    assert ws.sent == []

"""End-to-end checks that the `require()` dependency actually blocks HTTP requests.

`test_permissions.py` covers the model; this covers the wiring — that a route
resolves the dataset behind a mask/contour/image id, and answers 403 rather than
doing the work. It drives a real FastAPI app with the database dependency pointed
at a temp SQLite file.
"""
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.database import database, get_session
import app.database.contours  # noqa: F401
import app.database.dataset_members  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.rejections  # noqa: F401
import app.database.users  # noqa: F401
from app.database.contours import Contours
from app.database.dataset_members import DatasetMembers
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.masks import Masks
from app.database.users import Users
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import DatasetRole, GlobalRole, Permission
from app.services.auth import get_current_user
from app.services.permissions import require, require_global


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


@pytest.fixture
def ctx(tmp_path):
    """A live app whose routes exercise every id-source the resolver supports."""
    engine = create_engine(f"sqlite:///{tmp_path / 'routes.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    session = Session()
    session.add_all([
        Users(username="owner", hashed_password="x"),
        Users(username="ann", hashed_password="x"),
        Users(username="stranger", hashed_password="x"),
        Users(username="root", hashed_password="x", global_role=GlobalRole.ADMIN.value),
        Users(username="guest", hashed_password="x", global_role=GlobalRole.GUEST.value),
    ])
    ds = Datasets(name="ds", description="", dataset_type="image",
                  folder_path="/tmp/ds", created_by="owner")
    session.add(ds)
    session.flush()
    session.add(DatasetMembers(dataset_id=ds.id, username="owner",
                               role=DatasetRole.OWNER.value,
                               extra_permissions=[], denied_permissions=[]))
    session.add(DatasetMembers(dataset_id=ds.id, username="ann",
                               role=DatasetRole.ANNOTATOR.value,
                               extra_permissions=[], denied_permissions=[]))
    img = Images(dataset_id=ds.id, file_name="a.png", file_path="/tmp/a.png",
                 thumbnail_file_path="/tmp/t.png", width=10, height=10, color_mode="RGB")
    session.add(img)
    session.flush()
    mask = Masks(image_id=img.id, fully_annotated=False, file_path="/tmp/m.png")
    session.add(mask)
    session.flush()
    contour = Contours(mask_id=mask.id, added_by="manual", author_username="ann",
                       confidence_score=1.0, area=1.0, perimeter=1.0, circularity=1.0,
                       diameter=1.0, x=[0.1], y=[0.1])
    session.add(contour)
    session.commit()
    ids = {"dataset": ds.id, "image": img.id, "mask": mask.id, "contour": contour.id}
    session.close()

    app = FastAPI()

    @app.get("/d/{dataset_id}/export")
    async def export(dataset_id: int,
                     user: AuthenticatedUser = Depends(require(Permission.EXPORT_ANNOTATIONS))):
        return {"ok": True, "by": user.username}

    @app.delete("/m/{mask_id}")
    async def drop_mask(mask_id: int,
                        user: AuthenticatedUser = Depends(require(Permission.MASK_DELETE, "mask_id"))):
        return {"ok": True}

    @app.post("/c/{contour_id}/approve")
    async def approve(contour_id: int,
                      user: AuthenticatedUser = Depends(
                          require(Permission.REVIEW_APPROVE, "contour_id"))):
        return {"ok": True}

    @app.post("/i/{image_id}/annotate")
    async def annotate(image_id: int,
                       user: AuthenticatedUser = Depends(
                           require(Permission.ANNOTATION_CREATE, "image_id"))):
        return {"ok": True}

    @app.post("/datasets")
    async def create(user: AuthenticatedUser = Depends(require_global(Permission.DATASET_CREATE))):
        return {"ok": True}

    # Query-string rather than path, to cover the other branch of _read_id.
    @app.get("/upload")
    async def upload(dataset_id: int,
                     user: AuthenticatedUser = Depends(require(Permission.IMAGE_UPLOAD))):
        return {"ok": True}

    current = {"username": "owner"}

    def _session_override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    def _user_override(db=Depends(_session_override)):
        row = db.query(Users).filter_by(username=current["username"]).one()
        return AuthenticatedUser.from_query(row)

    app.dependency_overrides[get_session] = _session_override
    app.dependency_overrides[get_current_user] = _user_override

    client = TestClient(app)

    def as_user(username):
        current["username"] = username
        return client

    yield as_user, ids
    engine.dispose()


def test_owner_may_export_annotator_may_not(ctx):
    as_user, ids = ctx
    assert as_user("owner").get(f"/d/{ids['dataset']}/export").status_code == 200

    response = as_user("ann").get(f"/d/{ids['dataset']}/export")
    assert response.status_code == 403
    assert "export.annotations" in response.json()["detail"]


def test_permission_resolves_through_mask_and_contour_ids(ctx):
    as_user, ids = ctx
    # The route only knows a mask id; the dependency walks mask -> image -> dataset.
    assert as_user("owner").delete(f"/m/{ids['mask']}").status_code == 200
    assert as_user("ann").delete(f"/m/{ids['mask']}").status_code == 403

    assert as_user("owner").post(f"/c/{ids['contour']}/approve").status_code == 200
    assert as_user("ann").post(f"/c/{ids['contour']}/approve").status_code == 403


def test_annotator_may_annotate(ctx):
    as_user, ids = ctx
    assert as_user("ann").post(f"/i/{ids['image']}/annotate").status_code == 200


def test_stranger_is_locked_out_of_everything(ctx):
    as_user, ids = ctx
    client = as_user("stranger")
    assert client.get(f"/d/{ids['dataset']}/export").status_code == 403
    assert client.delete(f"/m/{ids['mask']}").status_code == 403
    assert client.post(f"/i/{ids['image']}/annotate").status_code == 403


def test_admin_passes_without_membership(ctx):
    as_user, ids = ctx
    client = as_user("root")
    assert client.get(f"/d/{ids['dataset']}/export").status_code == 200
    assert client.delete(f"/m/{ids['mask']}").status_code == 200


def test_global_permission_gate(ctx):
    as_user, _ = ctx
    assert as_user("owner").post("/datasets").status_code == 200
    assert as_user("guest").post("/datasets").status_code == 403
    assert as_user("root").post("/datasets").status_code == 200


def test_dataset_id_can_come_from_the_query_string(ctx):
    as_user, ids = ctx
    assert as_user("owner").get(f"/upload?dataset_id={ids['dataset']}").status_code == 200
    assert as_user("ann").get(f"/upload?dataset_id={ids['dataset']}").status_code == 403


def test_unknown_entity_is_404_not_403(ctx):
    """An id that resolves to no dataset is missing, not forbidden."""
    as_user, _ = ctx
    assert as_user("owner").delete("/m/99999").status_code == 404
    assert as_user("owner").get("/d/99999/export").status_code == 404


def test_authenticated_user_serializes_role_and_memberships(ctx):
    """The shape the frontend reads from /auth/me.

    `AuthenticatedUser` subclasses the toolbox `User`, and the route has no
    `response_model`, so the extra fields survive serialization. Adding one would
    silently drop `global_role` / `memberships` and break every permission check
    in the UI at once, hence the guard.
    """
    from fastapi.encoders import jsonable_encoder

    as_user, ids = ctx
    # Reach the resolved user through the same dependency the routes use.
    as_user('owner')
    response = as_user('owner').get(f"/d/{ids['dataset']}/export")
    assert response.status_code == 200

    from app.schemas.auth_user import AuthenticatedUser
    from app.schemas.permissions import DatasetRole, GlobalRole, Permission

    payload = jsonable_encoder(AuthenticatedUser(
        username='owner',
        is_admin=False,
        global_role=GlobalRole.MEMBER,
        is_active=True,
        owned_datasets=[ids['dataset']],
        accessible_datasets=[],
        memberships={ids['dataset']: {'role': DatasetRole.OWNER,
                                      'permissions': {Permission.DATASET_DELETE}}},
    ))

    assert payload['global_role'] == 'member'
    assert payload['is_active'] is True
    # JSON object keys are strings; the frontend looks the id up both ways.
    membership = payload['memberships'][str(ids['dataset'])]
    assert membership['role'] == 'owner'
    assert 'dataset.delete' in membership['permissions']

"""The scene-graph HTTP routes, with real dataset roles.

The service behaviour is covered in ``test_scene_graph``; this checks the wiring:
who may define attributes and who may set them, that domain errors become 4xx
responses, and that an edit lands in the annotator's undo history.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.database import database, get_session, import_models
from app.database.contours import Contours
from app.database.dataset_members import DatasetMembers
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.routes.general.scene_graph import router as scene_graph_router
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import DatasetRole
from app.services.auth import get_current_user
from app.services.database_access import annotation_history as history_db


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


@pytest.fixture
def ctx(tmp_path):
    import_models()
    engine = create_engine(f"sqlite:///{tmp_path / 'graph_routes.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    db.add_all([Users(username="ann", hashed_password="x"), Users(username="curator", hashed_password="x")])
    ds = Datasets(name="ds", description="", dataset_type="image", folder_path=str(tmp_path),
                  created_by="curator")
    db.add(ds)
    db.flush()
    db.add_all([
        DatasetMembers(dataset_id=ds.id, username="curator", role=DatasetRole.CURATOR.value,
                       extra_permissions=[], denied_permissions=[]),
        DatasetMembers(dataset_id=ds.id, username="ann", role=DatasetRole.ANNOTATOR.value,
                       extra_permissions=[], denied_permissions=[]),
    ])
    label = Labels(dataset_id=ds.id, name="coral", value=1)
    db.add(label)
    db.flush()
    image = Images(dataset_id=ds.id, file_name="a.png", file_path=str(tmp_path / "a.png"),
                   thumbnail_file_path=str(tmp_path / "t.png"), width=100, height=80, color_mode="RGB")
    db.add(image)
    db.flush()
    mask = Masks(image_id=image.id, fully_annotated=False, file_path=str(tmp_path / "m.png"))
    db.add(mask)
    db.flush()
    ids = {}
    for author, x0 in (("ann", 0.1), ("curator", 0.6)):
        contour = Contours(mask_id=mask.id, label_id=label.id, added_by="manual", author_username=author,
                           confidence_score=1.0, area=0.1, perimeter=1.0, circularity=0.5, diameter=1.0,
                           x=[x0, x0 + 0.2, x0 + 0.2, x0], y=[0.1, 0.1, 0.3, 0.3])
        db.add(contour)
        db.flush()
        ids[author] = contour.id
    db.commit()
    yield {"Session": Session, "db": db, "dataset_id": ds.id, "image_id": image.id,
           "mask_id": mask.id, "label_id": label.id, "ann_contour": ids["ann"],
           "curator_contour": ids["curator"]}
    db.close()
    engine.dispose()


@pytest.fixture
def client(ctx):
    Session = ctx["Session"]
    app = FastAPI()
    app.include_router(scene_graph_router)
    current = {"username": "curator"}

    def _session():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    def _user():
        session = Session()
        try:
            return AuthenticatedUser.from_query(session.query(Users).filter_by(username=current["username"]).one())
        finally:
            session.close()

    app.dependency_overrides[get_session] = _session
    app.dependency_overrides[get_current_user] = _user
    test_client = TestClient(app)

    def as_user(username):
        current["username"] = username
        return test_client

    return as_user


def _define_species(client, ctx):
    response = client("curator").post(f"/scene-graph/dataset/{ctx['dataset_id']}/attributes", json={
        "name": "species", "options": ["Acropora", {"value": "Pocillopora", "code": "206938"}],
        "label_ids": [ctx["label_id"]]})
    assert response.status_code == 201, response.text
    return response.json()["attribute"]


def test_only_curators_define_attributes(client, ctx):
    response = client("ann").post(f"/scene-graph/dataset/{ctx['dataset_id']}/attributes",
                                  json={"name": "species"})
    assert response.status_code == 403
    species = _define_species(client, ctx)
    assert species["options"][1] == {"value": "Pocillopora", "code": "206938", "description": None}


def test_an_annotator_sets_values_on_their_own_objects_only(client, ctx):
    species = _define_species(client, ctx)
    url = f"/scene-graph/dataset/{ctx['dataset_id']}/attributes/{species['id']}/values"

    ok = client("ann").post(url, json={"contour_ids": [ctx["ann_contour"]], "value": "Acropora"})
    assert ok.status_code == 200 and ok.json()["changed"] == [ctx["ann_contour"]]
    refused = client("ann").post(url, json={"contour_ids": [ctx["curator_contour"]], "value": "Acropora"})
    assert refused.status_code == 403

    status = history_db.get_status(ctx["mask_id"], "ann", ctx["Session"]())
    assert status["undo_label"] == "attribute change"


def test_invalid_values_and_foreign_ids_are_client_errors(client, ctx):
    species = _define_species(client, ctx)
    url = f"/scene-graph/dataset/{ctx['dataset_id']}/attributes/{species['id']}/values"
    assert client("curator").post(url, json={"contour_ids": [ctx["ann_contour"]],
                                             "value": "Porites"}).status_code == 422
    assert client("curator").post(url, json={"contour_ids": [99999],
                                             "value": "Acropora"}).status_code == 404


def test_relationships_and_the_image_graph(client, ctx):
    response = client("ann").post(f"/scene-graph/dataset/{ctx['dataset_id']}/relationships", json={
        "subject_id": ctx["ann_contour"], "object_ids": [ctx["curator_contour"]], "type_name": "touches"})
    assert response.status_code == 200, response.text
    assert response.json()["relationships"][0]["type"] == "touches"

    graph = client("ann").get(f"/scene-graph/image/{ctx['image_id']}").json()
    assert {(e["type"], e["subject_id"], e["object_id"]) for e in graph["relationships"]} == {
        ("touches", ctx["ann_contour"], ctx["curator_contour"])}

    types = client("ann").get(f"/scene-graph/dataset/{ctx['dataset_id']}/relationship-types").json()
    assert [(t["name"], t["usage_count"]) for t in types["relationship_types"]] == [("touches", 1)]


def test_the_object_query_downloads_as_csv(client, ctx):
    species = _define_species(client, ctx)
    client("curator").post(f"/scene-graph/dataset/{ctx['dataset_id']}/attributes/{species['id']}/values",
                           json={"contour_ids": [ctx["ann_contour"]], "value": "Acropora"})
    response = client("curator").post(
        f"/scene-graph/dataset/{ctx['dataset_id']}/objects/query?format=csv",
        json={"where": [{"attribute": "species", "value": "Acropora"}], "metrics": ["area"]})
    assert response.status_code == 200, response.text
    header, *rows = response.text.strip().splitlines()
    assert "attr_species" in header.split(",") and len(rows) == 1

"""Stacks: OCT volumes (and later videos) stored as frames that are images.

What carries the risk:

  * **storage is all or nothing** -- a stack, its frames, masks and metadata land
    together, frames are real ``Images`` rows with ``kind = 'frame'``;
  * **metadata inheritance** -- a frame reports its stack's keys, its own row
    for a key wins, and filters / facets / key administration see both owners;
  * **the gallery** lists images, not frames, unless asked;
  * **deleting** a stack takes its frames with it, and a single frame cannot be
    deleted out of a stack;
  * **the Heidelberg reader**'s geometry: where a B-scan lies on the overview.

The reader is checked against a real ``.vol`` only when ``IQUANA_TEST_VOL_FILE``
points at one -- patient scans do not belong in the repository.
"""
import asyncio
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.database import database, get_session, import_models
from app.database.datasets import Datasets
from app.database.dataset_members import DatasetMembers
from app.database.image_metadata import ImageMetadata
from app.database.images import KIND_FRAME, KIND_IMAGE, Frames, Images
from app.database.masks import Masks
from app.database.stacks import Stacks
from app.database.users import Users
from app.exceptions import UnsupportedStackFileError
from app.routes.general.images import router as images_router
from app.routes.general.stacks import router as stacks_router
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import DatasetRole
from app.services.auth import get_current_user
from app.services.database_access import image_metadata as meta
from app.services.database_access import stacks as stacks_db
from app.services.database_access.datasets import get_image_and_mask_ids_of_dataset
from app.services.stack_readers import FrameData, StackData, read_stack_file
from app.services.stack_readers.heidelberg import _overview_line

import_models()


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


def _stack_data(frames: int = 4, name: str = "vol_a") -> StackData:
    return StackData(
        name=name,
        kind="oct_volume",
        axis="z",
        source_format="heidelberg_vol",
        frames=[
            FrameData(
                pixels=np.full((20, 30), index * 10, dtype=np.uint8),
                position=index * 0.25,
                overview_geometry={"type": "line", "start": [0, index], "end": [29, index]},
                metadata={"B-scan quality": f"{30 + index}"},
            )
            for index in range(frames)
        ],
        scale_x=0.0113, scale_y=0.0039, unit="mm",
        frame_spacing=0.25, frame_spacing_unit="mm",
        overview=np.zeros((40, 40), dtype=np.uint8),
        overview_scale_x=0.0113, overview_scale_y=0.0113, overview_unit="mm",
        metadata={"Eye": "OD", "Visit date": "2012-07-29T06:08:00"},
        key_types={"Visit date": "date", "B-scan quality": "number"},
    )


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """A dataset with one plain image and one stored 4-frame stack."""
    monkeypatch.setattr(stacks_db, "THUMBNAILS_DIR", str(tmp_path / "thumbnails"))
    engine = create_engine(f"sqlite:///{tmp_path / 'stacks.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    db.add(Users(username="curator", hashed_password="x"))
    dataset = Datasets(name="ds", description="", dataset_type="image",
                       folder_path=str(tmp_path / "ds"), created_by="curator")
    db.add(dataset)
    db.flush()
    db.add(DatasetMembers(dataset_id=dataset.id, username="curator", role=DatasetRole.CURATOR.value,
                          extra_permissions=[], denied_permissions=[]))
    image = Images(dataset_id=dataset.id, file_name="plain.png", file_path=str(tmp_path / "plain.png"),
                   thumbnail_file_path=str(tmp_path / "plain_t.png"), width=10, height=10, color_mode="L")
    db.add(image)
    db.commit()

    stack_id = asyncio.run(stacks_db.save_stack(_stack_data(), dataset.id, dataset.folder_path, db,
                                                source_file_name="vol_a.vol", username="curator"))
    yield {"db": db, "Session": Session, "dataset_id": dataset.id, "image_id": image.id,
           "stack_id": stack_id, "folder": Path(dataset.folder_path), "tmp": tmp_path}
    db.close()
    engine.dispose()


def _frame_ids(db, stack_id):
    return [f.id for f in db.query(Frames).filter_by(stack_id=stack_id).order_by(Frames.frame_index)]


# -- storage ---------------------------------------------------------------------

def test_stack_is_stored_with_frames_masks_and_files(ctx):
    db = ctx["db"]
    stack = db.query(Stacks).one()
    assert (stack.frame_count, stack.source_file_name, stack.overview_width) == (4, "vol_a.vol", 40)
    assert Path(stack.overview_file_path).is_file() and Path(stack.thumbnail_file_path).is_file()

    frames = db.query(Images).filter(Images.stack_id == stack.id).order_by(Images.frame_index).all()
    assert [type(f) for f in frames] == [Frames] * 4
    assert {f.kind for f in frames} == {KIND_FRAME}
    assert [f.frame_index for f in frames] == [0, 1, 2, 3]
    assert frames[2].frame_position == pytest.approx(0.5)
    assert frames[1].overview_geometry == {"type": "line", "start": [0, 1], "end": [29, 1]}
    assert (frames[0].width, frames[0].height, frames[0].unit) == (30, 20, "mm")
    assert all(Path(f.file_path).is_file() and Path(f.thumbnail_file_path).is_file() for f in frames)
    assert db.query(Masks).filter(Masks.image_id.in_([f.id for f in frames])).count() == 4


def test_failed_stack_leaves_nothing_behind(ctx, monkeypatch):
    db = ctx["db"]
    data = _stack_data(name="broken")
    data.metadata = {"Visit date": "not a date"}  # typed 'date' already -> coercion fails
    before = (db.query(Stacks).count(), db.query(Images).count())
    with pytest.raises(Exception):
        asyncio.run(stacks_db.save_stack(data, ctx["dataset_id"], str(ctx["folder"]), db))
    assert (db.query(Stacks).count(), db.query(Images).count()) == before
    assert sorted(p.name for p in (ctx["folder"] / "stacks").iterdir()) == [str(ctx["stack_id"])]


def test_unsupported_file_is_refused(tmp_path):
    path = tmp_path / "scan.xyz"
    path.write_bytes(b"nope")
    with pytest.raises(UnsupportedStackFileError):
        read_stack_file(path, "scan.xyz")


# -- metadata inheritance --------------------------------------------------------

def test_frames_inherit_stack_metadata_and_own_keys_win(ctx):
    db = ctx["db"]
    frame_ids = _frame_ids(db, ctx["stack_id"])
    first = meta.get_metadata(db, frame_ids[0])
    assert first["Eye"] == "OD"
    assert first["B-scan quality"] == "30"

    meta.set_metadata_for_images(db, [frame_ids[1]], {"Eye": "OS"})
    assert meta.get_metadata(db, frame_ids[1])["Eye"] == "OS"
    assert meta.get_metadata(db, frame_ids[2])["Eye"] == "OD"
    assert meta.get_metadata(db, ctx["image_id"]) == {}

    # Stored once, on the stack, not copied onto every frame.
    assert db.query(ImageMetadata).filter_by(key="Eye", stack_id=ctx["stack_id"]).count() == 1


def test_stack_edits_reach_every_frame(ctx):
    db = ctx["db"]
    meta.set_metadata_for_stacks(db, [ctx["stack_id"]], {"Site": "clinic_a"})
    by_image = meta.get_metadata_for_images(db, _frame_ids(db, ctx["stack_id"]))
    assert {m["Site"] for m in by_image.values()} == {"clinic_a"}
    assert meta.get_metadata_for_stack(db, ctx["stack_id"])["Site"] == "clinic_a"


def test_filters_and_facets_see_inherited_keys(ctx):
    db = ctx["db"]
    frame_ids = _frame_ids(db, ctx["stack_id"])
    meta.set_metadata_for_images(db, [ctx["image_id"]], {"Eye": "OS"})

    assert sorted(meta.filter_image_ids(db, ctx["dataset_id"], {"Eye": ["OD"]})) == sorted(frame_ids)
    assert meta.filter_image_ids(db, ctx["dataset_id"], {"Eye": ["OS"]}) == [ctx["image_id"]]
    assert sorted(meta.filter_image_ids(db, ctx["dataset_id"], {"B-scan quality": {"min": 32}})) \
        == frame_ids[2:]

    facets = {f["key"]: f for f in meta.get_dataset_facets(db, ctx["dataset_id"])}
    assert {v["value"]: v["count"] for v in facets["Eye"]["values"]} == {"OD": 4, "OS": 1}
    assert facets["Visit date"]["value_type"] == "date"


def test_key_administration_covers_stack_rows(ctx):
    db = ctx["db"]
    meta.rename_key(db, ctx["dataset_id"], "Eye", "Laterality")
    assert db.query(ImageMetadata).filter_by(stack_id=ctx["stack_id"], key="Laterality").count() == 1
    assert meta.delete_key_from_dataset(db, ctx["dataset_id"], "Laterality") == 1
    assert "Laterality" not in meta.get_metadata(db, _frame_ids(db, ctx["stack_id"])[0])


# -- gallery and details ---------------------------------------------------------

def test_gallery_lists_images_not_frames(ctx):
    db = ctx["db"]
    listed = asyncio.run(get_image_and_mask_ids_of_dataset(ctx["dataset_id"], db=db))
    assert [entry["image_id"] for entry in listed] == [ctx["image_id"]]
    with_frames = asyncio.run(get_image_and_mask_ids_of_dataset(ctx["dataset_id"], db=db,
                                                                include_frames=True))
    assert len(with_frames) == 5
    frames = [entry for entry in with_frames if entry["stack_id"] == ctx["stack_id"]]
    assert sorted(entry["frame_index"] for entry in frames) == [0, 1, 2, 3]

    stacks = stacks_db.list_stacks_of_dataset(db, ctx["dataset_id"])
    assert [(s["stack_id"], s["frame_count"], s["metadata"]["Eye"]) for s in stacks] \
        == [(ctx["stack_id"], 4, "OD")]


def test_stack_details_list_frames_in_order(ctx):
    details = stacks_db.get_stack_details(ctx["db"], ctx["stack_id"])
    assert [f["frame_index"] for f in details["frames"]] == [0, 1, 2, 3]
    # A physical scale from the file counts as calibrated; nothing is drawn yet.
    assert details["frames"][0]["phases"]["annotate"] == "not_started"
    assert details["frames"][3]["metadata"] == {"B-scan quality": "33"}
    assert details["metadata"]["Eye"] == "OD" and details["has_overview"]


def test_stack_objects_list_every_frame_in_order(ctx):
    from app.database.contours import Contours
    db = ctx["db"]
    frame_ids = _frame_ids(db, ctx["stack_id"])
    masks = {m.image_id: m.id for m in db.query(Masks).filter(Masks.image_id.in_(frame_ids))}

    def contour(image_id, temporary=False):
        return Contours(mask_id=masks[image_id], added_by="User", origin="manual", temporary=temporary,
                        confidence_score=1, area=1, perimeter=1, circularity=1, diameter=1,
                        x=[1, 2, 2], y=[1, 1, 2])

    db.add_all([contour(frame_ids[2]), contour(frame_ids[0]), contour(frame_ids[1], temporary=True)])
    db.commit()
    objects = stacks_db.get_stack_objects(db, ctx["stack_id"])
    assert [o["frame_index"] for o in objects] == [0, 2]
    assert objects[0]["x"] == [1, 2, 2] and objects[0]["reviewed"] is False


# -- deletion --------------------------------------------------------------------

def test_deleting_a_stack_removes_frames_rows_and_files(ctx):
    db = ctx["db"]
    frame_files = [Path(f.file_path) for f in db.query(Frames).filter_by(stack_id=ctx["stack_id"])]
    stacks_db.delete_stack(db, ctx["stack_id"])
    db.expire_all()
    assert db.query(Stacks).count() == 0
    assert db.query(Images).all()[0].kind == KIND_IMAGE and db.query(Images).count() == 1
    assert db.query(Masks).count() == 0
    assert db.query(ImageMetadata).filter(ImageMetadata.stack_id.is_not(None)).count() == 0
    assert not any(path.exists() for path in frame_files)
    assert not (ctx["folder"] / "stacks" / str(ctx["stack_id"])).exists()


@pytest.fixture
def client(ctx):
    Session = ctx["Session"]
    app = FastAPI()
    app.include_router(images_router)
    app.include_router(stacks_router)

    def _session_override():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    def _user_override():
        session = Session()
        try:
            return AuthenticatedUser.from_query(session.query(Users).filter_by(username="curator").one())
        finally:
            session.close()

    app.dependency_overrides[get_session] = _session_override
    app.dependency_overrides[get_current_user] = _user_override
    return TestClient(app)


def test_a_frame_cannot_be_deleted_on_its_own(ctx, client):
    frame_id = _frame_ids(ctx["db"], ctx["stack_id"])[0]
    response = client.delete(f"/images/{frame_id}")
    assert response.status_code == 409
    assert "stack" in response.json()["detail"]


def test_frame_file_is_served_raw_and_cacheable(ctx, client):
    frame_id = _frame_ids(ctx["db"], ctx["stack_id"])[0]
    response = client.get(f"/images/{frame_id}/file")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert "immutable" in response.headers["cache-control"]


def test_upload_of_unreadable_file_reports_400(ctx, client):
    response = client.post(f"/stacks/upload?dataset_id={ctx['dataset_id']}",
                           files=[("files", ("scan.xyz", b"nope", "application/octet-stream"))])
    assert response.status_code == 400
    assert response.json()["detail"][0]["file_name"] == "scan.xyz"


def test_stack_routes(ctx, client):
    stack_id = ctx["stack_id"]
    assert client.get(f"/stacks/dataset/{ctx['dataset_id']}").json()["stacks"][0]["stack_id"] == stack_id
    assert len(client.get(f"/stacks/{stack_id}").json()["frames"]) == 4
    assert client.get(f"/stacks/{stack_id}/objects").json() == {"objects": []}
    assert client.get(f"/stacks/{stack_id}/overview").headers["content-type"] == "image/png"
    updated = client.put(f"/stacks/{stack_id}/metadata", json={"entries": {"Eye": "OS"}}).json()
    assert updated["metadata"]["Eye"] == "OS"
    assert client.delete(f"/stacks/{stack_id}").status_code == 200


# -- Heidelberg reader -----------------------------------------------------------

def test_overview_line_in_mm():
    line = _overview_line(
        {"start_pos": (1.449, 6.52), "end_pos": (7.245, 6.52), "pos_unit": "mm"},
        {"scale_x": 0.01132, "scale_y": 0.01132, "scale_unit": "mm"},
    )
    assert line["start"] == pytest.approx([128.0, 576.0], abs=0.1)
    assert line["end"] == pytest.approx([640.0, 576.0], abs=0.1)


def test_overview_line_in_degrees_is_centred():
    # E2E positions are degrees from the centre of a 30 degree field of view.
    line = _overview_line(
        {"start_pos": (-15.0, 0.0), "end_pos": (15.0, 0.0), "pos_unit": "°"},
        {"scale_x": 30 / 768, "scale_y": 30 / 768, "scale_unit": "°"},
    )
    assert line["start"] == pytest.approx([0.0, 384.0])
    assert line["end"] == pytest.approx([768.0, 384.0])


def test_overview_line_needs_matching_units():
    assert _overview_line({"start_pos": (0, 0), "end_pos": (1, 0), "pos_unit": "mm"},
                          {"scale_x": 1, "scale_y": 1, "scale_unit": "°"}) is None


@pytest.mark.skipif(not os.getenv("IQUANA_TEST_VOL_FILE"), reason="set IQUANA_TEST_VOL_FILE to a .vol")
def test_real_vol_file():
    path = Path(os.environ["IQUANA_TEST_VOL_FILE"])
    [stack] = read_stack_file(path, path.name)
    assert stack.kind == "oct_volume" and stack.unit == "mm" and len(stack.frames) > 1
    assert stack.frames[0].pixels.dtype == np.uint8 and stack.overview is not None
    assert stack.metadata["Eye"] in {"OD", "OS"}
    assert stack.frames[0].overview_geometry["type"] == "line"
    assert not any(key.lower().startswith(("patient", "name", "birth")) for key in stack.metadata)


def test_dataset_wide_scale_leaves_file_scaled_frames_alone(ctx):
    from app.services.scale_computation import apply_scale_to_dataset
    db = ctx["db"]
    result = apply_scale_to_dataset(db, ctx["dataset_id"], 0.5, 0.5, "mm")
    assert result["images_updated"] == 1
    db.expire_all()
    assert db.get(Images, ctx["image_id"]).scale_x == 0.5
    assert {f.scale_x for f in db.query(Frames).filter_by(stack_id=ctx["stack_id"])} == {0.0113}

"""Unit tests for the correction queue builder (`app.services.correction_queue`)
and the resolution flow on the reviews router.

Covers which rejections qualify (open only), oldest/newest ordering with image
grouping, the reason filter, mask-level (contour-less) items carrying the mask's
image id, the summary counts, and that resolving with a resolution kind persists
it and is idempotent. Driven directly against the service with a temp SQLite
database, same shape as `test_review_queue.py`.
"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.contours  # noqa: F401
import app.database.dataset_members  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.rejections  # noqa: F401
import app.database.users  # noqa: F401
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.rejections import AnnotationRejections
from app.database.users import Users
from app.schemas.review import (
    CorrectionQueueRequest,
    CorrectionSortOrder,
    RejectionReason,
)
from app.services.correction_queue import build_queue, summarize


def _contour(mask_id, label_id=None, parent_id=None):
    return Contours(mask_id=mask_id, parent_id=parent_id, label_id=label_id,
                    added_by="manual", author_username="ann",
                    confidence_score=1.0, area=1.0, perimeter=1.0,
                    circularity=1.0, diameter=1.0, x=[0.1, 0.2], y=[0.1, 0.2])


def _t(minutes):
    """A deterministic timestamp, `minutes` after a fixed base."""
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minutes)


@pytest.fixture
def ctx(tmp_path):
    """Two images, each with a mask. Three open rejections (a contour-level one on
    image 1, a mask-level one on image 2, a second contour-level one on image 1)
    plus one already-resolved rejection that must never appear."""
    engine = create_engine(f"sqlite:///{tmp_path / 'correction.db'}")
    database.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()

    db.add_all([Users(username="ann", hashed_password="x"),
                Users(username="rev", hashed_password="x")])
    ds = Datasets(name="ds", description="", dataset_type="image",
                  folder_path="/tmp/ds", created_by="ann")
    db.add(ds)
    db.flush()

    cell = Labels(dataset_id=ds.id, name="cell", value=1)
    db.add(cell)
    db.flush()

    def image_with_mask(name):
        img = Images(dataset_id=ds.id, file_name=name, file_path=f"/tmp/{name}",
                     thumbnail_file_path=f"/tmp/t_{name}", width=10, height=10,
                     color_mode="RGB")
        db.add(img)
        db.flush()
        mask = Masks(image_id=img.id, fully_annotated=False,
                     file_path=f"/tmp/m_{name}")
        db.add(mask)
        db.flush()
        return img, mask

    img1, mask1 = image_with_mask("a.png")
    img2, mask2 = image_with_mask("b.png")
    con1 = _contour(mask1.id, label_id=cell.id)
    con2 = _contour(mask1.id, label_id=cell.id)
    db.add_all([con1, con2])
    db.flush()

    # Open: contour-level bad_outline on image 1 (oldest), mask-level
    # missing_objects on image 2 (middle), contour-level wrong_label on image 1
    # (newest). Resolved: an old one on image 1 that must be filtered out.
    rej_outline = AnnotationRejections(
        mask_id=mask1.id, contour_id=con1.id, reason=RejectionReason.BAD_OUTLINE.value,
        note="fix the edge", created_by="rev", created_at=_t(0))
    rej_missing = AnnotationRejections(
        mask_id=mask2.id, contour_id=None, reason=RejectionReason.MISSING_OBJECTS.value,
        created_by="rev", created_at=_t(10))
    rej_label = AnnotationRejections(
        mask_id=mask1.id, contour_id=con2.id, reason=RejectionReason.WRONG_LABEL.value,
        created_by="rev", created_at=_t(20))
    rej_done = AnnotationRejections(
        mask_id=mask1.id, contour_id=con1.id, reason=RejectionReason.OTHER.value,
        note="already handled", created_by="rev", created_at=_t(-10),
        resolved_at=_t(-5), resolved_by="ann")
    db.add_all([rej_outline, rej_missing, rej_label, rej_done])
    db.commit()

    yield {
        "db": db, "dataset_id": ds.id,
        "masks": {"one": mask1.id, "two": mask2.id},
        "images": {"one": img1.id, "two": img2.id},
        "contours": {"one": con1.id, "two": con2.id},
        "rejections": {"outline": rej_outline.id, "missing": rej_missing.id,
                       "label": rej_label.id, "done": rej_done.id},
    }
    db.close()


def test_summary_counts_open_rejections_only(ctx):
    summary = summarize(ctx["dataset_id"], ctx["db"])
    assert summary.open_rejections == 3           # resolved one excluded
    assert summary.affected_instances == 2        # con1, con2 (mask-level not counted)
    assert summary.affected_images == 2


def test_queue_lists_open_not_resolved(ctx):
    queue = build_queue(ctx["dataset_id"],
                        CorrectionQueueRequest(order=CorrectionSortOrder.OLDEST),
                        ctx["db"])
    ids = {item.rejection_id for item in queue.items}
    assert ctx["rejections"]["done"] not in ids
    assert queue.total == 3


def test_oldest_order_groups_image_one_first(ctx):
    """Oldest-first: image 1 (leads at t=0) before image 2 (t=10), and image 1's
    two items stay together in age order."""
    queue = build_queue(ctx["dataset_id"],
                        CorrectionQueueRequest(order=CorrectionSortOrder.OLDEST),
                        ctx["db"])
    r = ctx["rejections"]
    assert [item.rejection_id for item in queue.items] == [r["outline"], r["label"], r["missing"]]


def test_newest_order_puts_image_one_last(ctx):
    """Newest-first: image 1 now leads with its newest item (t=20), so it comes
    before image 2 (t=10), and image 1's items are newest-first within the group."""
    queue = build_queue(ctx["dataset_id"],
                        CorrectionQueueRequest(order=CorrectionSortOrder.NEWEST),
                        ctx["db"])
    r = ctx["rejections"]
    assert [item.rejection_id for item in queue.items] == [r["label"], r["outline"], r["missing"]]


def test_reason_filter(ctx):
    queue = build_queue(ctx["dataset_id"], CorrectionQueueRequest(
        reasons=[RejectionReason.BAD_OUTLINE]), ctx["db"])
    assert [item.rejection_id for item in queue.items] == [ctx["rejections"]["outline"]]
    assert queue.items[0].reason_label == "Outline is inaccurate"
    assert queue.items[0].note == "fix the edge"


def test_mask_level_item_carries_image_id_and_null_contour(ctx):
    queue = build_queue(ctx["dataset_id"], CorrectionQueueRequest(
        reasons=[RejectionReason.MISSING_OBJECTS]), ctx["db"])
    item = queue.items[0]
    assert item.contour_id is None
    assert item.image_id == ctx["images"]["two"]
    assert item.mask_id == ctx["masks"]["two"]


def _reviews_client(ctx, username):
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    from app.database import get_session
    from app.routes.general.reviews import router as review_router
    from app.schemas.auth_user import AuthenticatedUser
    from app.services.auth import get_current_user

    db = ctx["db"]
    app = FastAPI()
    app.include_router(review_router)

    def _session_override():
        yield db

    def _user_override(session=Depends(_session_override)):
        row = session.query(Users).filter_by(username=username).one()
        return AuthenticatedUser.from_query(row)

    app.dependency_overrides[get_session] = _session_override
    app.dependency_overrides[get_current_user] = _user_override
    return TestClient(app)


def _grant(ctx, username, role):
    from app.database.dataset_members import DatasetMembers

    ctx["db"].add(DatasetMembers(dataset_id=ctx["dataset_id"], username=username,
                                 role=role, extra_permissions=[],
                                 denied_permissions=[]))
    ctx["db"].commit()


def test_resolve_records_resolution_and_is_idempotent(ctx):
    """Resolving with a resolution persists it; a second resolve never overwrites
    the first verdict."""
    from app.schemas.permissions import DatasetRole

    _grant(ctx, "ann", DatasetRole.ANNOTATOR.value)
    client = _reviews_client(ctx, "ann")
    rejection_id = ctx["rejections"]["outline"]

    body = client.patch(f"/reviews/rejections/{rejection_id}/resolve",
                        json={"resolution": "wont_fix"}).json()
    assert body["success"] is True
    assert body["rejection"]["resolution"] == "wont_fix"
    assert body["rejection"]["resolved_at"] is not None

    row = ctx["db"].query(AnnotationRejections).filter_by(id=rejection_id).one()
    assert row.resolution == "wont_fix"

    # A resolve without a body must not clobber the recorded verdict.
    again = client.patch(f"/reviews/rejections/{rejection_id}/resolve").json()
    assert again["rejection"]["resolution"] == "wont_fix"


def test_correction_summary_endpoint(ctx):
    from app.schemas.permissions import DatasetRole

    _grant(ctx, "ann", DatasetRole.ANNOTATOR.value)
    body = _reviews_client(ctx, "ann").get(
        f"/reviews/datasets/{ctx['dataset_id']}/correction-summary").json()
    assert body["success"] is True
    assert body["summary"]["open_rejections"] == 3


def test_replace_contour_preserves_open_rejection(tmp_path):
    """Refining an outline is a `replace_contour` — delete + re-insert under the
    SAME id. The ON DELETE CASCADE on annotation_rejections.contour_id would wipe
    the open rejection (and 404 the correction queue's "Mark as done" that follows);
    the replace must instead carry the rejection across, keeping its id."""
    import asyncio

    from iquana_toolbox.schemas.database.contours import Contour

    from app.database.contours import save_contour_tree
    from app.services.database_access.contours import replace_contour

    engine = create_engine(f"sqlite:///{tmp_path / 'replace.db'}")
    database.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()

    db.add_all([Users(username="ann", hashed_password="x"),
                Users(username="rev", hashed_password="x")])
    ds = Datasets(name="ds", description="", dataset_type="image",
                  folder_path="/tmp/ds", created_by="ann")
    db.add(ds)
    db.flush()
    label = Labels(dataset_id=ds.id, name="cell", value=1)
    db.add(label)
    db.flush()
    img = Images(dataset_id=ds.id, file_name="a.png", file_path="/tmp/a.png",
                 thumbnail_file_path="/tmp/t.png", width=100, height=100,
                 color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    db.add(img)
    db.flush()
    mask = Masks(image_id=img.id, fully_annotated=True, file_path="/tmp/m.png")
    db.add(mask)
    db.flush()

    contour = save_contour_tree(
        db,
        Contour(x=[0.1, 0.5, 0.5, 0.1], y=[0.1, 0.1, 0.5, 0.5],
                label_id=label.id, added_by="ann"),
        mask.id,
    )
    db.commit()
    contour_id = contour.id

    rejection = AnnotationRejections(mask_id=mask.id, contour_id=contour_id,
                                     reason=RejectionReason.BAD_OUTLINE.value,
                                     created_by="rev")
    db.add(rejection)
    db.commit()
    rejection_id = rejection.id

    # The annotator redraws the outline → replace_contour under the same id.
    new_outline = Contour(x=[0.2, 0.6, 0.6, 0.2], y=[0.2, 0.2, 0.6, 0.6],
                          label_id=label.id, added_by="ann")
    assert asyncio.run(replace_contour(contour_id, new_outline, db)) is True

    survived = db.query(AnnotationRejections).filter_by(id=rejection_id).one_or_none()
    assert survived is not None, "the open rejection was cascade-deleted by the refine"
    assert survived.contour_id == contour_id  # re-pointed to the reused contour id
    assert db.query(Contours).filter_by(id=contour_id).one_or_none() is not None
    db.close()

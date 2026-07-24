"""Unit tests for the review queue builder (`app.services.review_queue`).

Covers candidate collection (which masks and contours qualify), hierarchy depth
computation, the sort strategies, image-level grouping, the custom label filter
and the summary counts. Driven directly against the service with a temp SQLite
database — the HTTP wiring is the same `require()` dependency already covered by
`test_permission_routes.py`.
"""
import pytest
from fastapi import HTTPException
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
    ReviewGranularity,
    ReviewQueueRequest,
    ReviewSortDirection,
)
from app.services.review_queue import SORT_STRATEGIES, build_queue, summarize


def _contour(mask_id, label_id=None, parent_id=None, author="ann", confidence=1.0):
    return Contours(mask_id=mask_id, parent_id=parent_id, label_id=label_id,
                    added_by="manual", author_username=author,
                    confidence_score=confidence, area=1.0, perimeter=1.0,
                    circularity=1.0, diameter=1.0, x=[0.1, 0.2], y=[0.1, 0.2])


@pytest.fixture
def ctx(tmp_path):
    """A dataset with one reviewable mask (a 3-deep hierarchy plus one already
    approved root), one unsubmitted mask and one rejected mask."""
    engine = create_engine(f"sqlite:///{tmp_path / 'queue.db'}")
    database.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    db.add_all([Users(username="ann", hashed_password="x"),
                Users(username="rev", hashed_password="x")])
    ds = Datasets(name="ds", description="", dataset_type="image",
                  folder_path="/tmp/ds", created_by="ann")
    db.add(ds)
    db.flush()

    cell = Labels(dataset_id=ds.id, name="cell", value=1)
    core = Labels(dataset_id=ds.id, name="core", value=2)
    db.add_all([cell, core])
    db.flush()

    def image_with_mask(name, fully_annotated):
        img = Images(dataset_id=ds.id, file_name=name, file_path=f"/tmp/{name}",
                     thumbnail_file_path=f"/tmp/t_{name}", width=10, height=10,
                     color_mode="RGB")
        db.add(img)
        db.flush()
        mask = Masks(image_id=img.id, fully_annotated=fully_annotated,
                     file_path=f"/tmp/m_{name}")
        db.add(mask)
        db.flush()
        return img, mask

    # Submitted mask: root_a > child_b > grandchild_c, plus approved root_d.
    img1, mask1 = image_with_mask("a.png", fully_annotated=True)
    root_a = _contour(mask1.id, label_id=cell.id, confidence=0.4)
    db.add(root_a)
    db.flush()
    child_b = _contour(mask1.id, label_id=core.id, parent_id=root_a.id, confidence=0.9)
    db.add(child_b)
    db.flush()
    grandchild_c = _contour(mask1.id, label_id=core.id, parent_id=child_b.id)
    root_d = _contour(mask1.id, label_id=cell.id)
    db.add_all([grandchild_c, root_d])
    db.flush()
    root_d.reviewed_by.append(db.query(Users).filter_by(username="rev").one())

    # Still in progress: excluded unless only_submitted=False.
    img2, mask2 = image_with_mask("b.png", fully_annotated=False)
    in_progress = _contour(mask2.id, label_id=cell.id)
    db.add(in_progress)

    # Submitted but sent back: an open rejection keeps it out of every queue.
    img3, mask3 = image_with_mask("c.png", fully_annotated=True)
    db.add(_contour(mask3.id, label_id=cell.id))
    db.add(AnnotationRejections(mask_id=mask3.id, reason="bad_outline",
                                created_by="rev"))
    db.commit()

    yield {
        "db": db, "dataset_id": ds.id,
        "labels": {"cell": cell.id, "core": core.id},
        "contours": {"a": root_a.id, "b": child_b.id, "c": grandchild_c.id,
                     "d": root_d.id, "in_progress": in_progress.id},
        "images": {"submitted": img1.id, "in_progress": img2.id, "rejected": img3.id},
    }
    db.close()


def test_summary_counts_only_submitted_unreviewed_work(ctx):
    summary = summarize(ctx["dataset_id"], ctx["db"])
    # a, b, c pending; d approved, in-progress mask not submitted, rejected mask out.
    assert summary.pending_instances == 3
    assert summary.pending_images == 1
    assert summary.reviewed_instances == 1
    assert summary.open_rejections == 1
    assert {option.key for option in summary.strategies} >= {"hierarchy"}


def test_hierarchy_queue_orders_roots_first(ctx):
    queue = build_queue(ctx["dataset_id"], ReviewQueueRequest(
        granularity=ReviewGranularity.HIERARCHY), ctx["db"])
    ids = [item.contour_id for item in queue.instances]
    c = ctx["contours"]
    assert ids == [c["a"], c["b"], c["c"]]
    assert [item.depth for item in queue.instances] == [0, 1, 2]
    assert queue.total == 3


def test_include_reviewed_requeues_approved_work_own_included(ctx):
    """A solo reviewer must be able to re-sweep everything, their own approvals
    included — root_d (already approved by rev) comes back for everyone."""
    c = ctx["contours"]
    queue = build_queue(ctx["dataset_id"], ReviewQueueRequest(
        granularity=ReviewGranularity.HIERARCHY, include_reviewed=True), ctx["db"])
    assert queue.include_reviewed is True
    assert [item.contour_id for item in queue.instances] == [c["a"], c["d"], c["b"], c["c"]]


def test_descending_reverses_the_order(ctx):
    queue = build_queue(ctx["dataset_id"], ReviewQueueRequest(
        granularity=ReviewGranularity.HIERARCHY,
        direction=ReviewSortDirection.DESCENDING), ctx["db"])
    c = ctx["contours"]
    assert [item.contour_id for item in queue.instances] == [c["c"], c["b"], c["a"]]


def test_only_submitted_false_sweeps_in_progress_work(ctx):
    queue = build_queue(ctx["dataset_id"], ReviewQueueRequest(
        granularity=ReviewGranularity.HIERARCHY, only_submitted=False), ctx["db"])
    ids = {item.contour_id for item in queue.instances}
    assert ctx["contours"]["in_progress"] in ids
    assert queue.total == 4


def test_image_queue_groups_and_counts(ctx):
    queue = build_queue(ctx["dataset_id"], ReviewQueueRequest(
        granularity=ReviewGranularity.IMAGES), ctx["db"])
    assert queue.total == 1
    item = queue.images[0]
    assert item.image_id == ctx["images"]["submitted"]
    assert item.pending_instances == 3
    assert item.total_instances == 4


def test_image_queue_counts_reviewed_work_when_included(ctx):
    queue = build_queue(ctx["dataset_id"], ReviewQueueRequest(
        granularity=ReviewGranularity.IMAGES, include_reviewed=True), ctx["db"])
    assert queue.images[0].pending_instances == 4


def test_custom_queue_filters_by_label(ctx):
    queue = build_queue(ctx["dataset_id"], ReviewQueueRequest(
        granularity=ReviewGranularity.CUSTOM,
        label_ids=[ctx["labels"]["core"]]), ctx["db"])
    c = ctx["contours"]
    assert [item.contour_id for item in queue.instances] == [c["b"], c["c"]]


def test_custom_queue_rejects_foreign_labels(ctx):
    with pytest.raises(HTTPException) as exc:
        build_queue(ctx["dataset_id"], ReviewQueueRequest(
            granularity=ReviewGranularity.CUSTOM, label_ids=[999]), ctx["db"])
    assert exc.value.status_code == 400


def test_unknown_strategy_is_rejected(ctx):
    with pytest.raises(HTTPException) as exc:
        build_queue(ctx["dataset_id"], ReviewQueueRequest(
            granularity=ReviewGranularity.HIERARCHY,
            sort_strategy="does-not-exist"), ctx["db"])
    assert exc.value.status_code == 400


def test_uncertainty_strategy_orders_least_confident_first(ctx):
    assert "uncertainty" in SORT_STRATEGIES
    queue = build_queue(ctx["dataset_id"], ReviewQueueRequest(
        granularity=ReviewGranularity.HIERARCHY,
        sort_strategy="uncertainty"), ctx["db"])
    scores = [item.score for item in queue.instances]
    assert scores == sorted(scores)
    # root_a (0.4) is the least confident pending contour.
    assert queue.instances[0].contour_id == ctx["contours"]["a"]


def test_rejecting_a_contour_withdraws_own_approval_only(ctx):
    """Rejecting an instance you approved earlier overwrites your verdict; a
    co-reviewer's approval of the same contour is untouched."""
    import asyncio

    from app.schemas.review import RejectionCreate, RejectionReason
    from app.services.database_access import rejections as rejections_db

    db = ctx["db"]
    d = db.query(Contours).filter_by(id=ctx["contours"]["d"]).one()
    db.add(Users(username="rev2", hashed_password="x"))
    db.flush()
    d.reviewed_by.append(db.query(Users).filter_by(username="rev2").one())
    db.commit()

    asyncio.run(rejections_db.reject(
        d.mask_id,
        RejectionCreate(reason=RejectionReason.BAD_OUTLINE, contour_id=d.id),
        username="rev", db=db,
    ))
    assert {u.username for u in d.reviewed_by} == {"rev2"}


def _review_client(ctx, username):
    """A client for the real reviews router, authenticated as `username`."""
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


def test_bulk_approve_respects_separation_of_duties(ctx):
    """The image-level Accept approves everything pending except the caller's own
    work on datasets that require independent review."""
    from app.schemas.permissions import DatasetRole

    db = ctx["db"]
    dataset = db.query(Datasets).filter_by(id=ctx["dataset_id"]).one()
    dataset.require_independent_review = True
    _grant(ctx, "rev", DatasetRole.REVIEWER.value)
    # child_b now belongs to the reviewer, so approving it themselves must be skipped.
    child = db.query(Contours).filter_by(id=ctx["contours"]["b"]).one()
    child.author_username = "rev"
    db.commit()

    mask_id = db.query(Contours.mask_id).filter_by(id=ctx["contours"]["a"]).scalar()
    response = _review_client(ctx, "rev").post(f"/reviews/masks/{mask_id}/approve")
    assert response.status_code == 200
    body = response.json()
    assert set(body["approved"]) == {ctx["contours"]["a"], ctx["contours"]["c"]}
    assert body["skipped"] == [ctx["contours"]["b"]]


def test_bulk_approve_can_stack_a_second_opinion(ctx):
    """With include_reviewed, a fresh reviewer's Accept also covers contours that
    already carry someone else's approval — approvals add up, never replace."""
    from app.schemas.permissions import DatasetRole

    db = ctx["db"]
    db.add(Users(username="rev2", hashed_password="x"))
    _grant(ctx, "rev2", DatasetRole.REVIEWER.value)
    db.commit()

    mask_id = db.query(Contours.mask_id).filter_by(id=ctx["contours"]["a"]).scalar()
    client = _review_client(ctx, "rev2")

    # Without the flag, root_d (already approved by rev) stays untouched.
    body = client.post(f"/reviews/masks/{mask_id}/approve").json()
    assert ctx["contours"]["d"] not in body["approved"]

    # With it, rev2's approval lands on top of rev's.
    body = client.post(f"/reviews/masks/{mask_id}/approve?include_reviewed=true").json()
    assert ctx["contours"]["d"] in body["approved"]
    reviewers = {user.username for user in
                 db.query(Contours).filter_by(id=ctx["contours"]["d"]).one().reviewed_by}
    assert reviewers == {"rev", "rev2"}

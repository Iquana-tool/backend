"""Tests for completeness checks and for label assignment no longer self-approving.

Covers storing a reviewer's per-label answers, the send-back an "incomplete" answer
creates, reading back each reviewer's latest answer, the second look an "unsure"
answer asks for in the image review queue, and `approve_on_label_change`: a reviewer
relabelling someone else's object approves it, the object's own author does not.
Driven against the services with a temp SQLite database, same shape as
`test_review_queue.py`.
"""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.completeness  # noqa: F401
import app.database.contours  # noqa: F401
import app.database.dataset_members  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.labels  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.rejections  # noqa: F401
import app.database.users  # noqa: F401
from app.database.completeness import CompletenessChecks
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.rejections import AnnotationRejections
from app.database.users import Users
from app.schemas.review import (
    CompletenessCreate,
    CompletenessVerdict,
    ReviewGranularity,
    ReviewQueueRequest,
)
from app.services import hierarchy_cache, provenance
from app.services.database_access import completeness as completeness_db
from app.services.database_access import contours as contours_db
from app.services.review_queue import build_queue, summarize


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """One submitted image whose only object is already approved, plus two labels."""
    engine = create_engine(f"sqlite:///{tmp_path / 'completeness.db'}")
    database.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    # Side effects of an approval that need services this test does not run.
    monkeypatch.setattr(hierarchy_cache, "invalidate", lambda *_: None)
    monkeypatch.setattr(provenance, "record_review", lambda *_: None)

    db.add_all([Users(username=name, hashed_password="x") for name in ("ann", "rev", "rev2")])
    ds = Datasets(name="ds", description="", dataset_type="image",
                  folder_path="/tmp/ds", created_by="ann")
    db.add(ds)
    db.flush()
    colony = Labels(dataset_id=ds.id, name="Colony", value=1)
    polyp = Labels(dataset_id=ds.id, name="Polyp", value=2)
    db.add_all([colony, polyp])
    db.flush()
    image = Images(dataset_id=ds.id, file_name="a.png", file_path="/tmp/a.png",
                   thumbnail_file_path="/tmp/t_a.png", width=10, height=10, color_mode="RGB")
    db.add(image)
    db.flush()
    mask = Masks(image_id=image.id, fully_annotated=True, file_path="/tmp/m_a.png")
    db.add(mask)
    db.flush()
    contour = Contours(mask_id=mask.id, label_id=colony.id, added_by="manual",
                       author_username="ann", confidence_score=1.0, area=1.0,
                       perimeter=1.0, circularity=1.0, diameter=1.0,
                       x=[0.1, 0.2], y=[0.1, 0.2])
    contour.reviewed_by = [db.query(Users).filter_by(username="rev").one()]
    db.add(contour)
    db.commit()
    return SimpleNamespace(db=db, ds=ds, mask=mask, contour=contour, colony=colony, polyp=polyp)


def answers(ctx, colony, polyp, note=None):
    return CompletenessCreate(note=note, answers=[
        {"label_id": ctx.colony.id, "answer": colony},
        {"label_id": ctx.polyp.id, "answer": polyp},
    ])


def image_queue(ctx, username):
    request = ReviewQueueRequest(granularity=ReviewGranularity.IMAGES)
    return build_queue(ctx.ds.id, request, ctx.db, username).images


def test_complete_answers_are_stored_without_a_send_back(ctx):
    checks, verdict, rejection = run(completeness_db.record(
        ctx.mask.id, answers(ctx, "complete", "complete"), "rev", ctx.db))

    assert verdict is CompletenessVerdict.COMPLETE
    assert rejection is None
    assert {(check.label_id, check.answer) for check in checks} == {
        (ctx.colony.id, "complete"), (ctx.polyp.id, "complete")}
    assert ctx.db.query(AnnotationRejections).count() == 0


def test_incomplete_sends_the_image_back_naming_the_missing_labels(ctx):
    checks, verdict, rejection = run(completeness_db.record(
        ctx.mask.id, answers(ctx, "complete", "incomplete", note="Left colony"), "rev", ctx.db))

    assert verdict is CompletenessVerdict.INCOMPLETE
    assert rejection.reason == "missing_objects"
    assert rejection.contour_id is None
    assert rejection.note == "Missing: Polyp. Left colony"
    assert ctx.db.get(Masks, ctx.mask.id).fully_annotated is False
    polyp_row = next(check for check in checks if check.label_id == ctx.polyp.id)
    colony_row = next(check for check in checks if check.label_id == ctx.colony.id)
    assert polyp_row.rejection_id == rejection.id
    assert colony_row.rejection_id is None


def test_a_label_from_another_dataset_is_refused(ctx):
    other = Datasets(name="other", description="", dataset_type="image",
                     folder_path="/tmp/o", created_by="ann")
    ctx.db.add(other)
    ctx.db.flush()
    foreign = Labels(dataset_id=other.id, name="Foreign", value=1)
    ctx.db.add(foreign)
    ctx.db.commit()

    body = CompletenessCreate(answers=[{"label_id": foreign.id, "answer": "complete"}])
    with pytest.raises(HTTPException) as error:
        run(completeness_db.record(ctx.mask.id, body, "rev", ctx.db))
    assert error.value.status_code == 400
    assert ctx.db.query(CompletenessChecks).count() == 0


def test_each_item_can_be_answered_only_once():
    with pytest.raises(ValidationError):
        CompletenessCreate(answers=[{"label_id": 1, "answer": "complete"},
                                    {"label_id": 1, "answer": "unsure"}])
    with pytest.raises(ValidationError):
        CompletenessCreate(answers=[{"item": "label", "answer": "complete"}])


def test_the_latest_answer_of_each_reviewer_is_current(ctx):
    run(completeness_db.record(ctx.mask.id, answers(ctx, "unsure", "complete"), "rev", ctx.db))
    run(completeness_db.record(ctx.mask.id, answers(ctx, "complete", "complete"), "rev", ctx.db))
    run(completeness_db.record(ctx.mask.id, answers(ctx, "unsure", "unsure"), "rev2", ctx.db))

    current = completeness_db.current_for_mask(ctx.mask.id, ctx.db)
    by_reviewer = {(check.created_by, check.label_id): check.answer for check in current}
    assert by_reviewer[("rev", ctx.colony.id)] == "complete"
    assert by_reviewer[("rev2", ctx.colony.id)] == "unsure"
    assert len(current) == 4


def test_unsure_queues_a_fully_approved_image_for_someone_else(ctx):
    # Every object is approved, so the image would normally not be queued at all.
    assert image_queue(ctx, "rev2") == []

    run(completeness_db.record(ctx.mask.id, answers(ctx, "complete", "unsure"), "rev", ctx.db))

    queued = image_queue(ctx, "rev2")
    assert [item.mask_id for item in queued] == [ctx.mask.id]
    assert queued[0].completeness_unsure is True
    assert queued[0].pending_instances == 0
    assert summarize(ctx.ds.id, ctx.db, "rev2").unsure_images == 1
    # The reviewer who was unsure is not asked to give the second look.
    assert image_queue(ctx, "rev") == []


def test_a_second_answer_settles_an_unsure_image(ctx):
    run(completeness_db.record(ctx.mask.id, answers(ctx, "complete", "unsure"), "rev", ctx.db))
    run(completeness_db.record(ctx.mask.id, answers(ctx, "complete", "complete"), "rev2", ctx.db))

    assert image_queue(ctx, "ann") == []
    assert summarize(ctx.ds.id, ctx.db, "ann").unsure_images == 0


def test_unsure_queues_an_image_without_objects_for_someone_else(ctx):
    ctx.db.delete(ctx.contour)
    ctx.db.commit()
    assert image_queue(ctx, "rev2") == []

    run(completeness_db.record(ctx.mask.id, answers(ctx, "complete", "unsure"), "rev", ctx.db))

    queued = image_queue(ctx, "rev2")
    assert [(item.mask_id, item.total_instances) for item in queued] == [(ctx.mask.id, 0)]
    assert queued[0].completeness_unsure is True
    assert image_queue(ctx, "rev") == []
    # Independent review defaults the "By" filter to everyone but the reviewer.
    others = ReviewQueueRequest(granularity=ReviewGranularity.IMAGES, by="others")
    assert len(build_queue(ctx.ds.id, others, ctx.db, "rev2").images) == 1
    mine = ReviewQueueRequest(granularity=ReviewGranularity.IMAGES, by="mine")
    assert build_queue(ctx.ds.id, mine, ctx.db, "rev2").images == []


def _user(username, may_review=True):
    return SimpleNamespace(username=username, has_permission=lambda *_: may_review)


def test_the_author_labelling_their_own_object_does_not_approve_it(ctx):
    ctx.contour.reviewed_by = []
    ctx.db.commit()

    approved = run(contours_db.approve_on_label_change(ctx.contour.id, _user("ann"), ctx.ds.id, ctx.db))

    assert approved is False
    assert ctx.db.get(Contours, ctx.contour.id).reviewed_by == []


def test_a_reviewer_relabelling_someone_elses_object_approves_it(ctx):
    ctx.contour.reviewed_by = []
    ctx.db.commit()

    approved = run(contours_db.approve_on_label_change(ctx.contour.id, _user("rev2"), ctx.ds.id, ctx.db))

    assert approved is True
    assert [user.username for user in ctx.db.get(Contours, ctx.contour.id).reviewed_by] == ["rev2"]


def test_a_relabel_without_review_rights_does_not_approve(ctx):
    ctx.contour.reviewed_by = []
    ctx.db.commit()

    approved = run(contours_db.approve_on_label_change(
        ctx.contour.id, _user("rev2", may_review=False), ctx.ds.id, ctx.db))

    assert approved is False

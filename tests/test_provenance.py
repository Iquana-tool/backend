"""Tests for object provenance and the AI suggestion log."""
import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import database
import app.database.ai_suggestions  # noqa: F401
import app.database.contours  # noqa: F401
import app.database.dataset_members  # noqa: F401
import app.database.datasets  # noqa: F401
import app.database.images  # noqa: F401
import app.database.masks  # noqa: F401
import app.database.users  # noqa: F401
from app.database.ai_suggestions import AiSuggestions, SuggestionSource
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.masks import Masks
from app.database.users import Users
from app.services import provenance
from app.services.database_access import contours as contours_db

SQUARE = ([0.1, 0.2, 0.2, 0.1], [0.1, 0.1, 0.2, 0.2])


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'provenance.db'}")
    database.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add_all([Users(username="ann", hashed_password="x"), Users(username="rev", hashed_password="x")])
    ds = Datasets(name="ds", description="", dataset_type="image", folder_path="/tmp/ds", created_by="ann")
    db.add(ds)
    db.flush()
    img = Images(dataset_id=ds.id, file_name="a.png", file_path="/tmp/a.png",
                 thumbnail_file_path="/tmp/t_a.png", width=100, height=100, color_mode="RGB")
    db.add(img)
    db.flush()
    mask = Masks(image_id=img.id, fully_annotated=False, file_path="/tmp/m_a.png")
    db.add(mask)
    db.commit()
    # Metric bookkeeping is out of scope here and needs tables this test does not build.
    for name in ("mark_appearance_stale", "mark_contextual_stale_for_group",
                 "mark_relational_stale_for_parent", "_invalidate_metrics_for_modified_contour"):
        monkeypatch.setattr(contours_db, name, lambda *a, **k: None)
    monkeypatch.setattr(contours_db, "may_review", _allow, raising=False)
    return db, ds, img, mask


async def _allow(*_args, **_kwargs):
    return True


def _contour(db, mask, *, added_by="User", parent=None):
    contour = Contours(mask_id=mask.id, parent_id=parent.id if parent else None, added_by=added_by,
                       author_username="ann", confidence_score=0.9, area=1.0, perimeter=1.0,
                       circularity=1.0, diameter=1.0, x=list(SQUARE[0]), y=list(SQUARE[1]))
    db.add(contour)
    db.commit()
    return contour


def _suggestion(db, contour):
    return db.query(AiSuggestions).filter_by(contour_id=contour.id).order_by(AiSuggestions.id.desc()).first()


def test_recording_a_suggestion_stamps_the_object_and_keeps_its_outline(ctx):
    db, ds, img, mask = ctx
    contour = _contour(db, mask, added_by="sam2")

    provenance.record_suggestions(db, [contour.id], source=SuggestionSource.PROMPTED,
                                  model_key="sam2", username="ann")

    db.refresh(contour)
    suggestion = _suggestion(db, contour)
    assert contour.origin == "prompted"
    assert contour.suggestion_id == suggestion.id
    assert (suggestion.x, suggestion.y) == (list(SQUARE[0]), list(SQUARE[1]))
    assert (suggestion.dataset_id, suggestion.image_id, suggestion.mask_id) == (ds.id, img.id, mask.id)
    assert suggestion.model_key == "sam2" and suggestion.username == "ann"


def test_mark_manual_never_overwrites_an_existing_origin(ctx):
    db, _, _, mask = ctx
    ai = _contour(db, mask, added_by="sam2")
    drawn = _contour(db, mask)
    provenance.record_suggestions(db, [ai.id], source=SuggestionSource.PROMPTED,
                                  model_key="sam2", username="ann")

    provenance.mark_manual(db, [ai.id, drawn.id])

    db.refresh(ai)
    db.refresh(drawn)
    assert (ai.origin, drawn.origin) == ("prompted", "manual")


def test_a_hand_edit_marks_the_object_and_its_suggestion(ctx):
    db, _, _, mask = ctx
    contour = _contour(db, mask, added_by="sam2")
    provenance.record_suggestions(db, [contour.id], source=SuggestionSource.INSTANCE_SUGGESTION,
                                  model_key="sam3", username="ann")

    run(contours_db.modify_contour(contour.id, db, x=[0.1, 0.3, 0.3, 0.1]))

    db.refresh(contour)
    suggestion = _suggestion(db, contour)
    assert contour.geometry_edited_at is not None
    assert suggestion.edited_at is not None
    # The logged outline is still the original suggestion, for measuring the change.
    assert suggestion.x == list(SQUARE[0])


def test_changing_only_the_label_is_not_a_geometry_edit(ctx):
    db, _, _, mask = ctx
    contour = _contour(db, mask, added_by="sam2")
    provenance.record_suggestions(db, [contour.id], source=SuggestionSource.PROMPTED,
                                  model_key="sam2", username="ann")

    run(contours_db.modify_contour(contour.id, db, parent_id=None))

    db.refresh(contour)
    assert contour.geometry_edited_at is None
    assert _suggestion(db, contour).edited_at is None


def test_deleting_an_object_rejects_its_suggestion_and_its_childrens(ctx):
    db, _, _, mask = ctx
    parent = _contour(db, mask, added_by="sam2")
    child = _contour(db, mask, added_by="sam2", parent=parent)
    provenance.record_suggestions(db, [parent.id, child.id], source=SuggestionSource.CROSS_IMAGE,
                                  model_key="dino", username="ann")
    parent_suggestion, child_suggestion = _suggestion(db, parent), _suggestion(db, child)
    parent_id = parent.id

    run(contours_db.delete_contour(parent_id, db))

    db.expire_all()
    assert db.query(Contours).filter_by(id=parent_id).first() is None
    assert db.get(AiSuggestions, parent_suggestion.id).deleted_at is not None
    assert db.get(AiSuggestions, child_suggestion.id).deleted_at is not None


def test_undo_of_a_delete_clears_the_rejection(ctx):
    db, _, _, mask = ctx
    contour = _contour(db, mask, added_by="sam2")
    provenance.record_suggestions(db, [contour.id], source=SuggestionSource.PROMPTED,
                                  model_key="sam2", username="ann")
    provenance.record_deletion(db, [contour.id])

    provenance.record_restore(db, [contour.id])

    assert _suggestion(db, contour).deleted_at is None


def test_approving_an_object_records_the_first_review_of_its_suggestion(ctx):
    db, _, _, mask = ctx
    contour = _contour(db, mask, added_by="sam2")
    provenance.record_suggestions(db, [contour.id], source=SuggestionSource.PROMPTED,
                                  model_key="sam2", username="ann")

    run(contours_db.review_contour(contour.id, SimpleNamespace(username="rev"), db, strict=False))

    assert _suggestion(db, contour).reviewed_at is not None


def test_refinement_logs_a_new_suggestion_and_supersedes_the_old_one(ctx):
    db, _, _, mask = ctx
    contour = _contour(db, mask, added_by="sam2")
    provenance.record_suggestions(db, [contour.id], source=SuggestionSource.PROMPTED,
                                  model_key="sam2", username="ann")
    first = _suggestion(db, contour)

    provenance.record_refinement(db, contour.id, model_key="sam2", username="ann")

    db.refresh(contour)
    db.refresh(first)
    latest = _suggestion(db, contour)
    assert latest.id != first.id and latest.source == "refine"
    assert contour.suggestion_id == latest.id
    assert contour.origin == "prompted"  # created by prompting; refinement does not change that
    assert first.superseded_at is not None and first.edited_at is None
    # A later delete rejects the live suggestion only.
    provenance.record_deletion(db, [contour.id])
    db.refresh(first)
    assert first.deleted_at is None and _suggestion(db, contour).deleted_at is not None


def test_replacing_an_objects_geometry_keeps_its_identity(ctx, monkeypatch):
    db, _, _, mask = ctx
    contour = _contour(db, mask, added_by="sam2")
    provenance.record_suggestions(db, [contour.id], source=SuggestionSource.PROMPTED,
                                  model_key="sam2", username="ann")
    db.refresh(contour)
    before = (contour.origin, contour.suggestion_id, contour.created_at)

    replacement = SimpleNamespace(id=None)

    def fake_save(session, schema, mask_id, parent_id, author_username=None):
        session.add(Contours(id=schema.id, mask_id=mask_id, parent_id=parent_id, added_by="sam2",
                             author_username=author_username, confidence_score=1.0, area=1.0,
                             perimeter=1.0, circularity=1.0, diameter=1.0,
                             x=[0.5, 0.6, 0.6], y=[0.5, 0.5, 0.6]))
        session.flush()

    monkeypatch.setattr(contours_db, "save_contour_tree", fake_save)
    assert run(contours_db.replace_contour(contour.id, replacement, db))

    db.expire_all()
    replaced = db.get(Contours, contour.id)
    assert (replaced.origin, replaced.suggestion_id, replaced.created_at) == before


def test_a_failing_record_does_not_raise(ctx):
    db, _, _, _ = ctx
    broken = SimpleNamespace(query=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    assert provenance.record_geometry_edit(broken, 1) is None


def test_an_unnamed_ai_object_gets_the_model_as_added_by(ctx):
    db, _, _, mask = ctx
    contour = _contour(db, mask, added_by="")

    provenance.record_suggestions(db, [contour.id], source=SuggestionSource.INSTANCE_SUGGESTION,
                                  model_key="sam3", username="ann")

    db.refresh(contour)
    assert contour.added_by == "sam3"


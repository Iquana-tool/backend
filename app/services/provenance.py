"""Where each object came from, and what happened to the AI suggestions behind it.

Two records, kept in step by the write paths:

* ``contours.origin`` / ``suggestion_id`` / ``geometry_edited_at`` -- per object: which
  tool created it, which AI suggestion its outline currently is, and whether a person
  has since changed that outline.
* ``ai_suggestions`` -- per suggested object: the outline as the model returned it and
  timestamps for what the person did to it (edited, deleted, approved).

Together they answer the questions a user study asks of annotation data: which tool
each final object came from, how many suggestions were kept, fixed or thrown away,
and how far final outlines moved from what the AI proposed.

Recording never breaks the operation it describes. Every entry point swallows and
logs its own failure: losing one provenance row is better than failing the edit the
person just made.
"""
from __future__ import annotations

from datetime import datetime, timezone
from logging import getLogger
from typing import Iterable
from uuid import uuid4

from sqlalchemy.orm import Session

from app.database.ai_suggestions import AiSuggestions, SuggestionSource
from app.database.contours import Contours
from app.database.images import Images
from app.database.masks import Masks

logger = getLogger(__name__)

MANUAL = "manual"
#: `added_by` values that mean a person drew the outline.
MANUAL_ADDED_BY = {"User", "user", "manual", ""}
IMPORT = "import"
AI_SOURCES = {
    SuggestionSource.PROMPTED, SuggestionSource.REFINE, SuggestionSource.INSTANCE_SUGGESTION,
    SuggestionSource.INSTANCE_SEGMENTATION, SuggestionSource.CROSS_IMAGE,
    SuggestionSource.BATCH_INFERENCE,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_run_id() -> str:
    return uuid4().hex


def _safely(action: str):
    def wrap(fn):
        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:  # noqa: BLE001 -- provenance must never break the edit
                logger.warning("Could not record provenance (%s).", action, exc_info=True)
                # A failed flush leaves the session unusable for the caller's own work;
                # a healthy one is left alone so uncommitted caller writes survive.
                db = args[0] if args else kwargs.get("db")
                if isinstance(db, Session) and not db.is_active:
                    db.rollback()
                return None
        inner.__name__ = fn.__name__
        inner.__doc__ = fn.__doc__
        return inner
    return wrap


def _scope(db: Session, mask_id: int) -> tuple[int | None, int | None]:
    row = (db.query(Images.dataset_id, Masks.image_id)
           .join(Images, Images.id == Masks.image_id)
           .filter(Masks.id == mask_id).first())
    return (row[0], row[1]) if row else (None, None)


@_safely("manual origin")
def mark_manual(db: Session, contour_ids: Iterable[int], origin: str = MANUAL) -> None:
    """Stamp objects drawn by hand (or imported) with their origin. Commits."""
    ids = [cid for cid in contour_ids if cid is not None]
    if not ids:
        return
    (db.query(Contours).filter(Contours.id.in_(ids), Contours.origin.is_(None))
     .update({Contours.origin: origin}, synchronize_session=False))
    db.commit()


@_safely("suggestion log")
def record_suggestions(db: Session, contour_ids: Iterable[int], *, source: str,
                       model_key: str | None, username: str | None,
                       run_id: str | None = None, commit: bool = True) -> list[int]:
    """Log AI output that has just been saved as objects, and stamp the objects.

    Reads the saved rows back, so the logged outline is exactly what was stored (after
    hierarchy fitting), which is the geometry the person was shown.
    """
    ids = [cid for cid in contour_ids if cid is not None]
    if not ids:
        return []
    run_id = run_id or new_run_id()
    rows = db.query(Contours).filter(Contours.id.in_(ids)).all()
    scope_cache: dict[int, tuple[int | None, int | None]] = {}
    created = []
    for contour in rows:
        if contour.mask_id not in scope_cache:
            scope_cache[contour.mask_id] = _scope(db, contour.mask_id)
        dataset_id, image_id = scope_cache[contour.mask_id]
        suggestion = AiSuggestions(
            run_id=run_id, source=source, model_key=model_key, username=username,
            dataset_id=dataset_id, image_id=image_id, mask_id=contour.mask_id,
            contour_id=contour.id, label_id=contour.label_id,
            confidence=contour.confidence_score,
            x=list(contour.x or []), y=list(contour.y or []),
        )
        db.add(suggestion)
        db.flush()
        if contour.origin is None:
            contour.origin = source
        # Some services return objects without naming the model; the log knows it.
        if model_key and not (contour.added_by or "").strip():
            contour.added_by = model_key
        contour.suggestion_id = suggestion.id
        created.append(suggestion.id)
    if commit:
        db.commit()
    return created


@_safely("refinement")
def record_refinement(db: Session, contour_id: int, *, model_key: str | None,
                      username: str | None) -> None:
    """An AI refinement replaced an object's outline: the new outline is a new suggestion.

    The object keeps its origin (it was still created by whatever made it first); its
    suggestion link moves to the refined outline, and the replaced suggestion is marked
    superseded rather than edited, since a person did not change it by hand.
    """
    contour = db.query(Contours).filter_by(id=contour_id).first()
    if contour is None:
        return
    previous = contour.suggestion_id
    record_suggestions(db, [contour_id], source=SuggestionSource.REFINE,
                       model_key=model_key, username=username, commit=False)
    if previous is not None:
        (db.query(AiSuggestions).filter(AiSuggestions.id == previous,
                                        AiSuggestions.superseded_at.is_(None))
         .update({AiSuggestions.superseded_at: _now()}, synchronize_session=False))
    db.commit()


@_safely("geometry edit")
def record_geometry_edit(db: Session, contour_id: int) -> None:
    """A person changed an object's outline by hand."""
    contour = db.query(Contours).filter_by(id=contour_id).first()
    if contour is None:
        return
    now = _now()
    contour.geometry_edited_at = now
    if contour.suggestion_id is not None:
        (db.query(AiSuggestions).filter(AiSuggestions.id == contour.suggestion_id,
                                        AiSuggestions.edited_at.is_(None))
         .update({AiSuggestions.edited_at: now}, synchronize_session=False))
    db.commit()


@_safely("deletion")
def record_deletion(db: Session, contour_ids: Iterable[int], commit: bool = True) -> None:
    """Objects were deleted: their live suggestions count as rejected."""
    ids = [cid for cid in contour_ids if cid is not None]
    if not ids:
        return
    (db.query(AiSuggestions).filter(AiSuggestions.contour_id.in_(ids),
                                    AiSuggestions.deleted_at.is_(None),
                                    AiSuggestions.superseded_at.is_(None))
     .update({AiSuggestions.deleted_at: _now()}, synchronize_session=False))
    if commit:
        db.commit()


@_safely("restore")
def record_restore(db: Session, contour_ids: Iterable[int], commit: bool = True) -> None:
    """An undo brought deleted objects back: their suggestions are no longer rejected."""
    ids = [cid for cid in contour_ids if cid is not None]
    if not ids:
        return
    (db.query(AiSuggestions).filter(AiSuggestions.contour_id.in_(ids),
                                    AiSuggestions.deleted_at.is_not(None))
     .update({AiSuggestions.deleted_at: None}, synchronize_session=False))
    if commit:
        db.commit()


@_safely("review")
def record_review(db: Session, contour_id: int) -> None:
    """An object was approved: the first approval of its suggestion is kept."""
    contour = db.query(Contours).filter_by(id=contour_id).first()
    if contour is None or contour.suggestion_id is None:
        return
    (db.query(AiSuggestions).filter(AiSuggestions.id == contour.suggestion_id,
                                    AiSuggestions.reviewed_at.is_(None))
     .update({AiSuggestions.reviewed_at: _now()}, synchronize_session=False))
    db.commit()


def contour_ids_of(contour_schema) -> list[int]:
    """Ids of a saved contour tree (root first), for callers that hold the schema."""
    ids, stack = [], [contour_schema]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        if getattr(node, "id", None) is not None:
            ids.append(node.id)
        stack.extend(getattr(node, "children", None) or [])
    return ids

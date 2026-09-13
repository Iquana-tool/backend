"""Recording and replaying the annotator's action history (undo / redo).

The module has two halves:

  * ``record_*`` -- called by the write paths (the WebSocket handlers and the REST
    contour routes) right after they change something. Each call appends one row to
    ``annotation_actions`` and discards whatever redo branch was outstanding.
  * ``undo`` / ``redo`` -- pop the newest row off the appropriate stack and apply
    it in the corresponding direction.

Every row carries enough state to go BOTH ways, so undo and redo read the same
row and only differ in which direction they apply it. That is what makes
``create`` and ``delete`` one implementation: undoing a create *is* deleting, and
redoing it *is* restoring, which is exactly what undo/redo of a delete does with
the arrows reversed.

Restoration is faithful where it can be and honest where it cannot. A restored
contour keeps its id, its children, its label, its author and its approvals. But
the world may have moved on since the action was recorded -- the parent contour
may itself have been deleted, a label may have been removed from the dataset, a
reviewer's account may be gone. Rather than fail with a foreign-key error or
silently write a dangling reference, each of those cases degrades to something
defensible and says so in the returned message.
"""
import uuid
from datetime import datetime
from logging import getLogger

from sqlalchemy.orm import Session

from app.database.annotation_actions import (
    MAX_HISTORY_ENTRIES,
    ActionType,
    AnnotationActions,
)
from app.database.contours import Contours, dual_write_geometry_metrics
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.services import hierarchy_cache
from app.services.database_access.contours import (
    invalidate_metrics_for_new_contours,
    mark_contextual_stale_for_group,
    mark_relational_stale_for_parent,
)

logger = getLogger(__name__)


class HistoryError(Exception):
    """A history step could not be applied. The message is shown to the user."""


# --- Snapshots ----------------------------------------------------------------


def _serialize_contour(contour: Contours) -> dict:
    """One contour row as JSON, complete enough to re-create it exactly."""
    return {
        "id": contour.id,
        "parent_id": contour.parent_id,
        "label_id": contour.label_id,
        "x": list(contour.x or []),
        "y": list(contour.y or []),
        "added_by": contour.added_by,
        "author_username": contour.author_username,
        "confidence_score": contour.confidence_score,
        "temporary": bool(contour.temporary),
        "created_at": contour.created_at.isoformat() if contour.created_at else None,
        "reviewed_by": [reviewer.username for reviewer in contour.reviewed_by],
    }


def snapshot_subtree(contour_id: int, db: Session) -> dict | None:
    """Capture a contour and everything nested under it.

    Called before a deletion, because afterwards the rows (and, via the CASCADE,
    every descendant) are gone. The contours are laid out parents-first so a
    restore can insert them in list order without ever pointing a ``parent_id`` at
    a row that does not exist yet.

    Args:
        contour_id: The contour about to be deleted or just created.
        db: The database session.

    Returns:
        ``{"root_id": int, "contours": [...]}``, or ``None`` if the contour is gone.
    """
    root = db.query(Contours).filter_by(id=contour_id).first()
    if root is None:
        return None

    serialized = []
    queue = [root]
    while queue:
        contour = queue.pop(0)
        serialized.append(_serialize_contour(contour))
        queue.extend(
            db.query(Contours).filter_by(parent_id=contour.id).order_by(Contours.id).all()
        )

    return {"root_id": root.id, "contours": serialized}


def _restore_subtree(mask_id: int, snapshot: dict, db: Session) -> tuple[list[Contours], list[str]]:
    """Re-insert a snapshotted subtree under its original ids.

    Returns the created rows plus any human-readable notes about state that could
    not be restored faithfully (a vanished parent, label or reviewer).
    """
    notes: list[str] = []
    restored: list[Contours] = []
    # Ids that came back in this call, so a child whose parent is restored in the
    # same pass is not mistaken for one whose parent is gone.
    revived_ids: set[int] = set()

    for entry in snapshot.get("contours", []):
        contour_id = entry["id"]
        if db.query(Contours.id).filter_by(id=contour_id).first() is not None:
            # Something already occupies this id -- the action was undone twice, or
            # the id was reused. Skip rather than trample the current occupant.
            continue

        parent_id = entry.get("parent_id")
        if parent_id is not None and parent_id not in revived_ids:
            if db.query(Contours.id).filter_by(id=parent_id).first() is None:
                # The object this one was nested inside has since been deleted.
                # Bringing it back at the top level keeps the geometry rather than
                # failing the whole undo on a foreign key.
                notes.append("its parent object no longer exists, so it was "
                             "restored at the top level")
                parent_id = None

        label_id = entry.get("label_id")
        if label_id is not None and db.query(Labels.id).filter_by(id=label_id).first() is None:
            notes.append("its label has since been removed from the dataset")
            label_id = None

        created_at = entry.get("created_at")
        contour = Contours(
            id=contour_id,
            mask_id=mask_id,
            parent_id=parent_id,
            label_id=label_id,
            temporary=entry.get("temporary", False),
            added_by=entry.get("added_by") or "User",
            author_username=entry.get("author_username"),
            confidence_score=entry.get("confidence_score", 1.0),
            created_at=datetime.fromisoformat(created_at) if created_at else None,
            x=entry.get("x") or [],
            y=entry.get("y") or [],
            # NOT NULL columns; dual_write_geometry_metrics recomputes them below
            # from the coordinates, which is also what the original write did.
            area=0.0, perimeter=0.0, circularity=0.0, diameter=0.0,
        )
        db.add(contour)
        db.flush()
        revived_ids.add(contour_id)

        reviewers = entry.get("reviewed_by") or []
        if reviewers:
            found = db.query(Users).filter(Users.username.in_(reviewers)).all()
            contour.reviewed_by = found
            if len(found) != len(reviewers):
                notes.append("some reviewers of it no longer have accounts")

        restored.append(contour)

    if restored:
        dual_write_geometry_metrics(db, mask_id, restored)
        invalidate_metrics_for_new_contours(db, restored)
        # The contours are exemplars again now that they exist; no-op unless the
        # embedding lifecycle is enabled. Local import: embedding_lifecycle reaches
        # back into the contour modules this one already imports.
        from app.services.embedding_lifecycle import enqueue_embed_contours
        enqueue_embed_contours([c.id for c in restored if not c.temporary])

    # Dedupe while preserving order -- a 20-contour subtree should not report the
    # same missing label twenty times.
    return restored, list(dict.fromkeys(notes))


def _delete_subtree_root(contour_id: int, db: Session) -> bool:
    """Delete one contour and, by CASCADE, its descendants. False if already gone."""
    contour = db.query(Contours).filter_by(id=contour_id).first()
    if contour is None:
        return False
    mask_id, parent_id = contour.mask_id, contour.parent_id
    # Same invalidation the ordinary delete path does: the surviving siblings lost
    # a neighbour and the parent lost a child.
    mark_contextual_stale_for_group(db, mask_id, parent_id)
    mark_relational_stale_for_parent(db, [parent_id])
    db.delete(contour)
    db.flush()
    return True


# --- Recording ----------------------------------------------------------------


def new_group_id() -> str:
    """An id tying several recorded actions into one undo step."""
    return uuid.uuid4().hex


def clear_for_mask(mask_id: int, db: Session) -> None:
    """Discard every annotator's history for a mask. Does not commit.

    Called when something replaces a mask's contours wholesale rather than object
    by object -- "Remove all annotations", instance segmentation, batch inference.
    The recorded steps describe contours that no longer exist, and worse, are not
    merely inert: undoing a delete recorded before the wipe would re-insert that
    contour into a mask the user had just emptied, resurrecting one object out of
    a set the user deliberately cleared.

    Undo cannot span an operation it has no inverse for, so the honest thing is to
    start the history over at that point rather than let it reach across.
    """
    (db.query(AnnotationActions)
     .filter(AnnotationActions.mask_id == mask_id)
     .delete(synchronize_session=False))


def _prune(db: Session, mask_id: int, username: str) -> None:
    """Drop the oldest steps beyond ``MAX_HISTORY_ENTRIES``.

    Counts STEPS, not rows: a grouped action (one suggestion run adding thirty
    objects) is one step, and is kept or dropped whole. Pruning half a group would
    leave an undo that restores two thirds of what the user sees.
    """
    rows = (db.query(AnnotationActions)
            .filter_by(mask_id=mask_id, username=username)
            .order_by(AnnotationActions.id.desc())
            .all())

    kept_steps: list[str] = []
    doomed: list[int] = []
    for row in rows:
        step = row.group_id or f"solo:{row.id}"
        if step in kept_steps:
            continue
        if len(kept_steps) < MAX_HISTORY_ENTRIES:
            kept_steps.append(step)
        else:
            doomed.append(row.id)

    if doomed:
        (db.query(AnnotationActions)
         .filter(AnnotationActions.id.in_(doomed))
         .delete(synchronize_session=False))


def _record(db: Session, mask_id: int, username: str, action_type: str,
            payload: dict, group_id: str | None = None) -> None:
    """Append one row and invalidate the redo branch."""
    if mask_id is None or not username:
        return
    # A fresh action makes the undone steps unreachable: there is no coherent
    # state to redo them onto any more.
    (db.query(AnnotationActions)
     .filter_by(mask_id=mask_id, username=username, undone=True)
     .delete(synchronize_session=False))

    db.add(AnnotationActions(
        mask_id=mask_id,
        username=username,
        group_id=group_id,
        action_type=action_type,
        payload=payload,
        undone=False,
    ))
    _prune(db, mask_id, username)
    db.commit()


def record_create(db: Session, mask_id: int, username: str, contour_id: int,
                  group_id: str | None = None) -> None:
    """Record that ``contour_id`` was just added to the mask."""
    snapshot = snapshot_subtree(contour_id, db)
    if snapshot is None:
        return
    _record(db, mask_id, username, ActionType.CREATE,
            {"contour_id": contour_id, "snapshot": snapshot}, group_id)


def record_delete(db: Session, mask_id: int, username: str, snapshot: dict,
                  group_id: str | None = None) -> None:
    """Record a deletion. ``snapshot`` must come from :func:`snapshot_subtree`
    taken BEFORE the rows were removed."""
    if not snapshot:
        return
    _record(db, mask_id, username, ActionType.DELETE,
            {"contour_id": snapshot["root_id"], "snapshot": snapshot}, group_id)


def record_label_change(db: Session, mask_id: int, username: str, contour_id: int,
                        before_label_id: int | None, after_label_id: int | None,
                        group_id: str | None = None) -> None:
    """Record a relabelling. A no-op change records nothing."""
    if before_label_id == after_label_id:
        return
    _record(db, mask_id, username, ActionType.LABEL, {
        "contour_id": contour_id,
        "before_label_id": before_label_id,
        "after_label_id": after_label_id,
    }, group_id)


# --- Applying -----------------------------------------------------------------


def _apply_create(payload: dict, mask_id: int, db: Session, forward: bool) -> str:
    """Apply a ``create`` step. Forward re-adds the contour, backward removes it."""
    contour_id = payload["contour_id"]
    if forward:
        restored, notes = _restore_subtree(mask_id, payload["snapshot"], db)
        if not restored:
            raise HistoryError("That object is already on the image.")
        return _with_notes("Object restored", notes)

    snapshot = payload.get("snapshot") or {}
    known_ids = {entry["id"] for entry in snapshot.get("contours", [])}
    later_children = db.query(Contours.id).filter(Contours.parent_id == contour_id)
    if known_ids:
        later_children = later_children.filter(Contours.id.notin_(known_ids))
    if later_children.first() is not None:
        # Undoing the creation would cascade-delete work that was nested inside it
        # afterwards, which the user never asked to undo.
        raise HistoryError("This object now contains other objects. Delete or move "
                           "them before undoing it.")
    if not _delete_subtree_root(contour_id, db):
        raise HistoryError("That object is no longer on the image.")
    return "Object removed"


def _apply_label(payload: dict, db: Session, forward: bool) -> str:
    """Apply a ``label`` step in the given direction."""
    contour_id = payload["contour_id"]
    target = payload["after_label_id"] if forward else payload["before_label_id"]

    contour = db.query(Contours).filter_by(id=contour_id).first()
    if contour is None:
        raise HistoryError("That object no longer exists.")
    if target is not None and db.query(Labels.id).filter_by(id=target).first() is None:
        raise HistoryError("That label has since been removed from the dataset.")

    contour.label_id = target
    # A label change moves the contour between label-filtered neighbour sets.
    mark_contextual_stale_for_group(db, contour.mask_id, contour.parent_id)
    db.flush()
    return "Label restored" if not forward else "Label reapplied"


def _with_notes(message: str, notes: list[str]) -> str:
    return f"{message} ({'; '.join(notes)})" if notes else message


def _apply(row: AnnotationActions, db: Session, forward: bool) -> str:
    """Apply one recorded row forwards (redo) or backwards (undo)."""
    if row.action_type == ActionType.CREATE:
        return _apply_create(row.payload, row.mask_id, db, forward)
    if row.action_type == ActionType.DELETE:
        # A delete is a create with the arrows swapped: undoing it restores.
        return _apply_create(row.payload, row.mask_id, db, not forward)
    if row.action_type == ActionType.LABEL:
        return _apply_label(row.payload, db, forward)
    raise HistoryError(f"Unknown action type {row.action_type!r}.")


def _newest_step(db: Session, mask_id: int, username: str,
                 undone: bool) -> list[AnnotationActions]:
    """The rows making up the top step of one stack, oldest row first.

    A grouped action returns all of its rows; an ordinary one returns a single-item
    list. Callers apply the list in whichever order their direction needs.
    """
    newest = (db.query(AnnotationActions)
              .filter_by(mask_id=mask_id, username=username, undone=undone)
              .order_by(AnnotationActions.id.desc())
              .first())
    if newest is None:
        return []
    if newest.group_id is None:
        return [newest]
    return (db.query(AnnotationActions)
            .filter_by(mask_id=mask_id, username=username, undone=undone,
                       group_id=newest.group_id)
            .order_by(AnnotationActions.id.asc())
            .all())


_DESCRIPTIONS = {
    ActionType.CREATE: "add object",
    ActionType.DELETE: "delete object",
    ActionType.LABEL: "label change",
}


def _describe(rows: list[AnnotationActions]) -> str | None:
    """A short label for the step, for the toolbar tooltip."""
    if not rows:
        return None
    base = _DESCRIPTIONS.get(rows[0].action_type, "change")
    return f"{base} (×{len(rows)})" if len(rows) > 1 else base


def mask_id_for_image(image_id: int, db: Session) -> int | None:
    return db.query(Masks.id).filter_by(image_id=image_id).scalar()


def get_status(mask_id: int, username: str, db: Session) -> dict:
    """What the undo and redo buttons should show right now."""
    undo_rows = _newest_step(db, mask_id, username, undone=False)
    redo_rows = _newest_step(db, mask_id, username, undone=True)
    return {
        "can_undo": bool(undo_rows),
        "can_redo": bool(redo_rows),
        "undo_label": _describe(undo_rows),
        "redo_label": _describe(redo_rows),
    }


def _run(mask_id: int, username: str, db: Session, forward: bool) -> dict:
    """Shared body of undo and redo.

    Undo takes the newest not-yet-undone step and applies it backwards, newest row
    first so a grouped action unwinds in the reverse of the order it was made.
    Redo takes the newest undone step and applies it forwards in original order.
    """
    rows = _newest_step(db, mask_id, username, undone=forward)
    if not rows:
        raise HistoryError("There is nothing to redo." if forward
                           else "There is nothing to undo.")

    ordered = rows if forward else list(reversed(rows))
    messages = []
    try:
        for row in ordered:
            messages.append(_apply(row, db, forward))
            row.undone = not forward
        db.commit()
    except HistoryError:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        logger.exception("Failed to %s an annotation action on mask %s.",
                         "redo" if forward else "undo", mask_id)
        raise HistoryError("That step could not be applied.")

    hierarchy_cache.invalidate(mask_id)
    message = messages[0] if len(set(messages)) == 1 else "; ".join(messages)
    if len(rows) > 1:
        message = f"{message} ×{len(rows)}"
    return {"action_type": rows[0].action_type, "message": message}


def undo(mask_id: int, username: str, db: Session) -> dict:
    """Revert this annotator's most recent step on this image."""
    return _run(mask_id, username, db, forward=False)


def redo(mask_id: int, username: str, db: Session) -> dict:
    """Re-apply the step this annotator most recently undid on this image."""
    return _run(mask_id, username, db, forward=True)

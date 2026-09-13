"""The per-annotator action history that backs undo/redo in the workspace.

One row is one undoable step. The stack is implicit: rows for a given
``(mask_id, username)`` ordered by ``id``, split by the ``undone`` flag --
``undone=False`` rows are the undo stack (newest first), ``undone=True`` rows are
the redo stack. Recording a new action discards the ``undone=True`` rows, which is
the usual "a new edit kills the redo branch" rule.

Why a table rather than client state: undoing a delete has to bring back the
contour the user actually deleted -- same id, same children, same label, same
approvals -- and only the server can do that. A client-side stack could at best
re-create a lookalike under a fresh id, orphaning its children and any review
already recorded against it.

The history is deliberately narrow. It covers what an annotator does object by
object (add, delete, relabel); it does not cover geometry replacement
(``replace_contour``) or instance segmentation's wipe-and-replace, whose inverses
are not a single contour and would be misleading behind a one-step Ctrl+Z.
"""
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, JSON, String

from app.database import database

#: How many steps of history are kept per annotator per image. Older rows are
#: pruned as new ones arrive. Undo here is a mistake-recovery tool ("I deleted the
#: wrong object"), not a version-control system, and an unbounded log would keep a
#: full geometry snapshot of every contour ever deleted from the image.
MAX_HISTORY_ENTRIES = 10


class ActionType:
    """The kinds of step the history can invert.

    ``CREATE`` and ``DELETE`` are exact inverses of each other and share one
    payload shape, so a single snapshot serves both directions.
    """

    CREATE = "create"
    DELETE = "delete"
    LABEL = "label"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnnotationActions(database):
    """One undoable step taken by one annotator on one image."""

    __tablename__ = "annotation_actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mask_id = Column(Integer, ForeignKey("masks.id", ondelete="CASCADE"),
                     nullable=False, index=True)
    # The stack is per annotator: Ctrl+Z must never revert a colleague's work on
    # a shared image. FK to users so a deleted account takes its history with it.
    username = Column(String, ForeignKey("users.username", ondelete="CASCADE"),
                      nullable=False, index=True)
    # Rows sharing a group are one step in both directions. Set for fan-out
    # operations -- a suggestion run that adds thirty instances should cost one
    # Ctrl+Z, not thirty. NULL for ordinary single-object actions.
    group_id = Column(String(64), nullable=True, index=True)
    action_type = Column(String(16), nullable=False)
    # Everything needed to invert AND re-apply the step, so undo and redo read the
    # same row. See app.services.database_access.annotation_history for the shapes.
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    #: False => on the undo stack. True => undone, and on the redo stack.
    undone = Column(Boolean, nullable=False, default=False)


# Every read is "the newest (un)done row for this annotator on this image", so
# index the pair that scopes it; ordering falls out of the primary key.
Index(
    "ix_annotation_actions_stack",
    AnnotationActions.mask_id,
    AnnotationActions.username,
    AnnotationActions.undone,
)

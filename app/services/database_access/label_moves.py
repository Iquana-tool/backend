"""Moving a label to a different parent in a dataset's hierarchy.

Nesting a label means *part of*: a child label names a part of the thing its parent
names. Annotation enforces that reading — ``_check_contour_label`` only accepts a
contour's label if it is a DIRECT child of the label on the contour containing it — so
re-parenting a label is not a cosmetic edit. It can retroactively invalidate contours
that were legal when they were drawn.

The rule this module implements: a move that would strand annotated objects is refused
and reports what it would break, and the caller may repeat it with ``detach_affected``
to accept the consequence explicitly. Detaching keeps the label the annotator asserted
(the part-of link is the derived half, the class is the observation) and demotes the
contour to root level, where any label is legal.

Only contours carrying the moved label itself are ever affected. Contours labelled with
its children depend on *their* parent still being the moved label, which a move does not
change, and contours nested inside the moved label's objects are equally untouched.
"""

from dataclasses import dataclass
from logging import getLogger

from sqlalchemy import func, or_
from sqlalchemy.orm import Session, aliased

from app.database.contours import Contours
from app.database.labels import Labels
from app.database.masks import Masks
from app.services import hierarchy_cache
from app.services.database_access.contours import (
    mark_contextual_stale_for_group,
    mark_relational_stale_for_parent,
)

logger = getLogger(__name__)


class LabelMoveError(ValueError):
    """The requested move is not a legal hierarchy edit (missing, cyclic, cross-dataset)."""


@dataclass(frozen=True)
class AffectedContour:
    """One annotated object that the move would leave carrying an illegal label."""

    contour_id: int
    image_id: int
    mask_id: int
    old_parent_id: int


@dataclass(frozen=True)
class MoveImpact:
    """What a move would cost, before it is performed."""

    affected: tuple[AffectedContour, ...] = ()

    @property
    def count(self) -> int:
        return len(self.affected)


class LabelMoveBlocked(Exception):
    """The move is legal but would invalidate existing annotations.

    Carries the impact so the caller can report it and offer the detach path.
    """

    def __init__(self, impact: MoveImpact):
        self.impact = impact
        super().__init__(
            f"{impact.count} annotated object(s) would be invalidated by this move."
        )


def _descendant_ids(db: Session, label_id: int) -> set[int]:
    """Every label id beneath ``label_id``, excluding the label itself.

    Walked level by level rather than with a recursive CTE: label spaces are small
    (tens of terms, a handful of levels) and this stays portable across the SQLite
    used in tests and the Postgres used in deployment.
    """
    found: set[int] = set()
    frontier = [label_id]
    while frontier:
        rows = db.query(Labels.id).filter(Labels.parent_id.in_(frontier)).all()
        next_level = {row.id for row in rows} - found
        if not next_level:
            break
        found |= next_level
        frontier = list(next_level)
    return found


def nesting_summary(db: Session, label_id: int) -> dict:
    """How the objects carrying this label are nested today.

    Enough for a client to price EVERY candidate move of this label without asking
    again: the objects stranded by moving it under ``P`` are the nested ones whose
    container is not already labelled ``P``, i.e.
    ``nested_total - by_container_label[P]`` (and all of them when moving to the top
    level). This is what lets a drag show a live count as it passes over rows.

    Objects inside an unlabelled container are counted in ``nested_total`` but appear
    under no key, so they price as stranded by every destination -- which they are.

    The client's arithmetic is an estimate, not a verdict: it can be stale by the time
    the drop happens, so ``move_label`` re-derives the impact and remains the authority.

    Args:
        db: The database session.
        label_id: The label to summarise.

    Returns:
        ``{"nested_total": int, "by_container_label": {label_id: count}}``.
    """
    container = aliased(Contours)
    rows = (
        db.query(container.label_id.label("container_label_id"), func.count(Contours.id).label("n"))
        .join(container, Contours.parent_id == container.id)
        .filter(Contours.label_id == label_id)
        .group_by(container.label_id)
        .all()
    )

    return {
        "nested_total": sum(row.n for row in rows),
        "by_container_label": {
            row.container_label_id: row.n for row in rows if row.container_label_id is not None
        },
    }


def _resolve_target(db: Session, label: Labels, new_parent_id: int | None) -> Labels | None:
    """Validate the requested destination, returning the new parent (None = top level).

    Raises:
        LabelMoveError: If the destination does not exist, belongs to another dataset,
            is the label itself, or sits beneath the label (which would form a cycle).
    """
    if new_parent_id is None:
        return None

    if new_parent_id == label.id:
        raise LabelMoveError(f"'{label.name}' cannot be a part of itself.")

    new_parent = db.query(Labels).filter_by(id=new_parent_id).first()
    if new_parent is None:
        raise LabelMoveError(f"Label with id {new_parent_id} does not exist.")

    if new_parent.dataset_id != label.dataset_id:
        raise LabelMoveError(
            f"'{new_parent.name}' belongs to a different dataset than '{label.name}'."
        )

    if new_parent_id in _descendant_ids(db, label.id):
        raise LabelMoveError(
            f"'{new_parent.name}' is already a part of '{label.name}', so making "
            f"'{label.name}' a part of it would form a cycle."
        )

    return new_parent


def plan_move(db: Session, label_id: int, new_parent_id: int | None) -> MoveImpact:
    """Report which annotated objects a move would invalidate, without performing it.

    Args:
        db: The database session.
        label_id: The label to move.
        new_parent_id: Its destination parent, or ``None`` for the top level.

    Returns:
        The move's impact; empty when nothing would break.

    Raises:
        LabelMoveError: If the label or the destination is not a legal move target.
    """
    label = db.query(Labels).filter_by(id=label_id).first()
    if label is None:
        raise LabelMoveError(f"Label with id {label_id} does not exist.")

    _resolve_target(db, label, new_parent_id)

    if label.parent_id == new_parent_id:
        return MoveImpact()

    # A contour carrying this label is legal only while its containing contour carries
    # the label's parent. Every nested one whose container will no longer match is
    # stranded by the move -- and if the label becomes top-level, every nested one is,
    # because a top-level label is a direct child of nothing.
    container = aliased(Contours)
    query = (
        db.query(
            Contours.id.label("contour_id"),
            Contours.mask_id.label("mask_id"),
            Contours.parent_id.label("old_parent_id"),
            Masks.image_id.label("image_id"),
        )
        .join(container, Contours.parent_id == container.id)
        .join(Masks, Contours.mask_id == Masks.id)
        .filter(Contours.label_id == label_id)
    )
    if new_parent_id is not None:
        query = query.filter(
            or_(container.label_id.is_(None), container.label_id != new_parent_id)
        )

    return MoveImpact(
        tuple(
            AffectedContour(
                contour_id=row.contour_id,
                image_id=row.image_id,
                mask_id=row.mask_id,
                old_parent_id=row.old_parent_id,
            )
            for row in query.all()
        )
    )


async def move_label(
        db: Session,
        label_id: int,
        new_parent_id: int | None,
        detach_affected: bool = False,
) -> MoveImpact:
    """Move a label under a new parent, or to the top level.

    Args:
        db: The database session.
        label_id: The label to move.
        new_parent_id: Its destination parent, or ``None`` for the top level.
        detach_affected: Whether to demote the annotated objects the move would strand
            to root level instead of refusing the move.

    Returns:
        The impact that was applied (empty when nothing had to be detached).

    Raises:
        LabelMoveError: If the destination is not a legal move target.
        LabelMoveBlocked: If annotations would be invalidated and ``detach_affected``
            was not set. Nothing is written in that case.
    """
    label = db.query(Labels).filter_by(id=label_id).first()
    if label is None:
        raise LabelMoveError(f"Label with id {label_id} does not exist.")

    impact = plan_move(db, label_id, new_parent_id)
    if impact.count and not detach_affected:
        raise LabelMoveBlocked(impact)

    if impact.affected:
        contour_ids = [affected.contour_id for affected in impact.affected]
        db.query(Contours).filter(Contours.id.in_(contour_ids)).update(
            {Contours.parent_id: None}, synchronize_session=False
        )

        # Detaching changes two sibling groups per contour -- the one it left and the
        # image's root level it joined -- and the child count of the parent it left.
        # Marked after the UPDATE so the group subqueries see the new membership.
        for affected in impact.affected:
            mark_contextual_stale_for_group(db, affected.mask_id, affected.old_parent_id)
            mark_contextual_stale_for_group(db, affected.mask_id, None)
        mark_relational_stale_for_parent(
            db, {affected.old_parent_id for affected in impact.affected}
        )
        for mask_id in {affected.mask_id for affected in impact.affected}:
            hierarchy_cache.invalidate(mask_id)

    label.parent_id = new_parent_id
    db.commit()

    if impact.count:
        logger.info(
            "Moved label %s under %s, detaching %d contour(s) to root level.",
            label_id, new_parent_id, impact.count,
        )
    return impact

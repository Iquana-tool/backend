from logging import getLogger

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.user import User
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.contours import (
    Contours,
    dual_write_geometry_metrics,
    reviewer_contour_association,
    save_contour_tree,
)
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.masks import Masks
from app.database.rejections import AnnotationRejections
from app.database.users import Users
from app.services import hierarchy_cache
from app.services.database_access.labels import get_label_hierarchy
from app.services.quantification import (
    mark_appearance_stale,
    mark_contextual_stale,
    mark_relational_stale,
)

logger = getLogger(__name__)

# kwargs to modify_contour that change a contour's filled pixels and therefore
# invalidate its APPEARANCE-tier metrics (see mark_appearance_stale).
_GEOMETRY_FIELDS = {"x", "y"}

# kwargs to modify_contour that change a contour's neighbor set and therefore invalidate
# its whole parent group's CONTEXTUAL-tier metrics (see mark_contextual_stale_for_group):
# x/y move the centroid, parent_id re-parents the contour into a different sibling group
# (invalidating both the old and the new group), and label_id affects any future
# label-filtered contextual variant (see Step 5 notes on same-label kNN).
_CONTEXTUAL_FIELDS = {"x", "y", "parent_id", "label_id"}


# --- Metric staleness invalidation -------------------------------------------------
#
# The geometry tier is recomputed synchronously on every write (see
# ``app.database.contours.dual_write_geometry_metrics``) and is therefore never stale.
# The appearance / contextual / relational tiers are computed lazily in batches, so every
# write path that can change their inputs must flag the affected rows here. Without this,
# ``compute_*_for_dataset(only_stale=True)`` would consider already-computed rows fresh
# and silently keep serving outdated values.
#
# The helpers below answer "WHICH contours are affected", and delegate the actual flag
# flip to the primitives in ``app.services.quantification``.


def mark_contextual_stale_for_group(db: Session, mask_id: int, parent_id: int | None) -> int:
    """Mark the contextual rows of a whole sibling GROUP stale.

    Contextual metrics (nearest-neighbour distances) are computed from a contour's
    same-parent siblings, so adding, moving or removing ONE member changes the correct
    value for every other member - the invalidation must therefore cover the group, not
    just the contour that changed.

    A group is identified by ``(mask_id, parent_id)``, with ``parent_id is None`` meaning
    the image's root level (all root contours of a mask are siblings of each other). This
    mirrors the grouping in
    ``app.services.quantification._contours_needing_contextual_compute`` exactly; if the
    two ever disagree, stale rows would be flagged but never recomputed.

    Args:
        db: The database session (caller controls commit).
        mask_id: The mask the group lives on.
        parent_id: The group's parent contour id, or ``None`` for the root level.

    Returns:
        The number of rows marked stale.
    """
    # Passed to the primitive as a SUBQUERY rather than a materialized id list: importing
    # a mask saves its contours one by one, so fetching the (steadily growing) group into
    # python on every save would make a large import quadratic.
    group_select = select(Contours.id).where(Contours.mask_id == mask_id)
    if parent_id is None:
        group_select = group_select.where(Contours.parent_id.is_(None))
    else:
        group_select = group_select.where(Contours.parent_id == parent_id)
    return mark_contextual_stale(db, group_select)


def mark_relational_stale_for_parent(db: Session, parent_ids) -> int:
    """Mark the relational rows (``n_children``) of the given PARENT contours stale.

    ``n_children`` is PARENT-TARGETED: it changes only when a CHILD is added, removed or
    re-parented under a contour, which affects exactly that one parent - never that
    parent's siblings. Unlike ``mark_contextual_stale_for_group`` this therefore does not
    fan out to a whole sibling group (where every sibling is a potential neighbor of every
    other). Callers pass the old and/or new parent of a contour that was created, deleted
    or re-parented. ``None`` entries are dropped: a root-level contour has no parent whose
    ``n_children`` count could change.

    Args:
        db: The database session (caller controls commit).
        parent_ids: An iterable of parent contour ids, possibly containing ``None``.

    Returns:
        The number of rows marked stale.
    """
    return mark_relational_stale(db, {pid for pid in parent_ids if pid is not None})


def invalidate_metrics_for_new_contours(db: Session, contours: list[Contours]) -> None:
    """Invalidate the lazily-computed metrics that newly saved contours affect.

    Called from ``save_contour_tree`` once the whole tree is persisted. The new contours
    have no rows of their own yet (so the batch jobs already treat them as "needs
    compute"); what this flags is the metrics of the EXISTING contours around them:

      * contextual - every prior member of each group the new contours joined gained a
        potential nearest neighbour,
      * relational - every parent that gained a child now has a different ``n_children``.

    Appearance metrics are intentionally untouched: they depend only on a contour's own
    pixels, which no other contour's arrival can change.

    Args:
        db: The database session (caller controls commit).
        contours: The freshly saved ``Contours`` rows.
    """
    if not contours:
        return
    for mask_id, parent_id in {(c.mask_id, c.parent_id) for c in contours}:
        mark_contextual_stale_for_group(db, mask_id, parent_id)
    mark_relational_stale_for_parent(db, {c.parent_id for c in contours})


async def get_contour(
        contour_id: int,
        db: Session
) -> Contour:
    """ Get a contour by its contour id. """
    existing_contour = db.query(Contours).filter_by(id=contour_id).first()
    if not existing_contour:
        raise KeyError(f"Contour with id {contour_id} does not exist")
    return Contour.from_db(existing_contour)


async def get_contours(
        contour_ids: list[int],
        db: Session
):
    contours_db = db.query(Contours).filter(Contours.id.in_(contour_ids)).all()
    return [Contour.from_db(contour_db) for contour_db in contours_db]


async def _check_contour_label(
        contour: Contour,
        new_label_id: int,
        db: Session
):
    """ Change the label of a contour. """
    # We need to ensure the integrity of our label hierarchy here, hence this is handled separately
    # For this we need the label hierarchy of the dataset, so we first fetch the dataset id
    dataset_id = (
        db.query(Images.dataset_id)
        .join(Masks, Masks.image_id == Images.id)
        .join(Contours, Contours.mask_id == Masks.id)
        .filter(Contours.id == contour.id)
        .scalar()
    )
    label_hierarchy = await get_label_hierarchy(dataset_id, db)
    parent_contour = db.query(Contours).filter_by(id=contour.parent_id).one_or_none()
    if parent_contour is None:
        parent_label_id = None
    else:
        parent_label_id = parent_contour.label_id
    if label_hierarchy.is_label_valid(new_label_id, parent_label_id):
        # New label is valid
        contour.label_id = new_label_id
        return contour
    else:
        raise ValueError(
            f"Label with id {new_label_id} is not valid for this dataset."
        )


async def may_review(contour_db: Contours, username: str, db: Session) -> bool:
    """Whether `username` is allowed to approve this particular contour.

    Independent review is off by default so a single owner working alone can still
    finish their own dataset. Datasets that turn it on (multi-annotator work, where
    "finished" has to mean "checked by someone else") refuse approvals from the
    person who authored the contour.
    """
    dataset_id = (
        db.query(Images.dataset_id)
        .join(Masks, Masks.image_id == Images.id)
        .filter(Masks.id == contour_db.mask_id)
        .scalar()
    )
    if dataset_id is None:
        return True
    requires_independent = (
        db.query(Datasets.require_independent_review).filter_by(id=dataset_id).scalar()
    )
    if not requires_independent:
        return True
    return contour_db.author_username != username


async def review_contour(
        contour_id: int,
        user: User,
        db: Session,
        strict: bool = True
):
    """Record `user` as having approved a contour.

    With `strict=False` an approval that separation of duties forbids is skipped
    silently rather than raising. That is what the label-edit path wants: changing
    a label should still succeed for an annotator who simply is not allowed to
    approve their own work.
    """
    contour_db = db.query(Contours).filter_by(id=contour_id).first()
    if not contour_db:
        raise KeyError(f"Contour with id {contour_id} does not exist")
    user_db = db.query(Users).filter_by(username=user.username).first()
    if user_db is None:
        raise KeyError(f"User {user.username} does not exist")

    if not await may_review(contour_db, user.username, db):
        if strict:
            raise PermissionError(
                "This dataset requires independent review: a contour cannot be "
                "approved by the person who created it."
            )
        return False

    # Compare ORM rows, not the Pydantic caller: `user in contour_db.reviewed_by`
    # is always False, which used to append duplicates here and make removal a no-op.
    if user_db not in contour_db.reviewed_by:
        contour_db.reviewed_by.append(user_db)
        db.commit()
        # reviewed_by rides along in the hierarchy payload, so an approval changes it.
        hierarchy_cache.invalidate(contour_db.mask_id)
    return True


async def delete_contour(
        contour_id: int,
        db: Session
):
    # Fetch the contour and all descendants in one query
    contour = (
        db.query(Contours)
        .filter_by(id=contour_id)
        .first()
    )
    if not contour:
        return

    # Read the group keys before the row goes away (the instance is detached afterwards).
    mask_id, parent_id = contour.mask_id, contour.parent_id

    # The surviving siblings lost a neighbour, and the parent lost a child. Flagging them
    # before the DELETE is safe: this contour's own rows are flagged too, but they are
    # removed by the CASCADE moments later. Descendants need no handling - they are
    # cascade-deleted along with their rows, so no group survives to be invalidated.
    mark_contextual_stale_for_group(db, mask_id, parent_id)
    mark_relational_stale_for_parent(db, [parent_id])

    # Delete the root contour (CASCADE will handle the rest)
    db.delete(contour)
    db.flush()
    db.commit()
    hierarchy_cache.invalidate(mask_id)


async def remove_review(
        contour_id: int,
        user: User,
        db: Session
):
    contour_db = db.query(Contours).filter_by(id=contour_id).first()
    user_db = db.query(Users).filter_by(username=user.username).first()
    if not contour_db:
        raise KeyError(f"Contour with id {contour_id} does not exist")
    # Same identity fix as in `review_contour`: comparing the Pydantic user against
    # ORM rows never matched, so this branch never fired and reviews were unremovable.
    if user_db is not None and user_db in contour_db.reviewed_by:
        contour_db.reviewed_by.remove(user_db)
        db.commit()
        hierarchy_cache.invalidate(contour_db.mask_id)


async def modify_contour(
        contour_id: int,
        db: Session,
        **kwargs
):
    """
        Modify a contour by its contour id and kwargs.
        For simple field updates (label_id, reviewed_by, etc.) this performs an
        in-place UPDATE rather than delete+re-insert, which preserves relationships
        and is significantly faster.
    """

    contour_db = db.query(Contours).filter_by(id=contour_id).first()
    if not contour_db:
        raise KeyError(f"Contour with id {contour_id} does not exist")

    # Snapshot the metric-relevant state before the update so we can tell afterwards what
    # actually changed and invalidate only what that change affects. The coordinate lists
    # are copied because the attribute is reassigned below, not mutated in place.
    old_parent_id = contour_db.parent_id
    old_x, old_y = list(contour_db.x or []), list(contour_db.y or [])

    for key, value in kwargs.items():
        if key == "label_id":
            # Validate label against dataset hierarchy before setting
            contour_schema = Contour.from_db(contour_db)
            contour_schema = await _check_contour_label(contour_schema, value, db)
            contour_db.label_id = contour_schema.label_id
        elif key == "reviewed_by":
            # reviewed_by is a list of usernames → resolve to User objects
            if value:
                reviewers = db.query(Users).filter(
                    Users.username.in_(value)
                ).all()
                contour_db.reviewed_by = reviewers
            else:
                contour_db.reviewed_by = []
        elif hasattr(contour_db, key):
            setattr(contour_db, key, value)

    if _GEOMETRY_FIELDS & kwargs.keys():
        # The contour's filled pixels may have changed shape/position: its appearance
        # metrics (mean color / intensity) no longer reflect the right pixels until
        # recomputed. Geometry-tier rows are not touched here; those are the caller's
        # responsibility to recompute synchronously (this endpoint does not, today).
        mark_appearance_stale(db, contour_id)

    if _CONTEXTUAL_FIELDS & kwargs.keys():
        # x/y move the centroid, parent_id re-parents into a different sibling group,
        # label_id affects any future label-filtered contextual variant (see Step 5
        # notes) - any of these invalidate the contour's (new) parent group.
        mark_contextual_stale_for_group(db, contour_db.mask_id, contour_db.parent_id)
        if "parent_id" in kwargs and old_parent_id != contour_db.parent_id:
            # Re-parenting also invalidates the OLD group it left (one fewer sibling).
            mark_contextual_stale_for_group(db, contour_db.mask_id, old_parent_id)

    if "parent_id" in kwargs and old_parent_id != contour_db.parent_id:
        # RELATIONAL: re-parenting moves this contour from one parent's child set to
        # another's, so BOTH the old parent (lost a child) and the new parent (gained one)
        # have a stale n_children count. Parent-targeted only - neither parent's siblings
        # are affected. None parents (root level) are no-ops.
        mark_relational_stale_for_parent(db, [old_parent_id, contour_db.parent_id])

    db.flush()

    geometry_changed = (list(contour_db.x or []) != old_x
                        or list(contour_db.y or []) != old_y)
    parent_changed = contour_db.parent_id != old_parent_id
    if geometry_changed or parent_changed:
        _invalidate_metrics_for_modified_contour(
            db, contour_db, old_parent_id,
            geometry_changed=geometry_changed, parent_changed=parent_changed,
        )

    db.commit()
    hierarchy_cache.invalidate(contour_db.mask_id)

    return True


def _invalidate_metrics_for_modified_contour(
        db: Session,
        contour_db: Contours,
        old_parent_id: int | None,
        geometry_changed: bool,
        parent_changed: bool,
) -> None:
    """Re-derive / invalidate the metrics affected by an edit to one contour.

    Label and review changes are deliberately not handled here: they do not feed into any
    metric value (the summary groups by label at query time), so nothing is invalidated
    for them.

    Args:
        db: The database session (caller controls commit).
        contour_db: The contour AFTER the update was applied.
        old_parent_id: The contour's parent before the update.
        geometry_changed: Whether the contour's coordinates changed.
        parent_changed: Whether the contour was re-parented.
    """
    if geometry_changed:
        # Geometry is the one tier that must never be stale, so it is recomputed in place
        # rather than flagged - otherwise the legacy columns and the geometry rows would
        # keep describing the shape the contour had before this edit.
        dual_write_geometry_metrics(db, contour_db.mask_id, [contour_db])
        # The contour covers different pixels now, so its appearance values are obsolete.
        mark_appearance_stale(db, contour_db.id)

    # Contextual: moving a contour changes the neighbour distances of everyone in its
    # group; re-parenting additionally changes them in the group it left.
    affected_groups = {(contour_db.mask_id, contour_db.parent_id)}
    if parent_changed:
        affected_groups.add((contour_db.mask_id, old_parent_id))
    for mask_id, parent_id in affected_groups:
        mark_contextual_stale_for_group(db, mask_id, parent_id)

    # Relational: only a re-parent changes any n_children, and it changes exactly two
    # (the parent that lost the child and the one that gained it).
    if parent_changed:
        mark_relational_stale_for_parent(db, [old_parent_id, contour_db.parent_id])


async def replace_contour(
        old_contour_id,
        new_contour_model,
        db: Session,
        author_username: str | None = None
):
    """ Replace a contour with a new one. """
    new_contour_model.id = old_contour_id
    contour = db.query(Contours).filter_by(id=old_contour_id).first()
    if not contour:
        return False
    # Keep the original author when the caller does not name one, so replacing a
    # contour's geometry does not silently reassign who is credited with it.
    author_username = author_username or contour.author_username
    mask_id, parent_id = contour.mask_id, contour.parent_id

    # A replace is delete + re-insert under the SAME id — logically an in-place
    # geometry update (e.g. an outline refinement), not a real deletion. But the
    # DELETE fires the ON DELETE CASCADE on annotation_rejections.contour_id, which
    # would wipe the reviewer's open send-backs for this object — and in the
    # correction queue, 404 the "Mark as done" that follows the fix. Detach the
    # rejections across the swap and re-point them to the reused id, so both the
    # rejections and their ids survive the geometry change.
    rejection_ids = [
        rid for (rid,) in
        db.query(AnnotationRejections.id).filter_by(contour_id=old_contour_id).all()
    ]
    if rejection_ids:
        db.query(AnnotationRejections).filter(
            AnnotationRejections.id.in_(rejection_ids)
        ).update({AnnotationRejections.contour_id: None}, synchronize_session=False)
        db.flush()

    db.query(Contours).filter_by(id=old_contour_id).delete()
    save_contour_tree(db, new_contour_model, mask_id, parent_id,
                      author_username=author_username)

    if rejection_ids:
        # The contour keeps its id (set above), so the FK is valid again.
        db.query(AnnotationRejections).filter(
            AnnotationRejections.id.in_(rejection_ids)
        ).update({AnnotationRejections.contour_id: old_contour_id},
                 synchronize_session=False)
    # The contour keeps its id but its geometry (and therefore its filled pixels) is
    # entirely new; any appearance rows written above the delete's CASCADE (none should
    # remain, since the CASCADE removed them with the old contour row) must not be
    # trusted stale-free. Mark defensively so a re-insert racing the CASCADE, or a future
    # change to how replacement is implemented, cannot leave stale appearance data behind.
    mark_appearance_stale(db, old_contour_id)
    # Same reasoning for CONTEXTUAL: the replaced contour's centroid may have moved, so
    # its whole parent group (same mask_id/parent_id, unchanged by the replace) needs
    # recomputing. save_contour_tree only dual-writes GEOMETRY rows for the new contour;
    # contextual staleness for the group is marked here explicitly, mirroring
    # mark_appearance_stale above.
    mark_contextual_stale_for_group(db, mask_id, parent_id)
    db.commit()
    hierarchy_cache.invalidate(mask_id)
    return True

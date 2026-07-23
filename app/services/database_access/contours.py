from logging import getLogger

from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.user import User
from sqlalchemy.orm import Session

from app.database.contours import Contours, save_contour_tree
from app.database.images import Images
from app.database.masks import Masks
from app.database.users import Users
from app.services.database_access.labels import get_label_hierarchy
from app.services.quantification import mark_appearance_stale, mark_contextual_stale, mark_relational_stale

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


def mark_relational_stale_for_parent(
        db: Session,
        parent_id: int | None,
) -> int:
    """Mark the RELATIONAL-tier rows of ``parent_id`` stale (its child set changed).

    ``n_children`` is PARENT-TARGETED: it changes only when a CHILD is added, removed or
    re-parented under this contour, which affects exactly this one parent - never the
    parent's siblings. This is the key difference from ``mark_contextual_stale_for_group``,
    which fans out to a whole sibling group because every sibling is a potential neighbor of
    every other; here only the single parent whose children changed is invalidated.

    A ``None`` ``parent_id`` (the child was/became a root-level contour, so no parent gained
    or lost it) is a no-op: root-level contours have no parent whose ``n_children`` count
    could change.

    Args:
        db: The database session (caller controls commit).
        parent_id: The id of the parent contour whose child count changed, or ``None`` for
            root level (no-op).

    Returns:
        The number of ``contour_metrics`` rows marked stale.
    """
    if parent_id is None:
        return 0
    return mark_relational_stale(db, [parent_id])


def _sibling_ids(db: Session, mask_id: int, parent_id: int | None) -> list[int]:
    """All contour ids sharing ``(mask_id, parent_id)`` (None = root level of that image)."""
    query = db.query(Contours.id).filter(Contours.mask_id == mask_id)
    query = query.filter(Contours.parent_id.is_(None)) if parent_id is None else query.filter(
        Contours.parent_id == parent_id
    )
    return [row.id for row in query.all()]


def mark_contextual_stale_for_group(
        db: Session,
        contour_id: int,
        mask_id: int | None = None,
        parent_id: int | None = None,
) -> int:
    """Mark CONTEXTUAL-tier rows stale for ``contour_id`` AND all of its same-parent siblings.

    Nearest-neighbour-style metrics are relational: every contour in a parent group is a
    potential neighbor of every other, so a change to ONE contour (moved, re-parented,
    added, or removed) invalidates the correct value for ALL of them, not just itself -
    the KDTree for the group has to be rebuilt regardless of which member changed.

    ``mask_id`` / ``parent_id`` can be passed explicitly for callers that already know
    them (e.g. ``delete_contour``, which must capture them BEFORE the delete removes the
    row, and ``modify_contour`` when ``parent_id`` itself changed - both the OLD and the
    NEW group need invalidating). When omitted, they are looked up from ``contour_id``;
    if the contour cannot be found either way, only ``contour_id`` itself is marked
    (harmless - a missing contour has no siblings left to protect).

    Args:
        db: The database session (caller controls commit).
        contour_id: The contour whose parent group should be invalidated. Marked stale
            itself even if it no longer exists in ``contours`` (ON DELETE CASCADE will
            already have removed its own rows in that case, so this is a no-op for it,
            but siblings still get marked correctly).
        mask_id: The contour's ``mask_id``, if already known (avoids a lookup).
        parent_id: The contour's ``parent_id`` (None = root level), if already known.

    Returns:
        The number of ``contour_metrics`` rows marked stale.
    """
    if mask_id is None:
        contour_db = db.query(Contours).filter_by(id=contour_id).one_or_none()
        if contour_db is None:
            return mark_contextual_stale(db, [contour_id])
        mask_id = contour_db.mask_id
        parent_id = contour_db.parent_id

    ids = set(_sibling_ids(db, mask_id, parent_id))
    ids.add(contour_id)
    return mark_contextual_stale(db, ids)


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


async def review_contour(
        contour_id: int,
        user: User,
        db: Session
):
    contour_db = db.query(Contours).filter_by(id=contour_id).first()
    user_db = db.query(Users).filter_by(username=user.username).first()
    if not contour_db:
        raise KeyError(f"Contour with id {contour_id} does not exist")
    if user not in contour_db.reviewed_by:
        contour_db.reviewed_by.append(user_db)
        db.commit()


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

    # Capture mask_id/parent_id BEFORE the delete so the surviving siblings' contextual
    # rows can still be invalidated afterwards (deleting a contour removes a neighbor,
    # so the remaining siblings' nn_distance/mean_knn_distance are now stale - including
    # the case where only one sibling is left, which now has no neighbor at all and must
    # have its row REMOVED, not just marked stale; see compute_and_store_metrics's
    # delete-then-insert, which drops rows a metric omits for an only-child contour).
    mask_id, parent_id = contour.mask_id, contour.parent_id

    # Delete the root contour (CASCADE will handle the rest)
    db.delete(contour)
    db.flush()
    mark_contextual_stale_for_group(db, contour_id, mask_id=mask_id, parent_id=parent_id)
    # RELATIONAL: the deleted contour's PARENT (if any) just lost a child, so its
    # n_children row is stale (parent-targeted, unlike the sibling-group contextual
    # invalidation above). The deleted contour's own relational row is gone via CASCADE.
    mark_relational_stale_for_parent(db, parent_id)
    db.commit()


async def remove_review(
        contour_id: int,
        user: User,
        db: Session
):
    contour_db = db.query(Contours).filter_by(id=contour_id).first()
    user_db = db.query(Users).filter_by(username=user.username).first()
    if not contour_db:
        raise KeyError(f"Contour with id {contour_id} does not exist")
    if user in contour_db.reviewed_by:
        contour_db.reviewed_by.remove(user_db)
        db.commit()


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

    # Capture the OLD group before any field is changed: if parent_id changes below, the
    # old group (this contour's siblings before the move) loses a member and must be
    # invalidated too, in addition to the new group it joins.
    old_mask_id, old_parent_id = contour_db.mask_id, contour_db.parent_id

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
        mark_contextual_stale_for_group(db, contour_id, mask_id=contour_db.mask_id, parent_id=contour_db.parent_id)
        if "parent_id" in kwargs and old_parent_id != contour_db.parent_id:
            # Re-parenting also invalidates the OLD group it left (one fewer sibling).
            mark_contextual_stale_for_group(db, contour_id, mask_id=old_mask_id, parent_id=old_parent_id)

    if "parent_id" in kwargs and old_parent_id != contour_db.parent_id:
        # RELATIONAL: re-parenting moves this contour from one parent's child set to
        # another's, so BOTH the old parent (lost a child) and the new parent (gained one)
        # have a stale n_children count. Parent-targeted only - neither parent's siblings
        # are affected. None parents (root level) are no-ops.
        mark_relational_stale_for_parent(db, old_parent_id)
        mark_relational_stale_for_parent(db, contour_db.parent_id)

    db.flush()
    db.commit()

    return True


async def replace_contour(
        old_contour_id,
        new_contour_model,
        db: Session
):
    """ Replace a contour with a new one. """
    new_contour_model.id = old_contour_id
    contour = db.query(Contours).filter_by(id=old_contour_id).first()
    if not contour:
        return False
    mask_id, parent_id = contour.mask_id, contour.parent_id
    db.query(Contours).filter_by(id=old_contour_id).delete()
    save_contour_tree(db, new_contour_model, mask_id, parent_id)
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
    mark_contextual_stale_for_group(db, old_contour_id, mask_id=mask_id, parent_id=parent_id)
    db.commit()
    return True

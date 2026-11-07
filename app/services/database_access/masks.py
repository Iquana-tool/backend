import os
from logging import getLogger
from pathlib import Path

import numpy as np
from PIL import Image
from iquana_toolbox.schemas.database.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.database.contours import Contour
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from app.database.contours import Contours, save_contour_tree
from app.schemas.annotation_contour import AnnotationContour
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.services import hierarchy_cache
from app.services.database_access import labels as labels_db

logger = getLogger(__name__)


def add_contour_to_hierarchy(
        hierarchy: ContourHierarchy,
        contour: Contour,
        label_hierarchy: labels_db.LabelHierarchy | None = None,
) -> Contour | None:
    """Fit and index a contour without crashing when no pixels remain.

    The pinned toolbox returns an empty list when ``fit_to_mask`` removes the
    complete contour. Treat that valid overlap case as "nothing to add" at the
    database boundary instead of passing the list back as a ``Contour``.

    When ``label_hierarchy`` is supplied, true siblings on the same level are
    distinguished from descendant labels (such as child annotations created before
    their parent) and ancestor labels. Descendant annotations are temporarily
    unparented/unresolved and must not subtract pixels from a new ancestor contour.
    """
    from iquana_toolbox.schemas.database.contour_hierarchy import logical_or_contours

    resolution_shape = (1000, 1000)

    # Determine which contours on the same level are true siblings that compete for pixels
    if contour.parent_id is None:
        same_level_contours = hierarchy.root_contours
    else:
        parent = hierarchy.id_to_contour.get(contour.parent_id)
        same_level_contours = parent.children if parent is not None else []

    same_level_contours = [c for c in same_level_contours if getattr(c, "hierarchy_enabled", True)]
    if label_hierarchy is not None and contour.label_id is not None:
        descendant_label_ids = labels_db.get_descendant_label_ids(label_hierarchy, contour.label_id)
        ancestor_label_ids = labels_db.get_ancestor_label_ids(label_hierarchy, contour.label_id)
        # Exclude descendant and ancestor labels from the collision mask
        competing_siblings = [
            c for c in same_level_contours
            if c.id != contour.id
            and c.label_id not in descendant_label_ids
            and c.label_id not in ancestor_label_ids
        ]
        if competing_siblings:
            level_mask = logical_or_contours(competing_siblings, resolution_shape)
            allowed_level = np.logical_not(level_mask)
        else:
            allowed_level = np.ones(resolution_shape, dtype=bool)
    else:
        allowed_level = np.logical_not(logical_or_contours(same_level_contours, resolution_shape))

    allowed_pixels = np.logical_and(
        allowed_level,
        hierarchy.get_parent_mask(contour.parent_id, resolution_shape),
    )
    fitted_contour, _changed = contour.fit_to_mask(allowed_pixels)
    if not isinstance(fitted_contour, Contour):
        return None

    if fitted_contour.parent_id is None:
        hierarchy.root_contours.append(fitted_contour)
    else:
        parent = hierarchy.id_to_contour.get(fitted_contour.parent_id)
        if parent is not None:
            parent.add_child(fitted_contour)
    if fitted_contour.id is not None:
        hierarchy.id_to_contour[fitted_contour.id] = fitted_contour
    hierarchy.label_id_to_contours.setdefault(fitted_contour.label_id, []).append(fitted_contour)
    return fitted_contour


async def get_mask(
        mask_id: int,
        db: Session
):
    mask = db.query(Masks).filter_by(id=mask_id).first()
    if mask is None:
        raise KeyError(f"No mask with id {mask_id}")
    return mask


async def delete_mask(
        mask_id: int,
        db: Session
):
    mask = db.query(Masks).filter_by(id=mask_id).one_or_none()
    if mask is None:
        return
    db.delete(mask)
    db.commit()


async def mark_mask_as_complete(
        mask_id: int,
        db: Session,
        username: str | None = None,
):
    mask = db.query(Masks).filter_by(id=mask_id).first()
    image = mask.image

    # Check if the mask is already finished
    if mask.fully_annotated:
        return

    # Generate the mask from contours
    contours_hierarchy = await get_contour_hierarchy_of_mask(mask_id, db)
    labels_hierarchy = await labels_db.get_label_hierarchy(image.dataset_id, db)

    await save_semantic_mask(
        contours_hierarchy.to_semantic_mask(
            height=image.height,
            width=image.width,
            label_id_to_value_map=labels_hierarchy.id_to_value_map
        ),
        Path(str(mask.file_path))
    )

    # Mark the mask as finished
    mask.mark_finished_by(username)
    db.commit()


async def mark_mask_as_incomplete(
        mask_id: int,
        db: Session
):
    existing_mask = db.query(Masks).filter_by(id=mask_id).first()
    # Check if the mask is already unfinished
    if not existing_mask.fully_annotated:
        return
    # Mark the mask as unfinished
    existing_mask.fully_annotated = False
    db.commit()


async def save_semantic_mask(
        semantic_mask: np.ndarray,
        file_path: Path,
):
    # Convert to PIL image & save
    semantic_mask = Image.fromarray(semantic_mask, mode="L")  # <- Saves as a greyscale image, tiny file size
    semantic_mask.save(file_path)


async def create_new_mask(
        image_id: int,
        dataset_folder: str,
        db: Session,
):
    mask_folder = Path(dataset_folder) / "masks"
    os.makedirs(mask_folder, exist_ok=True)
    mask_path = mask_folder / f"{image_id}.png"
    new_mask = Masks(
        image_id=image_id,
        file_path=str(mask_path),
    )
    db.add(new_mask)
    db.flush()
    return new_mask


async def get_contour_hierarchy_of_mask(
        mask_id: int,
        db: Session
):
    """Build the contour hierarchy of a mask.

    ``reviewed_by`` is eager-loaded: ``Contour.from_db`` reads it for every contour, and
    without this each one costs its own SELECT, which is what made large masks load
    slowly (and, on a cold connection, appear to hang).
    """
    contours_query = (db.query(Contours)
                      .options(selectinload(Contours.reviewed_by))
                      .filter_by(mask_id=mask_id)
                      .all())
    size = await get_size_of_mask(mask_id, db)
    hierarchy = ContourHierarchy.from_query(contours_query, height=size["height"], width=size["width"])
    policies = {row.id: row.hierarchy_enabled for row in contours_query}
    def with_policy(contour):
        return AnnotationContour(**contour.model_dump(exclude={"children"}),
                                 hierarchy_enabled=policies[contour.id],
                                 children=[with_policy(child) for child in contour.children])
    roots = [with_policy(contour) for contour in hierarchy.root_contours]
    by_id = {}
    by_label = {}
    def index(contour):
        by_id[contour.id] = contour
        by_label.setdefault(contour.label_id, []).append(contour)
        for child in contour.children:
            index(child)
    for contour in roots:
        index(contour)
    return ContourHierarchy(root_contours=roots, id_to_contour=by_id, label_id_to_contours=by_label)


def contour_fingerprint(mask_id: int, db: Session) -> tuple[int, int]:
    """A cheap ``(row count, highest id)`` signature of a mask's contours.

    Used to spot contour writes made by another process -- batch inference runs in a
    Celery worker, whose cache invalidations never reach the API process. One aggregate
    query is a rounding error next to rebuilding the hierarchy it guards.
    """
    row = (db.query(func.count(Contours.id), func.coalesce(func.max(Contours.id), 0))
           .filter(Contours.mask_id == mask_id)
           .one())
    return int(row[0]), int(row[1])


async def get_cached_contour_hierarchy_of_mask(
        mask_id: int,
        db: Session
) -> tuple[ContourHierarchy, dict]:
    """Return ``(hierarchy, client_payload)`` for a mask, from cache when possible.

    Read-only callers only. The returned hierarchy is shared with every other reader of
    the same mask, so mutating it (``add_contour``, re-parenting) would corrupt what the
    next reader sees -- those paths must keep using
    :func:`get_contour_hierarchy_of_mask`, which always rebuilds.
    """
    fingerprint = contour_fingerprint(mask_id, db)
    cached = hierarchy_cache.get(mask_id, fingerprint)
    if cached is not None:
        logger.debug("Serving contour hierarchy of mask %s from cache.", mask_id)
        return cached

    hierarchy = await get_contour_hierarchy_of_mask(mask_id, db)
    return hierarchy, hierarchy_cache.put(mask_id, fingerprint, hierarchy)


async def add_contours_from_hierarchy(
        mask_id: int,
        hierarchy: ContourHierarchy,
        db: Session,
        author_username: str | None = None
):
    """ Delete all contours of a mask and then add the hierarchy to it."""
    # 1. Delete all contours of the mask
    await delete_all_contours_of_mask(mask_id, db=db)

    # 2. Populate with new contours from the given hierarchy.
    #    Metric invalidation is skipped: step 1 removed every contour of this mask, and
    #    with them (ON DELETE CASCADE) every metric row that could have gone stale. Since
    #    contextual groups and relational parents never span masks, there is nothing left
    #    to flag - and doing it per contour would rescan the whole growing mask each time.
    for root_contours in hierarchy.root_contours:
        save_contour_tree(db, root_contours, mask_id, author_username=author_username,
                          invalidate_metrics=False)
    hierarchy_cache.invalidate(mask_id)


def adopt_descendants_for_parent(
        db: Session,
        mask_id: int,
        parent_row: Contours,
        parent_schema: Contour,
        label_hierarchy: labels_db.LabelHierarchy,
        hierarchy: ContourHierarchy | None = None,
        reparented_info: list[dict] | None = None,
) -> list[int]:
    """Identify and adopt contained descendants, tolerating tiny raster boundary errors.

    Preserves child IDs, geometries, labels, and provenance. Overlaps beyond the
    boundary tolerance are neither adopted nor trimmed, leaving them intact
    for user review. Annotations already attached to an unrelated branch are never
    moved; only unparented descendants or children attached to an ancestor contour
    (inserting a missing intermediate parent) are eligible.
    """
    from app.services.database_access.contours import _invalidate_metrics_for_modified_contour

    if parent_schema.label_id is None or parent_row.hierarchy_enabled is False:
        return []

    child_label_ids = labels_db.get_immediate_child_label_ids(label_hierarchy, parent_schema.label_id)
    if not child_label_ids:
        return []

    # Allowed previous parents: unparented (None), the new parent's own parent,
    # or an ancestor of the new parent (intermediate parent insertion).
    allowed_parent_ids: set[int | None] = {None}
    if parent_row.parent_id is not None:
        allowed_parent_ids.add(parent_row.parent_id)
        curr_pid = parent_row.parent_id
        while curr_pid is not None:
            if hierarchy is not None and curr_pid in hierarchy.id_to_contour:
                ancestor_pid = hierarchy.id_to_contour[curr_pid].parent_id
            else:
                ancestor_row = db.query(Contours.parent_id).filter_by(id=curr_pid).first()
                ancestor_pid = ancestor_row.parent_id if ancestor_row else None
            if ancestor_pid is not None:
                allowed_parent_ids.add(ancestor_pid)
            curr_pid = ancestor_pid

    query = db.query(Contours).filter(
        Contours.mask_id == mask_id,
        Contours.id != parent_row.id,
        Contours.hierarchy_enabled.is_(True),
        Contours.label_id.in_(child_label_ids),
    )
    candidates = [row for row in query.all() if row.parent_id in allowed_parent_ids]
    if not candidates:
        return []

    resolution = (1000, 1000)
    parent_mask = parent_schema.to_binary_mask(resolution[0], resolution[1])
    if not np.any(parent_mask):
        return []

    # At most two comparison pixels of boundary error and 0.1% of child area.
    # Both bounds matter: a small area alone could admit a long, thin protrusion.
    padded_parent = np.pad(parent_mask, 2)
    tolerant_parent_mask = np.logical_or.reduce([
        padded_parent[dy:dy + resolution[0], dx:dx + resolution[1]]
        for dy in range(5) for dx in range(5)
    ])
    boundary_margin = 2 / resolution[0]

    parent_xs = parent_schema.x or []
    parent_ys = parent_schema.y or []
    if not parent_xs or not parent_ys:
        return []
    min_px, max_px = min(parent_xs), max(parent_xs)
    min_py, max_py = min(parent_ys), max(parent_ys)

    adopted_ids: list[int] = []
    for child_row in candidates:
        child_xs = child_row.x or []
        child_ys = child_row.y or []
        if len(child_xs) < 3 or len(child_ys) < 3:
            continue

        min_cx, max_cx = min(child_xs), max(child_xs)
        min_cy, max_cy = min(child_ys), max(child_ys)
        # Allow the same two-pixel margin as the raster containment check.
        if (min_cx < min_px - boundary_margin or max_cx > max_px + boundary_margin
                or min_cy < min_py - boundary_margin or max_cy > max_py + boundary_margin):
            continue

        child_contour = Contour(
            id=child_row.id,
            label_id=child_row.label_id,
            parent_id=child_row.parent_id,
            x=child_xs,
            y=child_ys,
        )
        child_mask = child_contour.to_binary_mask(resolution[0], resolution[1])
        if not np.any(child_mask):
            continue

        outside_pixels = int(np.count_nonzero(child_mask & ~parent_mask))
        within_tolerance = (
            outside_pixels <= np.count_nonzero(child_mask) * 0.001
            and not np.any(child_mask & ~tolerant_parent_mask)
        )
        if within_tolerance:
            old_parent_id = child_row.parent_id
            child_row.parent_id = parent_row.id
            _invalidate_metrics_for_modified_contour(
                db, child_row, old_parent_id,
                geometry_changed=False, parent_changed=True,
            )
            adopted_ids.append(child_row.id)
            if reparented_info is not None:
                reparented_info.append({
                    "contour_id": child_row.id,
                    "before_parent_id": old_parent_id,
                })

            # Update in-memory hierarchy if loaded
            if hierarchy is not None:
                c_in_h = hierarchy.id_to_contour.get(child_row.id)
                if c_in_h is not None:
                    c_in_h.parent_id = parent_row.id
                    if old_parent_id is None and c_in_h in hierarchy.root_contours:
                        hierarchy.root_contours.remove(c_in_h)
                    elif old_parent_id is not None and old_parent_id in hierarchy.id_to_contour:
                        old_p = hierarchy.id_to_contour[old_parent_id]
                        if c_in_h in old_p.children:
                            old_p.children.remove(c_in_h)
                    if parent_row.id in hierarchy.id_to_contour:
                        hierarchy.id_to_contour[parent_row.id].add_child(c_in_h)

    return adopted_ids


async def get_size_of_mask(
        mask_id: int,
        db: Session
):
    result = (db.query(Masks.id, Images.height, Images.width)
              .join(Images, Masks.image_id == Images.id)
              .filter(Masks.id == mask_id).first())
    return {
        "height": result.height,
        "width": result.width,
    }


async def add_contour_to_mask(
        mask_id: int,
        contour_to_add: Contour,
        db: Session,
        check_hierarchy: bool = True,
        author_username: str | None = None,
        reparented_contour_ids: list[int] | None = None,
        reparented_info: list[dict] | None = None,
        use_hierarchy: bool = True,
):
    """
    Add a contour to an existing mask and fit it into the hierarchy.
    :param mask_id: ID of the mask the contour should be added to.
    :param contour_to_add: Contour to be added to the mask.
    :param check_hierarchy: Whether to fit the contour into the existing hierarchy. This is true by default and should
        only be set to False, if the contour was already fitted. Otherwise, might lead to inconsistencies. When False,
        skips creating the hierarchy.
    :param db: Database session
    :param use_hierarchy: False saves independent geometry at root and prevents later adoption.
    :param author_username: The human whose session created this contour. Comes from
        the authenticated caller, never from the request payload, because separation
        of duties on review keys off it.
    :param reparented_contour_ids: Optional list to collect IDs of existing descendant
        contours adopted under the newly created parent contour.
    :param reparented_info: Optional list to collect dictionaries of adopted descendant
        info (``{"contour_id": int, "before_parent_id": int | None}``).
    """
    dataset_id = (db.query(Images.dataset_id)
                  .join(Masks, Masks.image_id == Images.id)
                  .filter(Masks.id == mask_id)
                  .scalar())
    label_hierarchy = None
    if dataset_id is not None:
        has_labels = db.query(Labels.id).filter_by(dataset_id=dataset_id).first() is not None
        if has_labels:
            try:
                label_hierarchy = await labels_db.get_label_hierarchy(dataset_id, db)
            except Exception as exc:
                logger.error(
                    "Failed to load taxonomy for dataset %s on mask %s: %s",
                    dataset_id, mask_id, exc, exc_info=True
                )
                raise RuntimeError(
                    f"Failed to load taxonomy for dataset {dataset_id} on mask {mask_id}: {exc}"
                ) from exc
        else:
            logger.info("Dataset %s has no labels configured in database", dataset_id)
            label_hierarchy = None

    if not use_hierarchy:
        if contour_to_add.children:
            raise ValueError("Independent objects must be saved individually.")
        contour_to_add.parent_id = None

    if check_hierarchy and use_hierarchy:
        hierarchy = await get_contour_hierarchy_of_mask(mask_id, db)
        contour_to_add = add_contour_to_hierarchy(hierarchy, contour_to_add, label_hierarchy=label_hierarchy)
        if contour_to_add is None:
            return None
    else:
        hierarchy = None

    # Add contour to the database
    entry = save_contour_tree(db, contour_to_add, mask_id, parent_id=contour_to_add.parent_id,
                              author_username=author_username)
    # Callers log and announce the whole saved tree, so every node needs its new id,
    # not just the root (nested model output such as vessels inside a choroid).
    # Must copy IDs before adoption runs, so older adopted children are not mixed into the child list!
    _copy_saved_ids(db, contour_to_add, entry.id)
    entry.hierarchy_enabled = use_hierarchy

    # Adopt eligible descendant annotations atomically in the same transaction
    adopted_ids: list[int] = []
    adopted_info: list[dict] = []
    if use_hierarchy and label_hierarchy is not None:
        adopted_ids = adopt_descendants_for_parent(
            db, mask_id, entry, contour_to_add, label_hierarchy, hierarchy=hierarchy,
            reparented_info=adopted_info,
        )

    db.commit()
    hierarchy_cache.invalidate(mask_id)

    # SVG path computation for the frontend
    # Get image dimensions and compute path
    size = await get_size_of_mask(mask_id, db)
    contour_to_add.compute_path(
        image_width=size["width"],
        image_height=size["height"],
    )
    contour_to_add = AnnotationContour(**(contour_to_add.model_dump() | {"hierarchy_enabled": use_hierarchy}))
    contour_to_add._reparented_ids = adopted_ids
    contour_to_add._reparented_info = adopted_info
    if reparented_contour_ids is not None:
        reparented_contour_ids.extend(adopted_ids)
    if reparented_info is not None:
        reparented_info.extend(adopted_info)

    return contour_to_add


def _copy_saved_ids(db: Session, contour_schema: Contour, contour_id: int):
    """Give each node of a just-saved schema tree the id of its row.

    ``save_contour_tree`` saves children in order, so a node's child rows sorted by id
    line up with its schema children.
    """
    contour_schema.id = contour_id
    if not contour_schema.children:
        return
    child_ids = [row.id for row in db.query(Contours.id).filter(Contours.parent_id == contour_id)
                 .order_by(Contours.id)]
    for child_schema, child_id in zip(contour_schema.children, child_ids):
        _copy_saved_ids(db, child_schema, child_id)


async def delete_all_contours_of_mask(
        mask_id: int,
        db: Session,
        unreviewed_only: bool = False,
):
    # Local import: annotation_history reaches back into the contour access layer.
    from app.services.database_access.annotation_history import clear_for_mask

    if unreviewed_only:
        db.query(Contours).filter(Contours.mask_id == mask_id, ~Contours.reviewed_by.any()).delete()
    else:
        db.query(Contours).filter_by(mask_id=mask_id).delete()
    # The undo history describes contours that are gone now. Left in place, an undo
    # recorded before this wipe would put one of them back into the emptied mask.
    clear_for_mask(mask_id, db)
    mask = db.query(Masks).filter_by(id=mask_id).first()
    mask.fully_annotated = False
    db.commit()
    hierarchy_cache.invalidate(mask_id)

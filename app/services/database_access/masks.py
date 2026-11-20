import os
from logging import getLogger
from pathlib import Path

import cv2
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


def _to_safe_normalized(coords: np.ndarray, dim: int) -> list[float]:
    """Convert integer pixel coordinates to normalized floats in [0, 1].

    Guarantees that when the normalized floats are multiplied by `dim` and cast to
    integer via `.astype(int)` (the rasterizer in `iquana_toolbox`), floating-point
    representation does not truncate downwards (e.g. 508.0 / 4001 * 4001 = 507.999... -> 507).
    Uses `np.nextafter(..., 1.0)` to nudge only underflowing coordinates by 1 ULP (~2e-17).
    """
    coords_f = np.asarray(coords, dtype=np.float64)
    norm = coords_f / dim
    underflow = (norm * dim).astype(int) < np.round(coords_f).astype(int)
    norm[underflow] = np.nextafter(norm[underflow], 1.0)
    return norm.tolist()


def fit_contour_at_image_resolution(
        contour: Contour,
        parent: Contour | None,
        competing_siblings: list[Contour],
        image_shape: tuple[int, int],
) -> tuple[Contour | None, bool]:
    """Check constraints and optionally refit a contour within its bounding box at image resolution.

    Limits rasterization to the contour's padded bounding box to avoid full-image mask
    allocations. Checks both parent containment and competing sibling non-overlap.
    If the contour is valid without clipping, returns (contour, False).
    If clipping is required, refits against allowed pixels (parent & ~siblings) within
    the crop at native image resolution and returns (fitted_contour, True).
    If no pixels remain, returns (None, True).
    """
    if not contour.x or not contour.y or len(contour.x) < 3 or len(contour.y) < 3 or len(contour.x) != len(contour.y):
        return None, True

    if parent is None and not competing_siblings:
        return contour, False

    h, w = image_shape

    pad = 4
    x_min, x_max = min(contour.x) * w, max(contour.x) * w
    y_min, y_max = min(contour.y) * h, max(contour.y) * h
    x0 = max(0, int(np.floor(x_min)) - pad)
    x1 = min(w, int(np.ceil(x_max)) + pad)
    y0 = max(0, int(np.floor(y_min)) - pad)
    y1 = min(h, int(np.ceil(y_max)) + pad)
    w_crop, h_crop = x1 - x0, y1 - y0
    if w_crop <= 0 or h_crop <= 0:
        return None, True

    c_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
    c_pts = (contour.to_rescaled_contour(h, w) - np.array([[[x0, y0]]])).astype(np.int32)
    cv2.drawContours(c_mask, [c_pts], -1, 1, cv2.FILLED)

    if parent is None:
        p_mask = np.ones((h_crop, w_crop), dtype=np.uint8)
    else:
        p_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
        p_pts = (parent.to_rescaled_contour(h, w) - np.array([[[x0, y0]]])).astype(np.int32)
        cv2.drawContours(p_mask, [p_pts], -1, 1, cv2.FILLED)

    s_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
    intersecting_siblings: list[Contour] = []
    for s in competing_siblings:
        if not s.x or not s.y or len(s.x) < 3 or len(s.y) < 3:
            continue
        sx_min, sx_max = min(s.x) * w, max(s.x) * w
        sy_min, sy_max = min(s.y) * h, max(s.y) * h
        if sx_max < x0 or sx_min > x1 or sy_max < y0 or sy_min > y1:
            continue
        intersecting_siblings.append(s)
        s_pts = (s.to_rescaled_contour(h, w) - np.array([[[x0, y0]]])).astype(np.int32)
        cv2.drawContours(s_mask, [s_pts], -1, 1, cv2.FILLED)

    allowed_crop = (p_mask == 1) & (s_mask == 0)
    outside_allowed = np.any((c_mask == 1) & (~allowed_crop))

    # Continuous vertex containment tests
    if not outside_allowed and parent is not None:
        p_pts_f = np.column_stack([np.array(parent.x) * w, np.array(parent.y) * h]).astype(np.float32)
        for x, y in zip(contour.x, contour.y):
            if cv2.pointPolygonTest(p_pts_f, (float(x * w), float(y * h)), measureDist=False) < 0:
                outside_allowed = True
                break

    if not outside_allowed:
        for s in intersecting_siblings:
            s_pts_f = np.column_stack([np.array(s.x) * w, np.array(s.y) * h]).astype(np.float32)
            for x, y in zip(contour.x, contour.y):
                if cv2.pointPolygonTest(s_pts_f, (float(x * w), float(y * h)), measureDist=False) > 0:
                    outside_allowed = True
                    break
            if outside_allowed:
                break

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Fast path for contours well inside allowed region:
    # If a contour does not touch or come within 1 pixel of ~allowed_crop,
    # OpenCV fixed-point scanline discrepancies (< 1 pixel) cannot touch it.
    # Return original vertices immediately without building full-image masks.
    if not outside_allowed:
        c_near_boundary = np.any(cv2.dilate(c_mask, kernel) & (~allowed_crop))
        if not c_near_boundary:
            return contour, False

    # Bound full-image validation and refitting to the crop window.
    # Rather than holding full-image masks for each sibling, extract crop slices immediately.
    if parent is not None:
        p_full = parent.to_binary_mask(h, w)
        p_mask_refit = p_full[y0:y1, x0:x1]
        del p_full
    else:
        p_mask_refit = np.ones((h_crop, w_crop), dtype=np.uint8)

    s_mask_refit = np.zeros((h_crop, w_crop), dtype=np.uint8)
    for s in intersecting_siblings:
        s_pts = (s.to_rescaled_contour(h, w) - np.array([[[x0, y0]]])).astype(np.int32)
        s_local = np.zeros((h_crop, w_crop), dtype=np.uint8)
        cv2.drawContours(s_local, [s_pts], -1, 1, cv2.FILLED)
        if np.any(cv2.dilate(c_mask, kernel) & (s_local == 1)):
            s_full = s.to_binary_mask(h, w)
            s_mask_refit |= s_full[y0:y1, x0:x1]
            del s_full
        else:
            s_mask_refit |= s_local

    # If the contour was not initially outside, check whether the boundary scanlines in full-image
    # parent/sibling rasters exclude any of its pixels.
    if not outside_allowed:
        if not np.any((c_mask == 1) & (p_mask_refit == 0)) and not np.any((c_mask == 1) & (s_mask_refit == 1)):
            return contour, False

    # Refit inside crop using both parent and sibling constraints sampled from full-image rasters.
    allowed_refit = (p_mask_refit == 1) & (s_mask_refit == 0)
    new_crop = (c_mask == 1) & allowed_refit
    if not np.any(new_crop):
        return None, True

    fitted = Contour.from_binary_mask(
        new_crop,
        only_return_biggest_contour=True,
        **contour.model_dump(exclude={"x", "y", "path", "quantification"})
    )
    if not isinstance(fitted, Contour):
        return None, True

    # Convert crop integer boundary coordinates back to normalized coordinates without downward float truncation
    px = np.round(np.array(fitted.x) * w_crop) + x0
    py = np.round(np.array(fitted.y) * h_crop) + y0
    fitted.x = _to_safe_normalized(px, w)
    fitted.y = _to_safe_normalized(py, h)

    # Validate clipped contours against the full-image raster within the crop window
    def _is_raster_valid(cand: Contour) -> bool:
        cand_crop = cand.to_binary_mask(h, w)[y0:y1, x0:x1]
        if np.any(cand_crop & ~p_mask_refit):
            return False
        if np.any(cand_crop & s_mask_refit):
            return False
        return True

    if not _is_raster_valid(fitted):
        curr_mask = new_crop.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fitted_valid = None
        for _ in range(3):
            curr_mask = cv2.erode(curr_mask, kernel)
            if not np.any(curr_mask):
                break
            candidate_eroded = Contour.from_binary_mask(
                curr_mask,
                only_return_biggest_contour=True,
                **contour.model_dump(exclude={"x", "y", "path", "quantification"}),
            )
            if not isinstance(candidate_eroded, Contour):
                break
            px_e = np.round(np.array(candidate_eroded.x) * w_crop) + x0
            py_e = np.round(np.array(candidate_eroded.y) * h_crop) + y0
            candidate_eroded.x = _to_safe_normalized(px_e, w)
            candidate_eroded.y = _to_safe_normalized(py_e, h)
            if _is_raster_valid(candidate_eroded):
                fitted_valid = candidate_eroded
                break
        if fitted_valid is None:
            return None, True
        fitted = fitted_valid

    return fitted, True


def check_parent_containment(
        contour: Contour,
        parent: Contour | None,
        image_shape: tuple[int, int] | None = None,
) -> bool:
    """Verify that a child contour is strictly contained within its parent at image resolution."""
    if parent is None:
        return True
    effective_shape = (
        image_shape
        if (image_shape is not None and image_shape[0] > 0 and image_shape[1] > 0)
        else (4000, 4000)
    )
    _, changed = fit_contour_at_image_resolution(contour, parent, [], effective_shape)
    return not changed


def add_contour_to_hierarchy(
        hierarchy: ContourHierarchy,
        contour: Contour,
        label_hierarchy: labels_db.LabelHierarchy | None = None,
        image_shape: tuple[int, int] | None = None,
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

    if not contour.x or not contour.y or len(contour.x) < 3 or len(contour.y) < 3 or len(contour.x) != len(contour.y):
        return None

    if image_shape is None:
        image_shape = getattr(hierarchy, "_image_shape", None)

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
        competing_siblings = [c for c in same_level_contours if c.id != contour.id]
        if competing_siblings:
            level_mask = logical_or_contours(competing_siblings, resolution_shape)
            allowed_level = np.logical_not(level_mask)
        else:
            allowed_level = np.ones(resolution_shape, dtype=bool)

    if contour.parent_id is None and not competing_siblings:
        fitted_contour = contour
    else:
        allowed_pixels = np.logical_and(
            allowed_level,
            hierarchy.get_parent_mask(contour.parent_id, resolution_shape),
        )
        fitted_contour, changed = contour.fit_to_mask(allowed_pixels)
        if not isinstance(fitted_contour, Contour):
            return None
        # The comparison grid must not quantize boundaries that need no clipping,
        # provided the contour respects both parent and sibling constraints at image resolution.
        if not changed:
            parent = hierarchy.id_to_contour.get(contour.parent_id) if contour.parent_id is not None else None
            effective_shape = (
                image_shape
                if (image_shape is not None and image_shape[0] > 0 and image_shape[1] > 0)
                else (4000, 4000)
            )
            fitted_at_res, changed_at_res = fit_contour_at_image_resolution(
                contour, parent, competing_siblings, effective_shape
            )
            if not changed_at_res:
                fitted_contour = contour
            else:
                if not isinstance(fitted_at_res, Contour):
                    return None
                fitted_contour = fitted_at_res

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
    res_hierarchy = ContourHierarchy(root_contours=roots, id_to_contour=by_id, label_id_to_contours=by_label)
    res_hierarchy._image_shape = (size["height"], size["width"])
    return res_hierarchy


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
    if not contour_to_add.x or not contour_to_add.y or len(contour_to_add.x) < 3 or len(contour_to_add.y) < 3 or len(contour_to_add.x) != len(contour_to_add.y):
        return None

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

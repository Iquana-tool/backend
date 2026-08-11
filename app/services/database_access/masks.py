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
from app.database.images import Images
from app.database.masks import Masks
from app.services import hierarchy_cache
from app.services.database_access import labels as labels_db

logger = getLogger(__name__)


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
        db: Session
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
    mask.fully_annotated = True
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
    return ContourHierarchy.from_query(contours_query,
                                       height=size["height"],
                                       width=size["width"]
                                       )


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


async def get_size_of_mask(
        mask_id: int,
        db: Session
):
    print(mask_id)
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
):
    """
    Add a contour to an existing mask and fit it into the hierarchy.
    :param mask_id: ID of the mask the contour should be added to.
    :param contour_to_add: Contour to be added to the mask.
    :param check_hierarchy: Whether to fit the contour into the existing hierarchy. This is true by default and should
        only be set to False, if the contour was already fitted. Otherwise, might lead to inconsistencies. When False,
        skips creating the hierarchy.
    :param db: Database session
    :param author_username: The human whose session created this contour. Comes from
        the authenticated caller, never from the request payload, because separation
        of duties on review keys off it.
    """
    if check_hierarchy:
        hierarchy = await get_contour_hierarchy_of_mask(mask_id, db)
        contour_to_add, changed = hierarchy.add_contour(contour_to_add)
    # Add contour to the database
    entry = save_contour_tree(db, contour_to_add, mask_id, parent_id=contour_to_add.parent_id,
                              author_username=author_username)
    db.commit()
    hierarchy_cache.invalidate(mask_id)
    contour_to_add.id = entry.id

    # SVG path computation for the frontend
    # Get image dimensions and compute path
    size = await get_size_of_mask(mask_id, db)
    contour_to_add.compute_path(
        image_width=size["width"],
        image_height=size["height"],
    )
    return contour_to_add


async def delete_all_contours_of_mask(
        mask_id: int,
        db: Session,
        unreviewed_only: bool = False,
):
    if unreviewed_only:
        db.query(Contours).filter(Contours.mask_id == mask_id, ~Contours.reviewed_by.any()).delete()
    else:
        db.query(Contours).filter_by(mask_id=mask_id).delete()
    mask = db.query(Masks).filter_by(id=mask_id).first()
    mask.fully_annotated = False
    db.commit()
    hierarchy_cache.invalidate(mask_id)

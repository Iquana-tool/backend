import os
from logging import getLogger
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import Depends
from iquana_toolbox.schemas.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.contours import Contour
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.contours import Contours, save_contour_tree
from app.database.images import Images
from app.database.masks import Masks
from app.services.database_access import labels as labels_db

logger = getLogger(__name__)


async def get_mask(
        mask_id: int,
        db: Session = Depends(get_session)
):
    mask = db.query(Masks).filter_by(id=mask_id).first()
    if mask is None:
        raise KeyError(f"No mask with id {mask_id}")
    return mask


async def delete_mask(
        mask_id: int,
        db: Session = Depends(get_session)
):
    mask = db.query(Masks).filter_by(id=mask_id).one_or_none()
    db.delete(mask)
    db.commit()


async def mark_mask_as_complete(
        mask_id: int,
        db: Session = Depends(get_session)
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
        db: Session = Depends(get_session)
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


async def get_contour_hierarchy_of_mask(mask_id: int, db: Session = Depends(get_session)):
    contours_query = db.query(Contours).filter_by(mask_id=mask_id).all()
    size = await get_size_of_mask(mask_id, db)
    return ContourHierarchy.from_query(contours_query,
                                       height=size["height"],
                                       width=size["width"]
                                       )


async def get_size_of_mask(mask_id: int, db: Session):
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
        db: Session = Depends(get_session),
        check_hierarchy: bool = True,
):
    """
    Add a contour to an existing mask and fit it into the hierarchy.
    :param mask_id: ID of the mask the contour should be added to.
    :param contour_to_add: Contour to be added to the mask.
    :param check_hierarchy: Whether to fit the contour into the existing hierarchy. This is true by default and should
        only be set to False, if the contour was already fitted. Otherwise, might lead to inconsistencies. When False,
        skips creating the hierarchy.
    :param db: Database session
    """
    if check_hierarchy:
        hierarchy = await get_contour_hierarchy_of_mask(mask_id, db)
        contour_to_add, changed = hierarchy.add_contour(contour_to_add)
    # Add contour to the database
    entry = save_contour_tree(db, contour_to_add, mask_id)
    db.commit()
    contour_to_add.id = entry.id

    # SVG path computation for the frontend
    # Get image dimensions and compute path
    size = await get_size_of_mask(mask_id, db)
    contour_to_add.compute_path(
        image_width=size["width"],
        image_height=size["height"],
    )
    return contour_to_add


async def delete_all_contours_of_mask(mask_id: int, unreviewed_only: bool = False, db: Session = Depends(get_session)):
    if unreviewed_only:
        db.query(Contours).filter(Contours.mask_id == mask_id, ~Contours.reviewed_by.any()).delete()
    else:
        db.query(Contours).filter_by(mask_id=mask_id).delete()
    mask = db.query(Masks).filter_by(id=mask_id).first()
    mask.fully_annotated = False
    db.commit()

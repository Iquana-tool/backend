import os
from logging import getLogger
from pathlib import Path

import numpy as np
from PIL import Image
from iquana_toolbox.schemas.contour_hierarchy import ContourHierarchy
from sqlalchemy.orm import Session

from app.database.contours import Contours
from app.database.images import Images
from app.database.masks import Masks

logger = getLogger(__name__)


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


async def get_contour_hierarchy_of_mask(mask_id: int, db: Session):
    contours_query = db.query(Contours).filter_by(mask_id=mask_id).all()
    size = await get_size_of_mask(mask_id, db)
    return ContourHierarchy.from_query(contours_query,
                                       height=size["height"],
                                       width=size["width"]
                                       )


async def get_size_of_mask(mask_id: int, db: Session):
    _, height, width = (db.query(Masks.id, Images.height, Images.width)
                        .join(Images, Masks.image_id == Images.id)
                        .filter(Masks.id == mask_id).first())
    return {
        "height": height,
        "width": width,
    }

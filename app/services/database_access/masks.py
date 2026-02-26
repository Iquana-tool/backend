import os
from logging import getLogger
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

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

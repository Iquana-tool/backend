import io
import os
from logging import getLogger
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from PIL import Image
from iquana_toolbox.schemas.contours import Contour
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile

from app.database.contours import Contours
from app.database.images import Images
from app.database.masks import Masks
from app.services.database_access.labels import get_hierarchical_label_name
from app.services.database_access.masks import create_new_mask
from config import THUMBNAILS_DIR

logger = getLogger(__name__)


async def get_dataset_as_df(
        dataset_id: int,
        exclude_not_fully_annotated: bool,
        exclude_unreviewed: bool,
        db: Session,
):
    query = (db.query(Contours, Images.file_name)
    .join(Masks, Masks.id == Contours.mask_id).join(Images, Images.id == Masks.image_id).filter(
        Images.dataset_id == dataset_id
    ))
    if exclude_not_fully_annotated:
        query = query.filter(Masks.fully_annotated == True)
    if exclude_unreviewed:
        query = query.filter(Contours.reviewed_by.any())

    data = query.all()
    df_data = {}
    for row in data:
        contour: Contour = Contour.from_db(row[0])
        file_name: str = row[1]
        label_name = get_hierarchical_label_name(contour.label_id)

        df_data.setdefault("file_name", []).append(file_name)
        df_data.setdefault("label", []).append(label_name)
        df_data.setdefault("label_id", []).append(contour.label_id)
        df_data.setdefault("contour_id", []).append(contour.id)
        df_data.setdefault("area", []).append(contour.quantification.area)
        df_data.setdefault("perimeter", []).append(contour.quantification.perimeter)
        df_data.setdefault("circularity", []).append(contour.quantification.circularity)
        df_data.setdefault("diameter_avg", []).append(contour.quantification.max_diameter)
        df_data.setdefault("coords_x", []).append(contour.x)
        df_data.setdefault("coords_y", []).append(contour.y)
        df_data.setdefault("centroid_x", []).append(np.mean(contour.x))
        df_data.setdefault("centroid_y", []).append(np.mean(contour.y))
    return pd.DataFrame(df_data)

from logging import getLogger
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import pandas as pd
import json
import os
import zipfile
import io
from io import StringIO
from app.database import get_session
from sqlalchemy.orm import Session
from app.database.masks import Masks
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.routes.contours import flatten_hierarchical_dict, get_contours_of_mask
from app.services.labels import get_hierarchical_label_name
from app.services.util import get_mask_path_from_image_path

router = APIRouter(prefix="/export", tags=["export"])
logger = getLogger(__name__)


@router.get("/get_mask_csv/{mask_id}")
async def get_mask_csv(mask_id: int, db: Session = Depends(get_session)):
    """ Download quantification data for the given mask_id as a CSV file. """
    image_name = db.query(Images.file_name).join(Masks, Images.id == Masks.image_id).filter(Masks.id == mask_id).first()
    response = await get_contours_of_mask(mask_id, db)
    flat_list = flatten_hierarchical_dict(response["quantification"])
    df = pd.DataFrame(flat_list)
    csv_content = df.to_csv(index=False)
    response = StreamingResponse(StringIO(csv_content), media_type="text/csv")
    response.headers["Content-Disposition"] = f'attachment; filename="{image_name[0]}_quantifications.csv"'
    return response


@router.get("/get_dataset_csv/{dataset_id}&include_manual={include_manual}&include_auto={include_auto}")
async def get_dataset_csv(dataset_id: int,
                           label_ids: list[int] = None,
                           include_manual: bool = True,
                           include_auto: bool = True,
                           db: Session = Depends(get_session)):
    """ Export quantification data for the given dataset_id and labels.

    Args:
        dataset_id (int): The ID of the dataset to export.
        label_ids (list[int], optional): List of label IDs to filter contours. Defaults to None.
        include_manual (bool, optional): Whether to include manual masks. Defaults to True.
        include_auto (bool, optional): Whether to include auto-generated masks. Defaults to True.
        db (Session, optional): The database session. Defaults to Depends(get_session). This is a fastapi dependency.

    Returns:
        dict: A dictionary containing the success status and message if error, or a
        StreamingResponse with the CSV file.
    """
    query = (db.query(Contours, Images.file_name, Masks.finished, Masks.generated)
                .join(Masks, Masks.id == Contours.mask_id).join(Images, Images.id == Masks.image_id).filter(
                    Images.dataset_id == dataset_id
    ))
    if include_manual and include_auto:
        query = query.filter(Masks.finished == True, Masks.generated == True)
    else:
        if include_manual:
            query = query.filter(Masks.finished == True, Masks.generated == False)
        elif include_auto:
            query = query.filter(Masks.finished == False, Masks.generated == True)
        else:
            return {
                "success": False,
                "message": "At least one of include_manual or include_auto must be True."
            }
    if label_ids:
        query = query.filter(Contours.label.in_(label_ids))

    dataset_name = db.query(Datasets).filter_by(id=dataset_id).first().name

    data = query.all()
    df_data = {}
    for row in data:
        contour, file_name, finished, generated = row
        label_name = get_hierarchical_label_name(contour.label)
        diameters = json.loads(contour.diameters) if isinstance(contour.diameters, str) else contour.diameters
        coords = json.loads(contour.coords) if isinstance(contour.coords, str) else contour.coords

        df_data.setdefault("file_name", []).append(file_name)
        df_data.setdefault("label", []).append(label_name)
        df_data.setdefault("label_id", []).append(contour.label)
        df_data.setdefault("contour_id", []).append(contour.id)
        df_data.setdefault("area", []).append(contour.area)
        df_data.setdefault("perimeter", []).append(contour.perimeter)
        df_data.setdefault("circularity", []).append(contour.circularity)
        df_data.setdefault("diameter_avg", []).append(str(np.mean(diameters)) if diameters else None)
        df_data.setdefault("coords_x", []).append(str(coords["x"]) if coords else None)
        df_data.setdefault("coords_y", []).append(str(coords["y"]) if coords else None)
        df_data.setdefault("centroid_x", []).append(str(np.mean(coords["x"])) if coords else None)
        df_data.setdefault("centroid_y", []).append(str(np.mean(coords["y"])) if coords else None)
        df_data.setdefault("finished", []).append(finished)
        df_data.setdefault("generated", []).append(generated)
    df = pd.DataFrame(df_data)
    if df.empty:
        return {
            "success": False,
            "message": "No data found for the given dataset and filters."
        }
    else:
        # Convert to CSV
        csv_content = df.to_csv(index=False)
        response = StreamingResponse(StringIO(csv_content), media_type="text/csv")
        response.headers["Content-Disposition"] = f'attachment; filename="{dataset_name.replace(' ', '_')}_dataset.csv"'
        return response


@router.get("/get_segmentation_mask_file/{mask_id}")
async def get_segmentation_mask_file(
    mask_id: int,
    db: Session = Depends(get_session)
):
    """Download the segmentation mask file for the given mask_id."""
    mask = db.query(Masks).filter_by(id=mask_id).first()
    if not mask:
        return {
            "success": False,
            "message": "Mask not found."
        }
    elif not mask.finished:
        return {
            "success": False,
            "message": "Cannot download mask that is not finished."
        }

    image = db.query(Images).filter_by(id=mask.image_id).first()
    file_path = get_mask_path_from_image_path(image.file_path) if image else None

    if not file_path or not os.path.exists(file_path):
        logger.error(f"Mask is finished but mask file not found at {file_path}.")
        raise HTTPException(status_code=404, detail="Mask file not found.")

    return FileResponse(
        file_path, media_type="image/png", filename=os.path.basename(file_path)
    )


@router.get("/get_segmentation_dataset/{dataset_id}&include_manual={include_manual}&include_auto={include_auto}")
async def get_segmentation_dataset(
    dataset_id: int,
    include_manual: bool = True,
    include_auto: bool = True,
    db: Session = Depends(get_session)
):
    """Download all images and masks as a ZIP file for the given dataset_id."""
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    masks_query = db.query(Images.file_path).join(Masks).filter(Images.dataset_id == dataset_id)

    if include_manual and include_auto:
        masks = masks_query.filter(Masks.finished == True, Masks.generated == True).all()
    else:
        if include_manual:
            masks = masks_query.filter(Masks.finished == True, Masks.generated == False).all()
        elif include_auto:
            masks = masks_query.filter(Masks.finished == False, Masks.generated == True).all()
        else:
            raise HTTPException(status_code=400, detail="At least one of include_manual or include_auto must be True.")

    if not masks:
        raise HTTPException(status_code=404, detail="No finished masks found for this dataset.")

    zip_filename = f"{dataset.name.replace(' ', '_')}.zip"

    # Create a streaming response with a ZIP file
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for mask in masks:
            image = db.query(Images).filter_by(file_path=mask.file_path).first()
            if not image:
                continue

            mask_file_path = get_mask_path_from_image_path(image.file_path)
            if not os.path.exists(mask_file_path):
                logger.error(f"Mask file not found at {mask_file_path}.")
                continue

            zipf.write(mask_file_path, os.path.join("masks", os.path.basename(mask_file_path)))
            zipf.write(image.file_path, os.path.join("images", os.path.basename(image.file_path)))

    # Seek to the start of the buffer
    buffer.seek(0)

    # Create a streaming response
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )

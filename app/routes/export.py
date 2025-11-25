import io
import json
import os
import zipfile
from collections import defaultdict
from io import StringIO
from logging import getLogger

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi import status
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.routes.contours import get_contours_of_mask
from app.schemas.contours import ContourHierarchy
from app.schemas.labels import LabelHierarchy
from app.services.labels import get_hierarchical_label_name
from app.services.util import get_mask_path_from_image_path
from app.schemas.user import User
from app.services.auth import get_current_user

router = APIRouter(prefix="/export", tags=["export"])
logger = getLogger(__name__)


@router.get("/get_mask_csv/{mask_id}")
async def get_mask_csv(
        mask_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """
    Download quantification data for the given mask_id as a CSV file.

    Args:
        mask_id (int): The ID of the mask.
        db (Session): The database session.
        user (User): The current authenticated user.
    """
    image_name = db.query(Images.file_name).join(Masks, Images.id == Masks.image_id).filter(Masks.id == mask_id).first()
    response = await get_contours_of_mask(mask_id, True, db)
    df = pd.DataFrame(response["contours"])
    csv_content = df.to_csv(index=False)
    response = StreamingResponse(StringIO(csv_content), media_type="text/csv")
    response.headers["Content-Disposition"] = f'attachment; filename="{image_name[0]}_quantifications.csv"'
    return response


@router.get(
    "/get_dataset_quantification/{dataset_id}&include_manual={include_manual}&include_auto={include_auto}&as_download={as_download}")
async def get_dataset_quantification(
        dataset_id: int,
        label_ids: list[int] = None,
        include_manual: bool = True,
        include_auto: bool = True,
        as_download: bool = False,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """
    Export quantification data for the given dataset_id and labels.

    Args:
        dataset_id (int): The ID of the dataset to export.
        label_ids (list[int], optional): List of label IDs to filter contours. Defaults to None.
        include_manual (bool, optional): Whether to include manual masks. Defaults to True.
        include_auto (bool, optional): Whether to include auto-generated masks. Defaults to True.
        as_download (bool, optional): Whether to export as CSV. Defaults to False. If False, returns the data as a json.
        db (Session, optional): The database session. Defaults to Depends(get_session). This is a fastapi dependency.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and message if error, or a
        StreamingResponse with the CSV file.
    """
    if dataset_id not in user.available_datasets:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access to this dataset.")
    query = (db.query(Contours, Images.file_name, Masks.finished, Masks.generated)
    .join(Masks, Masks.id == Contours.mask_id).join(Images, Images.id == Masks.image_id).filter(
        Images.dataset_id == dataset_id
    ))
    if include_manual and include_auto:
        query = query.filter(Masks.finished == True or Masks.generated == True)
    else:
        if include_manual:
            query = query.filter(Masks.finished == True and Masks.generated == False)
        elif include_auto:
            query = query.filter(Masks.finished == False and Masks.generated == True)
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
        label_name = get_hierarchical_label_name(contour.label_id)
        diameters = json.loads(contour.diameters) if isinstance(contour.diameters, str) else contour.diameters
        coords = json.loads(contour.coords) if isinstance(contour.coords, str) else contour.coords

        df_data.setdefault("file_name", []).append(file_name)
        df_data.setdefault("label", []).append(label_name)
        df_data.setdefault("label_id", []).append(contour.label_id)
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
        if as_download:
            # Convert to CSV
            csv_content = df.to_csv(index=False)
            response = StreamingResponse(StringIO(csv_content), media_type="text/csv")
            response.headers[
                "Content-Disposition"] = f'attachment; filename="{dataset_name.replace(' ', '_')}_dataset.csv"'
            return response
        else:
            return {
                "success": True,
                "message": "Successfully exported the dataset as json.",
                "data": df.to_json(orient="records"),
            }


@router.get(
    "/get_dataset_object_quantifications/{dataset_id}&include_manual={include_manual}&include_auto={include_auto}")
async def get_dataset_object_quantifications(dataset_id: int,
                                             include_manual: bool = True,
                                             include_auto: bool = False,
                                             db: Session = Depends(get_session),
                                             user: User = Depends(get_current_user),
                                             ):
    if dataset_id not in user.available_datasets:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access to this dataset.")
    images = db.query(Images).filter(Images.dataset_id == dataset_id).all()
    metrics_per_label = defaultdict(lambda: defaultdict(list))
    child_counts_per_label = defaultdict(lambda: defaultdict(list))
    for image in images:
        mask = db.query(Masks).filter(Masks.image_id == image.id).first()  # Only one should exist
        if not include_manual and mask.finished:
            continue
        elif not include_auto and mask.generated:
            continue
        contours = db.query(Contours).filter_by(mask_id=mask.id)
        contour_hierarchy = ContourHierarchy.from_query(contours)
        label_quants = contour_hierarchy.get_all_quantifications()
        for label, quants in label_quants.items(): 
            metrics = quants["metrics"]
            child_counts = quants["child_counts"]
            for k, v in metrics.items():
                metrics_per_label[label][k].extend(v)
            for k, v in child_counts.items():
                child_counts_per_label[label][k].extend(v)
    labels = db.query(Labels).filter_by(dataset_id=dataset_id)
    labels_hierarchy = LabelHierarchy.from_query(labels)
    return {
        "success": True,
        "message": "Successfully exported the object quantifications of this dataset as json.",
        "labels": labels_hierarchy.model_dump(),
        "metrics_per_label_id": metrics_per_label,
        "child_counts_per_label_id": child_counts_per_label,
    }


@router.get("/get_segmentation_mask_file/{mask_id}")
async def get_segmentation_mask_file(
        mask_id: int,
        user: User = Depends(get_current_user),
        db: Session = Depends(get_session)
):
    """Download the prompted_segmentation mask file for the given mask_id."""
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
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """
    Download all images and masks as a ZIP file for the given dataset_id.

    Args:
        dataset_id (int): The ID of the dataset.
        include_manual (bool): Whether to include manual masks.
        include_auto (bool): Whether to include auto-generated masks.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        StreamingResponse: A ZIP file containing all images and masks.
    """
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

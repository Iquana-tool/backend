from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse, FileResponse
import pandas as pd
import json
from io import StringIO

from fsspec.compression import unzip
from starlette.responses import JSONResponse

from app.database import get_session
from sqlalchemy.orm import Session
from app.database.mask_generation import Masks, Contours
from app.database.datasets import Datasets, Labels
from app.database.images import Images
from app.services.labels import get_hierarchical_label_name

router = APIRouter(prefix="/export", tags=["export"])


def query_to_streaming_response(query, filename: str):
    """ Convert a SQLAlchemy query to a StreamingResponse. """
    df = pd.read_sql(query.statement, query.session.bind)
    # Send the dataframe as a CSV file without saving locally
    stream = StringIO()
    df.to_csv(stream, index=False)  # Write to the StringIO stream
    response = FileResponse(stream.getvalue(), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


def build_hierarchical_json(mask_id, filter_labels_ids, db: Session, parent_id = None):
    """ Build a hierarchical JSON structure of contours for a given mask_id.

    Args:
        mask_id (int): The ID of the mask to filter contours.
        filter_labels_ids (list[int]): Optional list of label IDs to filter contours.
        db (Session): The database session.
        parent_id (int): Optional parent contour ID to filter contours.

    Returns:
        list: A list of contours in hierarchical JSON format."""
    query = db.query(Contours).filter_by(mask_id=mask_id, parent_id=parent_id)
    if filter_labels_ids:
        query = query.filter(Contours.label.in_(filter_labels_ids))
    contours = query.all()

    result = []
    for contour in contours:
        label_name = get_hierarchical_label_name(contour.label)
        child_contours = build_hierarchical_json(mask_id, filter_labels_ids, db, contour.id)
        diameters = json.loads(contour.diameters) if isinstance(contour.diameters, str) else contour.diameters
        coords = json.loads(contour.coords) if isinstance(contour.coords, str) else contour.coords
        result.append({
            "id": contour.id,
            "label": contour.label,
            "label_name": label_name,
            "area": contour.area,
            "perimeter": contour.perimeter,
            "circularity": contour.circularity,
            "diameters": diameters,
            "diameter_avg": np.average(diameters) if diameters else None,
            "coords": coords,
            "center_x": np.mean(coords["x"]) if coords and "x" in coords else None,
            "center_y": np.mean(coords["y"]) if coords and "y" in coords else None,
            "children": child_contours
        })
    return result


def flatten_hierarchical_dict(hierarchical_dict, parent_id=None):
    """ Flatten a hierarchical dictionary into a list of dictionaries.

    Args:
        hierarchical_dict (list): The hierarchical dictionary to flatten.
        parent_id (int): The parent ID for the current level.

    Returns:
        list: A flattened list of dictionaries."""
    flat_list = []
    for item in hierarchical_dict:
        flat_item = {
            "id": item["id"],
            "label": item["label"],
            "label_name": item["label_name"],
            "area": item["area"],
            "perimeter": item["perimeter"],
            "circularity": item["circularity"],
            "diameters": item["diameters"],
            "coords": item["coords"],
            "parent_id": parent_id
        }
        flat_list.append(flat_item)
        if item.get("children"):
            flat_list.extend(flatten_hierarchical_dict(item["children"], item["id"]))
    return flat_list


@router.get("/get_quantification/{mask_id}&flattened={flattened}")
async def get_quantification(mask_id: int, flattened: bool = True, db: Session = Depends(get_session)):
    """ Export quantification data for the given mask_id and labels. """
    quantification = build_hierarchical_json(mask_id, [], db)
    if flattened:
        quantification = flatten_hierarchical_dict(quantification)
    return {
        "success": True,
        "message": f"Quantification data for mask {mask_id} exported successfully.",
        "quantification": quantification
    }


@router.get("/download_quantification_csv/{mask_id}")
async def download_quantification_csv(mask_id: int, db: Session = Depends(get_session)):
    """ Download quantification data for the given mask_id as a CSV file. """
    image_name = db.query(Images.file_name).join(Masks, Images.id == Masks.image_id).filter(Masks.id == mask_id).first()
    response = await get_quantification(mask_id, db)
    flat_list = flatten_hierarchical_dict(response["quantification"])
    df = pd.DataFrame(flat_list)
    csv_content = df.to_csv(index=False)
    response = StreamingResponse(StringIO(csv_content), media_type="text/csv")
    response.headers["Content-Disposition"] = f'attachment; filename="{image_name[0]}_quantifications.csv"'
    return response


@router.post("/download_dataset/{dataset_id}")
async def download_dataset(dataset_id: int,
                     label_ids: list[int] = None,
                     annotation_level: Literal["manual_only", "manual+reviewed", "all"] = "manual_only",
                     db: Session = Depends(get_session)):
    """ Export quantification data for the given dataset_id and labels. """
    condition = True
    if annotation_level == "manual_only":
        query = db.query(Contours).join(Masks).join(Contours).filter(
            Masks.finished == True and Masks.generated == False)
    elif annotation_level == "manual+reviewed":
        query = db.query(Contours).join(Masks).join(Contours).filter(Masks.reviewed == True or Masks.finished == True)
    else:
        query = db.query(Contours).join(Masks).join(Contours).filter(Masks.finished == True)

    if label_ids:
        query = query.filter(Contours.label.in_(label_ids))

    dataset_name = db.query(Datasets).filter_by(id=dataset_id).first().name
    return query_to_streaming_response(query, f"{dataset_name}_quantifications.csv")

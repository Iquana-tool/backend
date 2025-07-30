import numpy as np
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse, FileResponse
import pandas as pd
import json
from io import StringIO

from app.database import get_session
from sqlalchemy.orm import Session
from app.database.mask_generation import Masks, Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.routes.contours import flatten_hierarchical_dict, get_contours_of_mask
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


@router.get("/download_quantification_csv/{mask_id}")
async def download_quantification_csv(mask_id: int, db: Session = Depends(get_session)):
    """ Download quantification data for the given mask_id as a CSV file. """
    image_name = db.query(Images.file_name).join(Masks, Images.id == Masks.image_id).filter(Masks.id == mask_id).first()
    response = await get_contours_of_mask(mask_id, db)
    flat_list = flatten_hierarchical_dict(response["quantification"])
    df = pd.DataFrame(flat_list)
    csv_content = df.to_csv(index=False)
    response = StreamingResponse(StringIO(csv_content), media_type="text/csv")
    response.headers["Content-Disposition"] = f'attachment; filename="{image_name[0]}_quantifications.csv"'
    return response


@router.post("/download_dataset/{dataset_id}")
async def download_dataset(dataset_id: int,
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
        df_data.setdefault("diameter_avg", []).append(np.average(diameters) if diameters else None)
        df_data.setdefault("coords_x", []).append(coords["x"] if coords and "x" in coords else None)
        df_data.setdefault("coords_y", []).append(coords["y"] if coords and "y" in coords else None)
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


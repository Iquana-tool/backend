from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, StreamingResponse
import pandas as pd
import os
from io import StringIO
from app.database import get_session
from sqlalchemy.orm import Session
from app.database.mask_generation import Masks, Contours
from app.database.datasets import Datasets, Labels
from app.database.images import Images

router = APIRouter(prefix="/export", tags=["export"])


def query_to_streaming_response(query, filename: str):
    """ Convert a SQLAlchemy query to a StreamingResponse. """
    df = pd.read_sql(query.statement, query.session.bind)
    # Send the dataframe as a CSV file without saving locally
    stream = StringIO()
    df.to_csv(stream, index=False)  # Write to the StringIO stream
    response = StreamingResponse(
        iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


@router.post("/download_quantification/{mask_id}")
def download_quantification(mask_id: int, label_ids: list[int] = None, db: Session = Depends(get_session)):
    """ Export quantification data for the given mask_id and labels. """
    query = db.query(Contours).filter_by(mask_id=mask_id)
    if label_ids:
        query = query.filter(Contours.label.in_(label_ids))
    return query_to_streaming_response(query, f"quantification_{mask_id}.csv")


@router.post("/download_dataset/{dataset_id}")
def download_dataset(dataset_id: int,
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


@router.post("/download_dataset/{dataset_id")
def download_dataset(dataset_id: int, db: Session = Depends(get_session)):
    """ Download the dataset as a zip containing the images and masks. """
    raise NotImplementedError("This is not implemented yet!")


@router.get("/get_quantification/{mask_id}")
def get_quantification(mask_id: int, db: Session = Depends(get_session)):
    """ Get quantification data for the given mask_id. """
    return db.query(Contours).filter_by(mask_id=mask_id).all()

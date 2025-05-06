from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, StreamingResponse
import pandas as pd
import os
from io import StringIO
from app.database import get_session
from sqlalchemy.orm import Session
from app.database.mask_generation import Masks, Contours
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
def export_quantification(mask_id: int, labels: list[int] = None, db: Session = Depends(get_session)):
    """ Export quantification data for the given mask_id and labels. """
    query = db.query(Contours).filter_by(mask_id=mask_id)
    if labels:
        query = query.filter(Contours.label.in_(labels))
    return query_to_streaming_response(query, f"quantification_{mask_id}.csv")


@router.post("/download_multiple_quantifications")
def export_multiple_quantifications(mask_ids: list[int], labels: list[int] = None, db: Session = Depends(get_session)):
    """ Export quantification data for the given mask_ids and labels. """
    query = db.query(Contours).filter(Contours.mask_id.in_(mask_ids))
    if labels:
        query = query.filter(Contours.label.in_(labels))
    return query_to_streaming_response(query, "quantifications.csv")


@router.get("/get_quantification/{mask_id}")
def get_quantification(mask_id: int, db: Session = Depends(get_session)):
    """ Get quantification data for the given mask_id. """
    return db.query(Contours).filter_by(mask_id=mask_id).all()


from fastapi import APIRouter, Depends, File, UploadFile
from starlette.responses import StreamingResponse
import pandas as pd
from app.database import get_session
from sqlalchemy.orm import Session
from app.database.mask_generation import Masks, Contours

router = APIRouter(prefix="/export", tags=["export"])


@router.get("/quantification/{mask_id}")
def export_quantification(mask_id: int, labels: list[int] = None, db: Session = Depends(get_session)):
    """ Export quantification data for the given mask_id and labels. """
    if labels:
        query = db.query(Contours).filter_by(mask_id=mask_id).filter(Contours.label.in_(labels))
    else:
        query = db.query(Contours).filter_by(mask_id=mask_id)
    df = pd.read_sql(query.statement, query.session.bind)
    return df.to_csv(index=False)


@router.get("/quantifications")
def export_multiple_quantifications(mask_ids: list[int], labels: list[int] = None, db: Session = Depends(get_session)):
    """ Export quantification data for the given mask_ids and labels. """
    if labels:
        query = db.query(Contours).filter(Contours.mask_id.in_(mask_ids)).filter(Contours.label.in_(labels))
    else:
        query = db.query(Contours).filter(Contours.mask_id.in_(mask_ids))
    df = pd.read_sql(query.statement, query.session.bind)
    return df.to_csv(index=False)

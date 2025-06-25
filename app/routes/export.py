from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import pandas as pd
import json
from io import StringIO
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


@router.get("/get_quantification/{mask_id}")
def get_quantification(mask_id: int, db: Session = Depends(get_session)):
    """ Get quantification data for the given mask_id. """
    contours = db.query(Contours).filter_by(mask_id=mask_id).all()

    # Get mask and image info to find dataset for label lookup
    mask = db.query(Masks).filter_by(id=mask_id).first()
    if not mask:
        return {"success": False, "error": "Mask not found"}

    image = db.query(Images).filter_by(id=mask.image_id).first()
    if not image:
        return {"success": False, "error": "Image not found"}

    # Get all labels for this dataset to map label IDs to names with hierarchy
    labels = db.query(Labels).filter_by(dataset_id=image.dataset_id).all()
    label_id_to_name = {label.id: label.name for label in labels}
    label_id_to_parent = {label.id: label.parent_id for label in labels}

    # Format quantifications data for frontend
    quantifications = []
    for contour in contours:
        # Parse diameters from JSON string
        diameters = json.loads(contour.diameters) if isinstance(contour.diameters, str) else contour.diameters
        label_name = get_hierarchical_label_name(contour.label)

        quantifications.append({
            "id": contour.id,
            "mask_id": contour.mask_id,
            "label": contour.label,
            "label_name": label_name,
            "area": contour.area,
            "perimeter": contour.perimeter,
            "circularity": contour.circularity,
            "diameters": diameters,
            "coords": json.loads(contour.coords) if isinstance(contour.coords, str) else contour.coords
        })

    return {
        "success": True,
        "quantifications": quantifications
    }

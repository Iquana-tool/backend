from logging import getLogger

from fastapi import Depends
from iquana_toolbox.schemas.labels import Label, LabelHierarchy
from sqlalchemy.orm import Session

from app.database import get_context_session, get_session
from app.database.labels import Labels

logger = getLogger(__name__)


async def get_hierarchical_label_name(label_id, db: Session = Depends(get_session)):
    """ Retrieves the hierarchical name of a label by its ID. This will be in the form of
    Label > SubLabel > SubSubLabel..."""
    label = db.query(Labels).filter_by(id=label_id).first()
    if not label:
        return f"Unknown Label ({label_id})"
    label_name = label.name
    parent_id = label.parent_id

    # If this label has a parent, prepend parent name
    if parent_id:
        parent_name = db.query(Labels).filter_by(id=parent_id).first().name
        return f"{parent_name} › {label_name}"

    return label_name


async def get_label(label_id, db: Session = Depends(get_session)):
    label_db = db.query(Labels).filter_by(id=label_id).first()
    if not label_db:
        raise KeyError("Label not found.")
    return Label.from_db(label_db)


async def get_label_hierarchy(
        dataset_id: int,
        db: Session = Depends(get_session)
) -> LabelHierarchy:
    label_db = db.query(Labels).filter_by(dataset_id=dataset_id)
    if not label_db:
        raise ValueError(f"There are no labels for dataset with id {dataset_id}!")
    return LabelHierarchy.from_query(label_db)



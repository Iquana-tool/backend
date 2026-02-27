from logging import getLogger

from fastapi import Depends
from iquana_toolbox.schemas.labels import Label, LabelHierarchy
from sqlalchemy.orm import Session

from app.database import get_session
from app.database.labels import Labels

logger = getLogger(__name__)


async def get_hierarchical_label_name(
        label_id,
        db: Session
):
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


async def get_label(
        label_id,
        db: Session
):
    label_db = db.query(Labels).filter_by(id=label_id).first()
    if not label_db:
        raise KeyError("Label not found.")
    return Label.from_db(label_db)


async def get_label_hierarchy(
        dataset_id: int,
        db: Session
) -> LabelHierarchy:
    label_db = db.query(Labels).filter_by(dataset_id=dataset_id)
    if not label_db:
        raise ValueError(f"There are no labels for dataset with id {dataset_id}!")
    return LabelHierarchy.from_query(label_db)


async def create_label(
        label_name: str,
        dataset_id: int,
        db: Session,
        parent_label_id: int = None,
        label_value: int = None,
):
    # Check if class already exists
    existing_class = db.query(Labels).filter_by(dataset_id=dataset_id, name=label_name).first()
    if existing_class:
        raise ValueError("Label already exists.")
    if parent_label_id and not db.query(Labels).filter_by(id=parent_label_id).exists():
        raise ValueError("Parent label not found.")
    if not label_value:
        label_value = db.query(Labels).filter_by(dataset_id=dataset_id).count() + 1  # Default value
    # Create a new class
    new_label = Labels(dataset_id=dataset_id,
                       name=label_name,
                       parent_id=parent_label_id,
                       value=label_value)
    db.add(new_label)
    db.commit()
    return new_label


async def update_label(
        label_id: int,
        updates: dict,
        db: Session
):
    existing_class = db.query(Labels).filter_by(id=label_id).first()
    for k, v in updates.items():
        setattr(existing_class, k, v)
    db.commit()


async def replace_label(
        label_id: int,
        new_label: Label,
        db: Session
):
    existing_class = db.query(Labels).filter_by(id=label_id).first()
    parent_id = existing_class.parent_id
    db.delete(existing_class)
    new_label.id = label_id
    new_label.parent = parent_id
    new_label_db = Labels.from_schema(new_label)
    db.add(new_label_db)
    db.commit()


async def delete_label(label_id: int, db: Session = Depends(get_session)):
    existing_label = db.query(Labels).filter_by(id=label_id).first()
    db.delete(existing_label)
    db.commit()

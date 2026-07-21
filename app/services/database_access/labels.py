from logging import getLogger

from iquana_toolbox.schemas.database.labels import Label, LabelHierarchy
from sqlalchemy.orm import Session

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
    if parent_label_id and db.query(Labels.id).filter_by(id=parent_label_id).scalar() is None:
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


async def bulk_create_labels(
        dataset_id: int,
        draft_labels: list,
        db: Session,
):
    """Persist a whole draft label hierarchy for a dataset in one transaction.

    ``draft_labels`` is a list of objects exposing ``name`` and ``children``
    (e.g. :class:`app.schemas.label_space.DraftLabel`). The tree is inserted
    depth-first; parent ids are resolved as each node is flushed. Label values
    continue from the dataset's current maximum.

    Names must be unique across the whole dataset (existing labels included),
    mirroring the constraint enforced by :func:`create_label`.

    Returns:
        int: The number of labels created.
    """
    existing = db.query(Labels).filter_by(dataset_id=dataset_id).all()
    existing_names = {label.name for label in existing}
    next_value = max((label.value for label in existing), default=0) + 1

    seen: set[str] = set()
    created_count = 0

    def insert(node, parent_id):
        nonlocal next_value, created_count
        name = (node.name or "").strip()
        if not name:
            raise ValueError("Label names must not be empty.")
        if name in existing_names or name in seen:
            raise ValueError(
                f"Duplicate label name: '{name}'. Names must be unique within a dataset."
            )
        seen.add(name)
        label = Labels(
            dataset_id=dataset_id,
            name=name,
            parent_id=parent_id,
            value=next_value,
        )
        next_value += 1
        db.add(label)
        db.flush()  # assign label.id so children can reference it
        created_count += 1
        for child in getattr(node, "children", []) or []:
            insert(child, label.id)

    try:
        for root in draft_labels:
            insert(root, None)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return created_count


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


async def delete_label(
        label_id: int,
        db: Session
):
    existing_label = db.query(Labels).filter_by(id=label_id).first()
    db.delete(existing_label)
    db.commit()

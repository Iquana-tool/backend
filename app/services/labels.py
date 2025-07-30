from app.database import get_context_session
from app.database.labels import Labels


def get_hierarchical_label_name(label_id):
    """ Retrieves the hierarchical name of a label by its ID. This will be in the form of
    Label > SubLabel > SubSubLabel..."""
    with get_context_session() as session:
        label = session.query(Labels).filter_by(id=label_id).first()
        if not label:
            return f"Unknown Label ({label_id})"
        label_name = label.name
        parent_id = label.parent_id

        # If this label has a parent, prepend parent name
        if parent_id:
            parent_name = session.query(Labels).filter_by(id=parent_id).first().name
            return f"{parent_name} › {label_name}"

        return label_name


def label_id_to_value(label_id):
    """
    Converts a label ID to its corresponding value.

    Args:
        label_id (int): The ID of the label.

    Returns:
        str: The name of the label or "Unknown Label" if not found.
    """
    with get_context_session() as session:
        label = session.query(Labels).filter_by(id=label_id).first()
        if label:
            return label.value
        else:
            return "Unknown Label"


def label_value_to_label_id(dataset_id, label_value):
    """
    Converts a label value to its corresponding label ID.

    Args:
        dataset_id (int): The ID of the dataset.
        label_value (int): The value of the label.

    Returns:
        int: The ID of the label or None if not found.
    """
    with get_context_session() as session:
        label = session.query(Labels).filter_by(dataset_id=dataset_id, value=label_value).first()
        if label:
            return label.id
        else:
            return None


from app.database import get_context_session
from app.database.datasets import Labels


def get_hierarchical_label_name(label_id):
    with get_context_session() as session:
        if label_id not in session.query(Labels).all():
            return f"Unknown Label ({label_id})"
        label = session.query(Labels).filter_by(id=label_id).first()
        label_name = label.name
        parent_id = label.parent_id

        # If this label has a parent, prepend parent name
        if parent_id:
            parent_name = session.query(Labels).filter_by(parent_id=parent_id).first().name
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

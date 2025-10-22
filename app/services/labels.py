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

import re
from app.database.datasets import Labels
from app.database import get_context_session


def extract_numbers(text):
    # This pattern matches positive integers
    pattern = r'\d+'
    numbers = re.findall(pattern, text)
    # Convert the extracted strings to integers
    return [int(num) for num in numbers]


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

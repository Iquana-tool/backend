import re


def extract_numbers(text):
    # This pattern matches positive integers
    pattern = r'\d+'
    numbers = re.findall(pattern, text)
    # Convert the extracted strings to integers
    return [int(num) for num in numbers]


def get_hierarchical_label_name(label_id):
    if label_id not in label_id_to_name:
        return f"Unknown Label ({label_id})"

    label_name = label_id_to_name[label_id]
    parent_id = label_id_to_parent.get(label_id)

    # If this label has a parent, prepend parent name
    if parent_id and parent_id in label_id_to_name:
        parent_name = label_id_to_name[parent_id]
        return f"{parent_name} › {label_name}"

    return label_name

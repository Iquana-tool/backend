from collections import deque

from pydantic import BaseModel, Field
from sqlalchemy.orm import Query

from app.database.labels import Labels


class Label(BaseModel):
    id: int = Field(..., description="The label id. Ids are unique to the database.")
    dataset_id: int = Field(..., description="The dataset id.")
    name: str = Field(..., description="Human-readable name of the label.")
    value: int = Field(..., description="The label value, eg. what value does it map to. Usually different from the id."
                                        "Values are unique to the dataset.")
    parent_id: int | None = Field(None, description="The parent label's id.")
    children: list["Label"] = Field([], description="The child label object.")
    
    @classmethod
    def from_db(cls, label: Labels):
        return Label(
            id=label.id,
            dataset_id=label.dataset_id,
            name=label.name,
            value=label.value if label.value else None,
            parent_id=label.parent_id,
            children=[]
        )
    
    def add_child(self, child):
        self.children.append(child)
    
    
class LabelHierarchy(BaseModel):
    root_level_labels: list[Label] = Field(..., description="The label hierarchy. A list of root-level Label objects. "
                                                            "Each Label object may have children forming a tree structure.")
    id_to_label_object: dict[int, Label] = Field(..., description="A dictionary mapping label ids to Label objects.")
    value_to_label_object: dict[int, Label] = Field(...,
                                                    description="A dictionary mapping label values to Label objects. "
                                                                "The difference between id and value is that ids are "
                                                                "unique to the database, while values are unique to the "
                                                                "dataset only (ie datasets can have labels with the same "
                                                                "value but not with the same id.")

    @classmethod
    def from_query(cls, query: "Query[type[Labels]]"):
        # Fetch all root labels (parent_id is None)
        root_labels = query.filter_by(parent_id=None).all()
        root_ids = [label.id for label in root_labels]

        # Fetch all labels in the hierarchy
        queue = deque(root_labels)

        # Build a map from id/value to Label
        id_to_label = {}
        value_to_label = {}

        # Build the hierarchy
        while queue:
            label = queue.popleft()
            label_obj = Label.from_db(label)
            id_to_label[label_obj.id] = label_obj
            value_to_label[label_obj.value] = label_obj
            if label.parent_id is not None:
                parent = id_to_label[label.parent_id]
                parent.add_child(label_obj)
            queue.extend(query.filter_by(parent_id=label.id).all())

        # Return the root-level labels
        return cls(
            root_level_labels=[id_to_label[root_id] for root_id in root_ids],
            id_to_label_object=id_to_label,
            value_to_label_object=value_to_label,
           )

    def __len__(self):
        return len(self.id_to_label_object.keys())

    def get_parent_by_id_of_child(self, child_id):
        child = self.id_to_label_object[child_id]
        if child.parent_id is not None:
            return self.id_to_label_object[child.parent_id]
        else:
            return None

    def get_parent_by_value_of_child(self, child_value):
        child = self.value_to_label_object[child_value]
        if child.parent_id is not None:
            return self.id_to_label_object[child.parent_id]
        else:
            return None

    def build_flat_hierarchy(self, breadth_first: bool = True) -> list[Label]:
        flat_list = []
        if breadth_first:
            # Breadth-first traversal
            queue = deque(self.root_level_labels)
            while queue:
                node = queue.popleft()
                flat_list.append(node)
                queue.extend(node.children)
        else:
            # Depth-first traversal
            stack = list(reversed(self.root_level_labels))
            while stack:
                node = stack.pop()
                flat_list.append(node)
                stack.extend(reversed(node.children))
        return flat_list

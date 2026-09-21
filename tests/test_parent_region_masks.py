"""parent_region_masks: which existing contours a hierarchy-aware suggestion model searches in."""
from types import SimpleNamespace

from app.services.annotation_session.operations import parent_region_masks


class _Contour:
    def __init__(self, name):
        self.name = name

    def to_binary_mask_model(self, height, width):
        return (self.name, height, width)


def _label_hierarchy(parents: dict[int, int | None]):
    labels = {i: SimpleNamespace(id=i) for i in parents}
    return SimpleNamespace(
        id_to_label_object=labels,
        get_parent_by_id_of_child=lambda i: labels.get(parents[i]) if parents[i] is not None else None,
    )


# fragment (1) -> polyp (2)
LABELS = _label_hierarchy({1: None, 2: 1})
HIERARCHY = SimpleNamespace(label_id_to_contours={
    1: [_Contour("fragment_a"), _Contour("fragment_b")],
    2: [_Contour("polyp")],
})


def test_child_concept_gets_every_parent_label_contour():
    assert parent_region_masks(HIERARCHY, LABELS, 2, 10, 20) == [
        ("fragment_a", 10, 20), ("fragment_b", 10, 20),
    ]


def test_root_unknown_or_missing_concept_gets_nothing():
    assert parent_region_masks(HIERARCHY, LABELS, 1, 10, 20) == []
    assert parent_region_masks(HIERARCHY, LABELS, 99, 10, 20) == []
    assert parent_region_masks(HIERARCHY, LABELS, None, 10, 20) == []


def test_no_parent_objects_annotated_yet():
    empty = SimpleNamespace(label_id_to_contours={})
    assert parent_region_masks(empty, LABELS, 2, 10, 20) == []

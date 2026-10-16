"""How a model's fixed label space maps onto one dataset's labels.

A model with a fixed label space (``ModelInfo.classes``) names its classes in its own
terms -- "Choroidal vessels" -- and every prediction says which class it is
(``Contour.model_class``). Each dataset decides, once per model and class, what that
class means here: one of its labels, or nothing (discard those predictions). A class
without a row falls back to the dataset label of the same name, if there is one.

Deleting a label deletes the rows mapped to it, which leaves those classes unmapped
again rather than pointing at a label that is gone.
"""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, UniqueConstraint

from app.database import database
# The attribute a class may also set lives in the scene-graph tables; importing them here
# registers them on the metadata wherever this table is created.
from app.database import scene_graph as _scene_graph  # noqa: F401


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


#: ``action`` values. ``map`` needs ``label_id``; ``discard`` has none.
MAP = "map"
DISCARD = "discard"


class ModelLabelMappings(database):
    """One model class's meaning in one dataset."""

    __tablename__ = "model_label_mappings"
    __table_args__ = (
        UniqueConstraint("dataset_id", "model_registry_key", "model_class",
                         name="uq_model_label_mappings_class"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    model_registry_key = Column(String, nullable=False)
    model_class = Column(String, nullable=False)
    action = Column(String(16), nullable=False)
    label_id = Column(Integer, ForeignKey("labels.id", ondelete="CASCADE"), nullable=True)
    #: Optionally, an attribute value the class also stands for: a species detector's
    #: class "Acropora" maps to the label "Coral fragment" *and* ``species = Acropora``.
    #: Deleting the attribute keeps the label mapping.
    attribute_id = Column(Integer, ForeignKey("object_attribute_definitions.id", ondelete="SET NULL"),
                          nullable=True)
    attribute_value = Column(String(512), nullable=True)
    updated_by = Column(
        String, ForeignKey("users.username", ondelete="SET NULL", onupdate="CASCADE"), nullable=True
    )
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)

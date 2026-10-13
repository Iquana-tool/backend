"""The scene graph over a dataset's objects: given attributes and named relationships.

Every contour is an object, grounded by its outline. Its **label** answers a
recognition question ("is this a coral fragment?"). Its **attributes** answer
classification questions about an object that is already there ("which species?",
"bleached?"). The rule of thumb: if changing the value would change the outline, it
is a label; otherwise it is an attribute. That is why is-a hierarchies do not live in
the label tree. A bird is labelled ``Bird`` and carries ``species = Weißkopfseeadler``,
and whatever taxonomy that value sits in stays outside the tool (an option can carry
an external ``code`` for joining one later).

**Relationships** connect two objects. Part-of is not stored here: it is the
contour's ``parent_id``, constrained by the label tree. Everything else is a named,
directed edge (``A points_at B``) whose names are reused across the dataset.

Attribute definitions mirror the image-metadata keys on purpose
(``dataset_metadata_keys``): the same type system, the same text + ``value_num``
storage, and the same "options lock the vocabulary" behaviour.

Values and edges cascade with their objects. Paths that change an object without
deleting it (outline edits, refinements) update the contour row in place, so they
keep both.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, relationship

from app.database import database
from app.database.image_metadata import MAX_KEY_LENGTH, MAX_VALUE_LENGTH


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


#: Where a value or an edge came from.
SOURCE_USER = "user"
SOURCE_MODEL = "model"
SOURCE_IMPORT = "import"
SOURCES = (SOURCE_USER, SOURCE_MODEL, SOURCE_IMPORT)

#: Whether a value or an edge stands. Model output starts as ``suggested`` and becomes
#: ``confirmed`` once a person accepts it; a person's own input is confirmed at once.
STATUS_CONFIRMED = "confirmed"
STATUS_SUGGESTED = "suggested"
STATUSES = (STATUS_CONFIRMED, STATUS_SUGGESTED)


object_attribute_label_scope = Table(
    "object_attribute_label_scope",
    database.metadata,
    Column("attribute_id", Integer,
           ForeignKey("object_attribute_definitions.id", ondelete="CASCADE"), primary_key=True),
    Column("label_id", Integer,
           ForeignKey("labels.id", ondelete="CASCADE"), primary_key=True, index=True),
)


class ObjectAttributeDefinitions(database):
    """What one attribute means, within one dataset."""
    __tablename__ = "object_attribute_definitions"

    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"),
                                     nullable=False, index=True)
    name: Mapped[str] = Column(String(MAX_KEY_LENGTH), nullable=False)
    #: One of ``MetadataValueType``, as for image-metadata keys.
    value_type: Mapped[str] = Column(String(16), nullable=False, default="categorical")
    #: Display unit for a numeric attribute. Never used in arithmetic.
    unit: Mapped[str] = Column(String(16), nullable=True)
    #: The allowed vocabulary of a categorical attribute, as
    #: ``[{"value": str, "code": str | None, "description": str | None}]``. Empty means
    #: open: any value is accepted. ``code`` is an external identifier (an AphiaID, an
    #: ontology IRI) that is stored and exported, never interpreted.
    options: Mapped[list] = Column(JSON, nullable=False, default=list)
    description: Mapped[str] = Column(String(256), nullable=True)
    created_by: Mapped[str] = Column(String, ForeignKey("users.username", ondelete="SET NULL",
                                                        onupdate="CASCADE"), nullable=True)
    created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = Column(DateTime, nullable=False, default=_utcnow,
                                          onupdate=_utcnow)

    #: The labels this attribute applies to. Empty means every label.
    labels = relationship("Labels", secondary=object_attribute_label_scope, lazy="selectin")

    __table_args__ = (
        UniqueConstraint("dataset_id", "name", name="uq_object_attribute_definitions_name"),
    )

    @property
    def option_values(self) -> list[str]:
        return [option["value"] for option in (self.options or [])]

    def applies_to(self, label_id: int | None) -> bool:
        """Whether an object with ``label_id`` is in this attribute's scope."""
        if not self.labels:
            return True
        return label_id is not None and any(label.id == label_id for label in self.labels)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "value_type": self.value_type,
            "unit": self.unit,
            "options": list(self.options or []),
            "description": self.description,
            "label_ids": sorted(label.id for label in self.labels),
        }


class ObjectAttributes(database):
    """One attribute's value on one object.

    No row means *not annotated*. A row whose ``value`` is NULL means someone looked
    and *cannot tell* (a species that cannot be identified from the image). Both are
    different from ``false`` and ``0``, which are values.
    """
    __tablename__ = "object_attributes"

    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    contour_id: Mapped[int] = Column(Integer, ForeignKey("contours.id", ondelete="CASCADE"),
                                     nullable=False, index=True)
    attribute_id: Mapped[int] = Column(
        Integer, ForeignKey("object_attribute_definitions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    value: Mapped[str] = Column(String(MAX_VALUE_LENGTH), nullable=True)
    #: Sortable form of ``value`` for the ordered types (see ``metadata_types``).
    value_num: Mapped[float] = Column(Float, nullable=True, index=True)
    source: Mapped[str] = Column(String(16), nullable=False, default=SOURCE_USER)
    status: Mapped[str] = Column(String(16), nullable=False, default=STATUS_CONFIRMED)
    model_key: Mapped[str] = Column(String(255), nullable=True)
    confidence: Mapped[float] = Column(Float, nullable=True)
    #: Who set it, or who accepted a model's value. Set server-side.
    author_username: Mapped[str] = Column(String, ForeignKey("users.username", ondelete="SET NULL",
                                                             onupdate="CASCADE"), nullable=True)
    created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = Column(DateTime, nullable=False, default=_utcnow,
                                          onupdate=_utcnow)

    definition = relationship("ObjectAttributeDefinitions", lazy="joined")

    __table_args__ = (
        UniqueConstraint("contour_id", "attribute_id", name="uq_object_attributes_contour_attribute"),
    )


class RelationshipTypes(database):
    """A relationship name used in one dataset ("points_at", "attached_to")."""
    __tablename__ = "relationship_types"

    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"),
                                     nullable=False, index=True)
    name: Mapped[str] = Column(String(MAX_KEY_LENGTH), nullable=False)
    #: A symmetric relationship ("touches") is the same read from either end. It is
    #: stored once, with the lower contour id as the subject.
    symmetric: Mapped[bool] = Column(Boolean, nullable=False, default=False)
    description: Mapped[str] = Column(String(256), nullable=True)
    created_by: Mapped[str] = Column(String, ForeignKey("users.username", ondelete="SET NULL",
                                                        onupdate="CASCADE"), nullable=True)
    created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        UniqueConstraint("dataset_id", "name", name="uq_relationship_types_name"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "symmetric": bool(self.symmetric),
            "description": self.description,
        }


class ObjectRelationships(database):
    """A directed, named edge between two objects: ``subject <type> object``."""
    __tablename__ = "object_relationships"

    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    #: Denormalised from the contours, so that dataset-wide queries need no join
    #: through masks and images.
    dataset_id: Mapped[int] = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"),
                                     nullable=False, index=True)
    type_id: Mapped[int] = Column(Integer, ForeignKey("relationship_types.id", ondelete="CASCADE"),
                                  nullable=False, index=True)
    subject_id: Mapped[int] = Column(Integer, ForeignKey("contours.id", ondelete="CASCADE"),
                                     nullable=False, index=True)
    object_id: Mapped[int] = Column(Integer, ForeignKey("contours.id", ondelete="CASCADE"),
                                    nullable=False, index=True)
    source: Mapped[str] = Column(String(16), nullable=False, default=SOURCE_USER)
    status: Mapped[str] = Column(String(16), nullable=False, default=STATUS_CONFIRMED)
    model_key: Mapped[str] = Column(String(255), nullable=True)
    confidence: Mapped[float] = Column(Float, nullable=True)
    author_username: Mapped[str] = Column(String, ForeignKey("users.username", ondelete="SET NULL",
                                                             onupdate="CASCADE"), nullable=True)
    created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=_utcnow)

    type = relationship("RelationshipTypes", lazy="joined")

    __table_args__ = (
        UniqueConstraint("type_id", "subject_id", "object_id", name="uq_object_relationships_edge"),
        CheckConstraint("subject_id <> object_id", name="ck_object_relationships_no_self_loop"),
    )

"""Per-dataset declarations of the image-metadata keys in use.

``image_metadata`` stores the pairs; this table says what each key *is* — its
type, its unit, and (for a categorical key) the vocabulary it is allowed to take.
Three things need that:

* **Filters.** A depth wants a range, a collection date wants a date picker, a
  site wants chips. Without a declared type every key gets the chip list.
* **Grouping.** Only some types are meaningful to split a quantification by (see
  ``GROUPABLE_TYPES``); a free-text note is not a subgroup.
* **Consistency.** A locked option list makes ``Reef A`` next to ``reef_a``
  impossible within a key, and turns fixing such a split into a rename on one row
  here rather than an UPDATE across every image.

A descriptor is created automatically the first time a key is written, typed
``categorical`` (see ``DEFAULT_TYPE``) so that behaviour before types existed is
preserved exactly. The rows are therefore a *description* of what the dataset
uses, not a schema anyone has to fill in first.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped

from app.database import database
from app.database.image_metadata import MAX_KEY_LENGTH


class DatasetMetadataKeys(database):
    """What one metadata key means, within one dataset."""
    __tablename__ = "dataset_metadata_keys"

    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = Column(
        Integer, ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    key: Mapped[str] = Column(String(MAX_KEY_LENGTH), nullable=False)

    #: One of ``MetadataValueType``. Stored as its string value so the enum can
    #: gain members without a schema change.
    value_type: Mapped[str] = Column(String(16), nullable=False, default="categorical")

    #: Display unit for a numeric key ("m", "°C"). Never used in arithmetic — the
    #: values are not converted between units, so this is a label, and mixing
    #: units under one key is a curation mistake the tool cannot detect.
    unit: Mapped[str] = Column(String(16), nullable=True)

    #: Allowed vocabulary for a categorical key. Empty (the default) means the key
    #: is open and accepts whatever is written; filling it in locks the key down
    #: and makes new values a 422 rather than a silent new subgroup.
    options: Mapped[list] = Column(JSON, nullable=False, default=list)

    description: Mapped[str] = Column(String(256), nullable=True)

    created_by: Mapped[str] = Column(String, nullable=True)
    created_at: Mapped[datetime] = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = Column(
        DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint("dataset_id", "key", name="uq_dataset_metadata_keys_dataset_key"),
    )

    def to_dict(self) -> dict:
        """Serializable form, as the API returns it."""
        return {
            "key": self.key,
            "value_type": self.value_type,
            "unit": self.unit,
            "options": list(self.options or []),
            "description": self.description,
        }

    def __repr__(self) -> str:
        return (f"<DatasetMetadataKey(dataset_id={self.dataset_id}, key='{self.key}', "
                f"type='{self.value_type}')>")

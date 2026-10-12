"""Free-form per-image metadata, one row per (image, key).

A research dataset is almost never one homogeneous pile of images: they come from
a site, a dive, a transect, a treatment arm, a date. That grouping is what the
measurements are eventually compared *across*, so it has to live next to the
images rather than in the filenames or in someone's spreadsheet.

Shape: a tall (key, value) table rather than a JSON blob on ``images``, for the
same reason ``contour_metrics`` is tall — the interesting queries are "which keys
does this dataset use", "which values does ``site`` take", "give me every image
with ``site=reef_a``". Those are indexable here and are string-munging in a JSON
column, especially on SQLite.

Values are stored as text and are never coerced. A dataset that writes
``depth=12`` and ``depth=12.0`` has two subgroups, which is the honest answer:
the tool does not know that those were meant to be the same, and silently
merging them would quietly change what a comparison is over. The frontend
offers the dataset's existing keys and values as suggestions, which is where
consistency is cheap to enforce.

``UniqueConstraint(image_id, key)`` makes a row *the* current value of that key
for that image — setting a key again overwrites it. Rows go with the image
(ON DELETE CASCADE).

**Owners.** A row belongs to exactly one image *or* one stack. Stack rows hold
what is true of the whole stack (an OCT volume's eye and visit date) and every
frame of the stack inherits them; a frame's own row for the same key wins. Reads
go through the effective view in ``app.services.database_access.image_metadata``,
so a frame reports its inherited keys like any image reports its own.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped

from app.database import database

#: Maximum length of a metadata key. Keys are column-like names ("site",
#: "collection_date"), not prose; the cap keeps them usable as export headers.
MAX_KEY_LENGTH = 64

#: Maximum length of a value. Generous enough for a note, short enough that the
#: field cannot be used as a document store.
MAX_VALUE_LENGTH = 512


class ImageMetadata(database):
    """One metadata key/value pair on one image."""
    __tablename__ = "image_metadata"

    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = Column(
        Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=True, index=True
    )
    #: Set instead of ``image_id`` for metadata of a whole stack, which its frames
    #: inherit.
    stack_id: Mapped[int] = Column(
        Integer, ForeignKey("stacks.id", ondelete="CASCADE"), nullable=True, index=True
    )

    #: Normalised by :func:`app.services.database_access.image_metadata.normalize_key`
    #: (trimmed, whitespace collapsed) before it ever reaches here.
    key: Mapped[str] = Column(String(MAX_KEY_LENGTH), nullable=False)
    #: Trimmed, but otherwise stored exactly as typed. Never null; clearing a key
    #: deletes the row rather than blanking it, so "absent" has one representation.
    value: Mapped[str] = Column(String(MAX_VALUE_LENGTH), nullable=False)

    #: Sortable form of ``value`` for the ordered types — the number itself, a
    #: date's epoch seconds, a boolean's 0/1 (see ``app.services.metadata_types``).
    #: Null for text and categorical keys, which have no order to filter on. The
    #: human-readable spelling always stays in ``value``, so nothing is lost to the
    #: conversion and an export reads the way it was typed.
    value_num: Mapped[float] = Column(Float, nullable=True, index=True)

    #: Account that last wrote this pair, for the same reason contours carry
    #: ``author_username``: subgroup assignment is a curation decision.
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
        UniqueConstraint("image_id", "key", name="uq_image_metadata_image_key"),
        UniqueConstraint("stack_id", "key", name="uq_image_metadata_stack_key"),
        CheckConstraint("(image_id IS NULL) <> (stack_id IS NULL)",
                        name="ck_image_metadata_one_owner"),
        # The facet query ("which keys/values does this dataset use") groups by
        # key and value across every image of a dataset.
        Index("ix_image_metadata_key_value", "key", "value"),
    )

    def __repr__(self) -> str:
        return (f"<ImageMetadata(image_id={self.image_id}, stack_id={self.stack_id}, "
                f"key='{self.key}', value='{self.value}')>")

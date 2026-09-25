from sqlalchemy import Boolean, Column, ForeignKey, Index, Integer, String
from sqlalchemy.orm import relationship

from . import database


class Datasets(database):
    """ Represents a dataset in the database."""
    __tablename__ = 'datasets'
    __table_args__ = (
        # A unique index rather than `unique=True`: databases from before names had
        # to be unique got this index (after deduplication) instead of a constraint,
        # and declaring it the same way keeps every database on one shape.
        Index("uq_datasets_name", "name", unique=True),
    )
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    description = Column(String(255), nullable=True)
    dataset_type = Column(String(20), nullable=False)  # Type of dataset, e.g., "image", "scan", "DICOM"
    folder_path = Column(String(255), nullable=False)  # Path to the dataset folder on disk
    # Immutable provenance: who created the dataset. Control over it lives on the
    # membership row with role "owner", so ownership can be transferred.
    created_by = Column(String, ForeignKey("users.username", ondelete="CASCADE"), nullable=False)
    # When on, a contour cannot be approved by the person who authored it. Off by
    # default so a single owner working alone can still finish their own dataset;
    # turn it on for multi-annotator work where "finished" has to mean "checked by
    # someone else".
    require_independent_review = Column(Boolean, nullable=False, default=False)
    # AI tools switched off for this dataset, comma-separated (see app.services.ai_tools).
    # Empty or NULL means every tool is available.
    disabled_ai_tools = Column(String(255), nullable=True)

    owner = relationship("Users", back_populates="owned_datasets", foreign_keys=[created_by])
    memberships = relationship("DatasetMembers",
                               back_populates="dataset",
                               cascade="all, delete-orphan")
    invites = relationship("DatasetInvites",
                           back_populates="dataset",
                           cascade="all, delete-orphan")
    # Read-only list of collaborators, kept for call sites that only need the users.
    shared_with = relationship("Users",
                               secondary="dataset_members",
                               primaryjoin="Datasets.id == DatasetMembers.dataset_id",
                               secondaryjoin="Users.username == DatasetMembers.username",
                               viewonly=True)

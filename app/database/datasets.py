from sqlalchemy import Column, Integer, String, ForeignKey, Table
from sqlalchemy.orm import relationship

from . import database


# Association table for sharing datasets with users
dataset_user_association = Table(
    "dataset_user_association",
    database.metadata,
    Column("dataset_id", Integer, ForeignKey("datasets.id"), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
)

class Datasets(database):
    """ Represents a dataset in the database."""
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    description = Column(String(255), nullable=True)
    dataset_type = Column(String(20), nullable=False)  # Type of dataset, e.g., "image", "scan", "DICOM"
    folder_path = Column(String(255), nullable=False)  # Path to the dataset folder on disk
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)

    owner = relationship("Users", back_populates="owned_datasets")
    shared_with = relationship("Users", secondary=dataset_user_association, back_populates="accessible_datasets")

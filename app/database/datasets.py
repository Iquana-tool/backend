from sqlalchemy import Column, Integer, String

from . import database


class Datasets(database):
    """ Represents a dataset in the database."""
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    description = Column(String(255), nullable=True)
    dataset_type = Column(String(20), nullable=False)  # Type of dataset, e.g., "image", "scan", "DICOM"
    folder_path = Column(String(255), nullable=False)  # Path to the dataset folder on disk

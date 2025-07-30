from sqlalchemy import Column, Integer, ForeignKey, String, JSON

from app.database import database


class Scans(database):
    """ Represents a scan in the database."""
    __tablename__ = 'scans'
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False)  # Foreign key to the datasets table
    name = Column(String, nullable=False)  # Name of the scan
    folder_path = Column(String, nullable=False)  # Path to the folder containing the scan images
    type = Column(String)  # Type of scan (e.g., 'CT', 'MRI')
    description = Column(String)  # Description of the scan
    number_of_slices = Column(Integer, nullable=False)  # Number of slices in the scan
    meta_data = Column(JSON)  # Save any additional metadata about the scan

from sqlalchemy import Column, Integer, String, ForeignKey, Float, JSON

from . import database


class Scans(database):
    __tablename__ = 'scans'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)  # Name of the scan
    type = Column(String, nullable=False)  # Type of scan (e.g., 'CT', 'MRI')
    description = Column(String)  # Description of the scan
    number_of_slices = Column(Integer, nullable=False)  # Number of slices in the scan
    metadata = Column(JSON, nullable=False) # Save any additional metadata about the scan


class Slices(database):
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_id = Column(Integer, ForeignKey('scans.id', ondelete='CASCADE'), nullable=False)
    slice_number = Column(Integer, nullable=False)  # Slice number in the scan
    filename = Column(String, nullable=False)  # Path to the slice file

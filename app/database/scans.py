from sqlalchemy import Column, Integer, String, ForeignKey, Float, JSON

from . import database


class Scans(database):
    __tablename__ = 'scans'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)  # Name of the scan
    description = Column(String)  # Description of the scan
    created_at = Column(String, nullable=False)  # Timestamp when the scan was created
    updated_at = Column(String, nullable=False)  # Timestamp when the scan was last updated
    status = Column(String, nullable=False)  # Status of the scan (e.g., 'pending', 'completed')

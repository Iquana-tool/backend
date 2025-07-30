from sqlalchemy import Column, Integer, ForeignKey, String

from app.database import database


class Labels(database):
    """ Represents a label in the database."""
    __tablename__ = 'labels'
    id = Column(Integer, primary_key=True, autoincrement=True)  # Unique ID for the label
    dataset_id = Column(Integer, ForeignKey('datasets.id', ondelete='CASCADE'))
    parent_id = Column(Integer, ForeignKey('labels.id', ondelete='CASCADE'))
    name = Column(String, nullable=False)  # Name of the label, eg "Background", "Class 1", etc.
    value = Column(Integer, nullable=False)  # Value of the label, eg 0 for background, 1 for class 1, etc.

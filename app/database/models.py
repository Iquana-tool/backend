from sqlalchemy import Column, String, Text, Integer, ForeignKey
from . import database
from .datasets import Datasets


class Models(database):
    """ Represents a machine learning model in the database."""
    __tablename__ = 'models'
    id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True)
    base_model_identifier = Column(String, nullable=False)  # e.g., 'SAM2', 'SAM2.1', "Unet"
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    model_type = Column(String, nullable=False)  # one of: 'prompted', 'automatic', 'prompted3D', 'automatic3D'
    version = Column(String, nullable=False)
    created_at = Column(String, nullable=True)  # Timestamp when the model was created
    updated_at = Column(String, nullable=True)  # Timestamp when the model was last updated
    weights = Column(String, nullable=True)  # Path to the model weights file, Null if not applicable
    config = Column(String, nullable=True)  # Path to the model configuration file, Null if not applicable
    dataset_id = Column(String, nullable=True)  # Optional, if the model is associated with a specific dataset

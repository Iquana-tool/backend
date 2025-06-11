from sqlalchemy import Column, String, Text, Integer

from . import database


class Models(database):
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


class Training(database):
    __tablename__ = 'training'
    id = Column(String, primary_key=True, unique=True, nullable=False)
    model_id = Column(String, nullable=False)  # Foreign key to Models table
    dataset_id = Column(String, nullable=False)  # Foreign key to Datasets table
    status = Column(String, nullable=False)  # e.g., 'pending', 'in_progress', 'completed', 'failed'
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)
    logs = Column(Text, nullable=True)  # Optional logs for training process

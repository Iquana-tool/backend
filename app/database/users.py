from sqlalchemy import Column, Integer, ForeignKey, String, JSON, Boolean

from app.database import database
from app.database.datasets import dataset_user_association
from sqlalchemy.orm import relationship

class Users(database):
    """ Represents our users. """
    __tablename__ = "users"
    username = Column(String, nullable=False, unique=True, primary_key=True)  # Ensure usernames are unique
    hashed_password = Column(String, nullable=False)  # Store hashed passwords only
    is_admin = Column(Boolean, nullable=False, default=False)

    owned_datasets = relationship("Datasets",
                                  back_populates="owner")
    accessible_datasets = relationship("Datasets",
                                       secondary=dataset_user_association,
                                       back_populates="shared_with")

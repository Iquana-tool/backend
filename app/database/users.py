from sqlalchemy import Column, Integer, ForeignKey, String, JSON

from app.database import database


from sqlalchemy.orm import relationship

class Users(database):
    """ Represents our users. """
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)  # Ensure usernames are unique
    enc_password = Column(String, nullable=False)  # Store hashed passwords only

    owned_datasets = relationship("Datasets", back_populates="owner")
    shared_datasets = relationship("Datasets", secondary="dataset_user_association", back_populates="shared_with")

from sqlalchemy import Column, Integer, ForeignKey, String, JSON

from app.database import database


class Users(database):
    """ Represents our users. """
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)  # Ensure usernames are unique
    enc_password = Column(String, nullable=False)  # Store hashed passwords only

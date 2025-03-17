from sqlalchemy import Column, String, Text

from . import database


class Models(database):
    __tablename__ = 'models'
    id = Column(String, primary_key=True, unique=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String, nullable=False)
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)

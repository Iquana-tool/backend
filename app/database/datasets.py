from sqlalchemy import Column, Integer, String, ForeignKey

from . import database


class DataSets(database):
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    description = Column(String(255), nullable=True)


class Labels(database):
    __tablename__ = 'labels'
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id', ondelete='CASCADE'))
    parent_id = Column(Integer, ForeignKey('labels.id', ondelete='CASCADE'))
    name = Column(String, nullable=False)

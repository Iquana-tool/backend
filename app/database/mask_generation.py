from sqlalchemy import Column, Integer, String, ForeignKey, Float, JSON

from . import database


class Masks(database):
    __tablename__ = 'masks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'),
                      nullable=False)


class Labels(database):
    __tablename__ = 'labels'
    id = Column(Integer, primary_key=True, autoincrement=True)
    parent_id = Column(Integer, ForeignKey('labels.id', ondelete='CASCADE'))
    name = Column(String, nullable=False)


class Contours(database):
    __tablename__ = 'contours'
    id = Column(Integer, primary_key=True, autoincrement=True)
    mask_id = Column(Integer, ForeignKey('masks.id', ondelete='CASCADE'),
                     nullable=False)
    label = Column(Integer, ForeignKey('labels.id', ondelete='CASCADE'), nullable=False)
    area = Column(Float, nullable=False)
    perimeter = Column(Float, nullable=False)
    circularity = Column(Float, nullable=False)
    diameters = Column(JSON, nullable=False)
    coords = Column(JSON, nullable=False)

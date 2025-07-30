from sqlalchemy import Column, Integer, ForeignKey, Float, JSON
from app.database import database


class Contours(database):
    __tablename__ = 'contours'
    id = Column(Integer, primary_key=True, autoincrement=True)
    mask_id = Column(Integer, ForeignKey('masks.id', ondelete='CASCADE'),
                     nullable=False)
    parent_id = Column(Integer, ForeignKey('contours.id', ondelete='CASCADE'))
    label = Column(Integer, ForeignKey('labels.id', ondelete='CASCADE'), nullable=False)
    area = Column(Float, nullable=False)
    perimeter = Column(Float, nullable=False)
    circularity = Column(Float, nullable=False)
    diameters = Column(JSON, nullable=False)
    coords = Column(JSON, nullable=False)

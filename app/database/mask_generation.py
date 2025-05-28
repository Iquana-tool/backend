from sqlalchemy import Column, Integer, String, ForeignKey, Float, JSON, Boolean

from . import database


class Masks(database):
    __tablename__ = 'masks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'),
                      nullable=False)
    finished = Column(Boolean, default=False, nullable=False)  # True if the mask is finished
    generated = Column(Boolean, default=False, nullable=False)  # True if the mask was generated
    reviewed = Column(Boolean, default=False, nullable=False)  # True if the generated mask was reviewed


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

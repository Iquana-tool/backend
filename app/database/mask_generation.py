from sqlalchemy import Column, Integer, String, ForeignKey, Float, JSON, Boolean

from . import database


class Masks(database):
    __tablename__ = 'masks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'),
                      nullable=False)
    pixel_coverage = Column(Float, nullable=False)  # Percentage of pixels covered by the mask
    ai_generated = Column(Integer, nullable=False)  # 0 for manual, 1 for AI-generated aka no human involved


class Contours(database):
    __tablename__ = 'contours'
    id = Column(Integer, primary_key=True, autoincrement=True)
    mask_id = Column(Integer, ForeignKey('masks.id', ondelete='CASCADE'),
                     nullable=False)
    parent_id = Column(Integer, ForeignKey('contours.id', ondelete='CASCADE'))
    label = Column(Integer, ForeignKey('labels.id', ondelete='CASCADE'), nullable=False)
    auto_added = Column(Boolean, default=False, nullable=False)  # True if the contour was added automatically
    area = Column(Float, nullable=False)
    perimeter = Column(Float, nullable=False)
    circularity = Column(Float, nullable=False)
    diameters = Column(JSON, nullable=False)
    coords = Column(JSON, nullable=False)

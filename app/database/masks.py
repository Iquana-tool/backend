from sqlalchemy import Column, Integer, ForeignKey, Boolean
from . import database


class Masks(database):
    """ Represents a mask in the database. A mask holds all added contours."""
    __tablename__ = 'masks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'),
                      nullable=False)
    finished = Column(Boolean, default=False, nullable=False)  # True if the mask is finished
    generated = Column(Boolean, default=False, nullable=False)  # True if the mask was generated
    reviewed = Column(Boolean, default=False, nullable=False)  # True if the generated mask was reviewed

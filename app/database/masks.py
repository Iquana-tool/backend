from sqlalchemy import Column, Integer, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from . import database


class Masks(database):
    """ Represents a mask in the database. A mask holds all added contours."""
    __tablename__ = 'masks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'),
                      nullable=False)
    fully_annotated = Column(Boolean, default=False, nullable=False)  # Users can mark a mask as fully annotated indicating that all objects are there.
    contours = relationship("Contours", backref="mask")

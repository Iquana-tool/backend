from sqlalchemy import Column, Integer, String, Text, ForeignKey
from . import database


class Masks(database):
    __tablename__ = 'masks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'),
                      nullable=False)
    filename = Column(String, nullable=False)
    mask_label = Column(String, nullable=False)

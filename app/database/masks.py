from sqlalchemy import Column, Integer, String, Text, ForeignKey
from . import database


class Masks(database):
    __tablename__ = 'masks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'),
                      nullable=False)
    mask_label = Column(String, nullable=False)
    counter = Column(Integer, nullable=False, default=0)  # Counts the masks for the same image and mask_label

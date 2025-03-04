from sqlalchemy import Column, Integer, String, Text, ForeignKey
from . import database


class Cutouts(database):
    __tablename__ = 'cutouts'
    parent_image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'),
                             nullable=False)  # Image ID of the parent image,
    # aka the image from which the cutout was cut out
    image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'),
                      nullable=False, unique=True)  # The image id which represents this cutout
    lower_left_x = Column(Integer, nullable=False)  # X coordinate of the lower left corner of the cutout
    lower_left_y = Column(Integer, nullable=False)  # Y coordinate of the lower left corner of the cutout

    def __repr__(self):
        return (f"<Cutout(image_id='{self.image_id}', "
                f"parent_image_id='{self.parent_image_id}', "
                f"lower_left_x='{self.lower_left_x}',"
                f"lower_left_y='{self.lower_left_y}')>")

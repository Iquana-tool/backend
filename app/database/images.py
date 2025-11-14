from sqlalchemy import Column, Integer, String, ForeignKey, Float

from . import database


class Images(database):
    """
    Represents an image in the database. An image is part of a dataset and can be associated with masks and
    contours.
    """
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False)  # Foreign key to the datasets table

    file_name = Column(String, nullable=False)  # Path to the image file
    file_path = Column(String, nullable=False)  # Full path to the image file on disk
    thumbnail_file_path = Column(String, nullable=False)  # Full path to the thumbnail image file on disk
    description = Column(String, nullable=True)  # Optional description of the image

    width = Column(Integer, nullable=False)  # Width of the image in pixels
    height = Column(Integer, nullable=False)  # Height of the image in pixels
    channels = Column(Integer, nullable=False, default=3)  # Number of channels in the image (e.g., 3 for RGB)

    scale_x = Column(Float, nullable=False, default=1)  # mm per pixel in X
    scale_y = Column(Float, nullable=False, default=1)  # mm per pixel in Y
    unit = Column(String(10), default="px")  # Default unit: px (for pixels)

    scan_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'))  # Scan id for CT or MRI or OCT, etc.
    index_in_scan = Column(Integer)  # Index of the image in the scan

    def __repr__(self):
        return (f"<Image(id='{self.id}', "
                f"path='{self.file_name}', "
                f"width='{self.width}',"
                f"height='{self.height}')>"
                f"scale_x='{self.scale_x}', "
                f"scale_y='{self.scale_y}', "
                f"unit='{self.unit}')>")

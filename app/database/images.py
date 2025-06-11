from sqlalchemy import Column, Integer, String, ForeignKey, Float, JSON
from . import database


class Images(database):
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False)  # Foreign key to the datasets table
    file_name = Column(String, nullable=False)  # Path to the image file
    file_path = Column(String, nullable=False)  # Full path to the image file on disk
    width = Column(Integer, nullable=False)  # Width of the image in pixels
    height = Column(Integer, nullable=False)  # Height of the image in pixels
    hash_code = Column(String(64), nullable=False, unique=True)  # Hash of the image file
    scale_x = Column(Float, nullable=True)  # mm per pixel in X
    scale_y = Column(Float, nullable=True)  # mm per pixel in Y
    unit = Column(String(10), default="mm")  # Default unit: mm
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


class ImageEmbeddings(database):
    __tablename__ = 'image_embeddings'
    id = Column(Integer, primary_key=True, autoincrement=True)  # Unique ID for the embedding
    image_id = Column(Integer,
                      # Foreign key to the images table, cascade on delete to remove embeddings when image is deleted
                      ForeignKey('images.id', ondelete='CASCADE'),
                      nullable=False)  # Image ID is the primary key
    model = Column(String(50), nullable=False)  # The model used to generate the embedding
    embed_dimensions = Column(String(50), nullable=False)  # The dimensions of the embedding

    def __repr__(self):
        return (f"<ImageEmbedding(image_id='{self.image_id}', "
                f"model='{self.model}', "
                f"dimension='{self.embed_dimensions}'>")


class Scans(database):
    __tablename__ = 'scans'
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False)  # Foreign key to the datasets table
    name = Column(String, nullable=False)  # Name of the scan
    folder_path = Column(String, nullable=False)  # Path to the folder containing the scan images
    type = Column(String)  # Type of scan (e.g., 'CT', 'MRI')
    description = Column(String)  # Description of the scan
    number_of_slices = Column(Integer, nullable=False)  # Number of slices in the scan
    meta_data = Column(JSON)  # Save any additional metadata about the scan

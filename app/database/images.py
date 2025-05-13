from sqlalchemy import Column, Integer, String, ForeignKey

from . import database


class Images(database):
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False)  # Foreign key to the datasets table
    filename = Column(String, nullable=False)  # Path to the image file
    width = Column(Integer, nullable=False)  # Width of the image in pixels
    height = Column(Integer, nullable=False)  # Height of the image in pixels
    hash_code = Column(String(64), nullable=False, unique=True)  # Hash of the image file
    scan_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'))  # Scan id for CT or MRI or OCT, etc.
    index_in_scan = Column(Integer)  # Index of the image in the scan

    def __repr__(self):
        return (f"<Image(id='{self.id}', "
                f"path='{self.filename}', "
                f"width='{self.width}',"
                f"height='{self.height}')>")


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
    name = Column(String, nullable=False)  # Name of the scan
    type = Column(String)  # Type of scan (e.g., 'CT', 'MRI')
    description = Column(String)  # Description of the scan
    number_of_slices = Column(Integer, nullable=False)  # Number of slices in the scan
    meta_data = Column(JSON)  # Save any additional metadata about the scan

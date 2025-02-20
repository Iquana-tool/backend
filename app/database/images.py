from sqlalchemy import Column, Integer, String, Text, ForeignKey
from . import database


class Images(database):
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(String, nullable=False)  # Path to the image file
    type = Column(String, nullable=False)  # Type of image (e.g. 'png', 'jpg', 'tif')
    size = Column(Integer, nullable=False)  # Size of the image in bytes

    def __repr__(self):
        return (f"<Image(image_id='{self.image_id}', "
                f"image_path='{self.image_path}', "
                f"image_type='{self.image_type}',"
                f"image_size='{self.image_size}')>")


class ImageEmbeddings(database):
    __tablename__ = 'image_embeddings'
    id = Column(Integer, primary_key=True, autoincrement=True)  # Unique ID for the embedding
    image_id = Column(Integer,
                      # Foreign key to the images table, cascade on delete to remove embeddings when image is deleted
                      ForeignKey('images.id', ondelete='CASCADE'),
                      nullable=False, unique=True)  # Image ID is the primary key
    model = Column(String, nullable=False)  # The model used to generate the embedding
    dimensions = Column(String, nullable=False)  # The dimensions of the embedding (e.g. (256, 64, 64))
    vector = Column(Text, nullable=False)  # The flattened embedding vector as a string

    def __repr__(self):
        return (f"<ImageEmbedding(image_id='{self.image_id}', "
                f"model='{self.embedding_model}', "
                f"dimension='{self.embedding_dim}',"
                f"vector='{self.embedding}')>")

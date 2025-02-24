from sqlalchemy import Column, Integer, String, Text, ForeignKey
from . import database


class Images(database):
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(String, nullable=False)  # Path to the image file
    type = Column(String, nullable=False)  # Type of image (e.g. 'png', 'jpg', 'tif')
    size = Column(Integer, nullable=False)  # Size of the image in bytes

    def __repr__(self):
        return (f"<Image(image_id='{self.id}', "
                f"image_path='{self.path}', "
                f"image_type='{self.type}',"
                f"image_size='{self.size}')>")


class ImageEmbeddings(database):
    __tablename__ = 'image_embeddings'
    id = Column(Integer, primary_key=True, autoincrement=True)  # Unique ID for the embedding
    image_id = Column(Integer,
                      # Foreign key to the images table, cascade on delete to remove embeddings when image is deleted
                      ForeignKey('images.id', ondelete='CASCADE'),
                      nullable=False, unique=True)  # Image ID is the primary key
    model = Column(String, nullable=False)  # The model used to generate the embedding
    dimensions = Column(String, nullable=False)  # The dimensions of the embedding
    embed = Column(Text, nullable=False)  # The flattened embedding vector as a string
    high_res_features = Column(Text, nullable=False)  # The high resolution features as a string

    def __repr__(self):
        return (f"<ImageEmbedding(image_id='{self.image_id}', "
                f"model='{self.model}', "
                f"dimension='{self.dimensions}',"
                f"embed='{self.embed}', "
                f"high_res_features='{self.high_res_features}')>")

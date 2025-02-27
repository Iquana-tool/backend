import numpy as np
from PIL import Image
from fastapi import UploadFile
import config
from app.database.images import Images, ImageEmbeddings
from app.database import get_session, get_context_session
from logging import getLogger
import hashlib

logger = getLogger(__name__)


# Function to generate hash of a file
def generate_hash_for_image(image: UploadFile):
    """Generate a hash for the given image file."""
    hasher = hashlib.sha256()
    while True:
        data = image.file.read(65536)  # Read in 64k chunks
        if not data:
            break
        hasher.update(data)
    return hasher.hexdigest()


def load_image(image_id):
    """Load an image from the database by its ID."""
    image = Images.query.filter_by(id=image_id).first()
    if image:
        return np.array(Image.open(image.path))
    else:
        return None

def load_embedding(image_id):
    """Load an image embedding from the database by its image ID."""
    embedding = ImageEmbeddings.query.filter_by(image_id=image_id).first()
    if embedding:
        image_embed = np.fromstring(embedding.embed, sep=',')
        high_res_feats = np.fromstring(embedding.high_res_features, sep=',')
        image_embed = image_embed.reshape(embedding.dimensions)
        return {"image_embed": image_embed, "high_res_feats": high_res_feats}
    else:
        return None

async def save_image(image: UploadFile):
    """Save an image to disk and to the database and return the new image ID."""
    image_data = image.file.read()
    hash_code = generate_hash_for_image(image)
    with get_context_session() as session:
        if session.query(Images).filter_by(hash_code=hash_code).first():
            logger.info("Image already exists in the database.")
            return session.query(Images).filter_by(hash_code=hash_code).first().id
        else:
            next_id = session.query(Images).count() + 1
    original_extension = image.filename.split(".")[-1]
    new_file_name = f"{next_id}.{original_extension}"
    path = f"{config.Paths.meso_dir}\\{new_file_name}"
    with open(path, "wb") as file:
        file.write(image_data)
    image_array = np.array(Image.open(path))
    with get_context_session() as session:
        session.add(Images(path=new_file_name,
                           width=image_array.shape[1],
                           height=image_array.shape[0],
                           hash_code=hash_code))
        session.commit()
    logger.info("New image saved to disk and database.")
    return session.query(Images).order_by(Images.id.desc()).first().id


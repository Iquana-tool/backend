import numpy as np
from PIL import Image
from app.database.images import Images, ImageEmbeddings


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

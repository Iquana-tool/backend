import numpy as np
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import ValidationError
import logging
from sqlalchemy.orm import Session

import config
from app.services.segmentation import segment_with_prompts, segment_without_prompts, embed_image
from app.services.prompts import Prompts
from app.services.database_access import load_image, load_embedding, save_embeddings_to_disk
from app.services.postprocessing import rle_encode
from app.schemas.segmentation import SegmentationRequest, SegmentationResponse
from app.database import get_session
from app.database.images import ImageEmbeddings, Images
from app.schemas.util import validate_request

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/segmentation")


@router.post('/segment_image')
async def segment_image(request: SegmentationRequest, db: Session = Depends(get_session)):
    """Perform segmentation with optional prompts, using data validation."""
    if request.use_prompts:
        prompts = Prompts()
        for point in request.point_prompts:
            prompts.add_point_annotation(point.x, point.y, point.label)

        for box in request.box_prompts:
            prompts.add_box_annotation(box.min_x, box.min_y, box.max_x, box.max_y)

        width = db.query(Images).filter_by(id=request.image_id).first().width
        height = db.query(Images).filter_by(id=request.image_id).first().height
        embedding = load_embedding(request.image_id)
        if embedding is None:
            # Image has not been embedded yet
            embedding = embed_image(load_image(request.image_id))
            new_embedding = ImageEmbeddings(
                image_id=request.image_id,
                model=config.ModelConfig.selected_model,
                embed_dimensions=str(embedding["image_embed"].shape),
            )
            db.add(new_embedding)
            db.commit()
            save_embeddings_to_disk(embedding, new_embedding.id)
        masks, quality = segment_with_prompts(embedding, (width, height), prompts)
    else:
        image = load_image(request.image_id)
        masks, quality = segment_without_prompts(image)

    response = {"rle_masks": [rle_encode(mask) for mask in masks],
                "quality": quality.tolist()}
    return SegmentationResponse(**response)


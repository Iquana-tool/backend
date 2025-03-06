import cv2
import numpy as np
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import ValidationError
import logging
from sqlalchemy.orm import Session

import config
from app.services.segmentation.sam2 import SAM2
from app.services.prompts import Prompts
from app.services.database_access import load_image_as_array_from_disk, load_embedding, save_embeddings_to_disk
from app.services.postprocessing import base64_encode_image
from app.schemas.segmentation import SegmentationRequest
from app.database import get_session
from app.database.images import ImageEmbeddings, Images

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/segmentation", tags=["segmentation"])


@router.post('/segment_image')
async def segment_image(request: SegmentationRequest, db: Session = Depends(get_session)):
    """Perform segmentation with optional prompts, using data validation."""
    model = SAM2(config.ModelConfig.available_models[request.model]())
    if request.use_prompts:
        prompts = Prompts()
        for point in request.point_prompts:
            prompts.add_point_annotation(point.x, point.y, point.label)

        for box in request.box_prompts:
            prompts.add_box_annotation(box.min_x, box.min_y, box.max_x, box.max_y)
        embedding = db.query(ImageEmbeddings).filter_by(image_id=request.image_id, model=request.model).first()
        width = db.query(Images).filter_by(id=request.image_id).first().width
        height = db.query(Images).filter_by(id=request.image_id).first().height
        if embedding is not None:
            embedding = load_embedding(embedding.id)
        else:
            # Image has not been embedded yet
            image = load_image_as_array_from_disk(request.image_id)
            if image.shape[-1] != 3:
                logger.warning("Converting RGBA image to RGB.")
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            embedding = model.embed_image(image)
            new_embedding = ImageEmbeddings(
                image_id=request.image_id,
                model=request.model,
                embed_dimensions=str(embedding["image_embed"].shape),
            )
            db.add(new_embedding)
            db.commit()
            save_embeddings_to_disk(embedding, new_embedding.image_id, new_embedding.model)
        masks, quality = model.segment_with_prompts(embedding, (height, width), prompts)
    else:
        image = load_image_as_array_from_disk(request.image_id)
        masks, quality = model.segment_without_prompts(image)

    return {"base64_masks": [base64_encode_image(mask) for mask in masks],
            "quality": quality.tolist()}

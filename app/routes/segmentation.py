from fastapi import APIRouter, Request, HTTPException
from pydantic import ValidationError
import logging
import config
from app.services.segmentation import segment_with_prompts, segment_without_prompts, embed_image
from app.services.prompts import Prompts
from app.services.dataloader import load_image, load_embedding
from app.schemas.segmentation_schemas import SegmentationResponseSchema, SegmentationRequestSchema
from app.database.images import ImageEmbeddings

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/segmentation")

@router.post('/segment_image')
async def segment_image(request: Request):
    """Perform segmentation with optional prompts, using data validation."""
    try:
        request_data = await request.json()
        validated_data = SegmentationRequestSchema(**request_data)
    except ValidationError as err:
        raise HTTPException(status_code=400, detail=err.errors())

    if validated_data.use_prompts:
        prompts = Prompts()

        for point in validated_data.point_prompts:
            prompts.add_point_annotation(point.x, point.y, point.label)

        for box in validated_data.box_prompts:
            prompts.add_box_annotation(box.min_x, box.min_y, box.max_x, box.max_y)

        embedding = load_embedding(validated_data.image_id)
        if embedding is None:
            # Image has not been embedded yet
            embedding = embed_image(load_image(validated_data.image_id))
            ImageEmbeddings.create(
                image_id=validated_data.image_id,
                model=config.ModelConfig.selected_model,
                dimensions=embedding["image_embed"].shape,
                vector=embedding["image_embed"],
                high_res_features=embedding["high_res_feats"]
            )
        masks, quality = segment_with_prompts(embedding, prompts)
    else:
        image = load_image(validated_data.image_id)
        masks, quality = segment_without_prompts(image)

    response = {"masks": masks.tolist(), "quality": quality.tolist()}
    return SegmentationResponseSchema(**response)

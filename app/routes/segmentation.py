from fastapi import APIRouter, UploadFile, File, HTTPException
import logging
from flask import Blueprint, request, jsonify
from marshmallow import ValidationError

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

# Create Flask blueprint
segmentation_bp = Blueprint('segmentation', __name__)

request_schema = SegmentationRequestSchema()
response_schema = SegmentationResponseSchema()


@router.post('/segment_image')
def segment_image():
    """ Perform segmentation with optional prompts, using data validation. """
    try:
        request_data = request.json or {}
        validated_data = request_schema.load(request_data)
    except ValidationError as err:
        return jsonify({"error": err.messages}), 400

    if validated_data.get("use_prompts"):
        prompts = Prompts()

        for point in validated_data.get("point_prompts", []):
            prompts.add_point_annotation(point["x"], point["y"], point["label"])

        for box in validated_data.get("box_prompts", []):
            prompts.add_box_annotation(box["min_x"], box["min_y"], box["max_x"], box["max_y"])
        embedding = load_embedding(validated_data["image_id"])
        if embedding is None:
            # Image has not been embedded yet
            embedding = embed_image(load_image(validated_data["image_id"]))
            ImageEmbeddings.create(image_id=validated_data["image_id"],
                                   model=config.ModelConfig.selected_model,
                                   dimensions=embedding["image_embed"].shape,
                                   vector=embedding["image_embed"],
                                   high_res_features=embedding["high_res_feats"])
        masks, quality = segment_with_prompts(embedding, prompts)
    else:
        image = load_image(validated_data["image_id"])
        masks, quality = segment_without_prompts(image)
    response = {"masks": masks.tolist(), "quality": quality.tolist()}
    return jsonify(response_schema.dump(response))
from flask import Blueprint, request, jsonify
import numpy as np
from PIL import Image
from io import BytesIO
from marshmallow import ValidationError

from app.services.segmentation.sam2 import SAM2
from app.services.prompts import Prompts
from app.schemas.sam2_schemas import SAM2ResponseSchema, SAM2RequestSchema

# Initialize the model
sam2_model = SAM2()

# Create Flask blueprint
segmentation_bp = Blueprint('segmentation', __name__)


def process_image(image_file):
    """ Convert image file to a numpy array. """
    image = Image.open(BytesIO(image_file.read())).convert("RGB")
    return np.array(image)


request_schema = SAM2RequestSchema()
response_schema = SAM2ResponseSchema()


@segmentation_bp.route('/segment', methods=['POST'])
def segment_image():
    """ Perform segmentation with optional prompts, using data validation. """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = process_image(request.files['image'])

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

        masks, quality = sam2_model.segment_with_prompts(image, prompts)
        response = {"masks": masks.tolist(), "quality": quality.tolist()}
    else:
        response = {"segmentation_result": sam2_model.segment_without_prompts(image)}

    return jsonify(response_schema.dump(response))



import logging

import cv2
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import config
from app.database import get_session
from app.database.images import ImageEmbeddings, Images
from app.schemas.segmentation_and_masks import (
    SegmentationRequest, SegmentationResponse, ContourModel, 
    SegmentationMaskModel, QuantificationsModel
)
from app.schemas.scale import ScaleInput
from app.services.database_access import load_image_as_array_from_disk, load_embedding, save_embeddings_to_disk
from app.services.prompts import Prompts
from app.services.segmentation.sam2 import SAM2, set_current_image_id
from app.services.contours import get_contours
from app.services.quantifications import Contour
from app.services.scale_computation import compute_pixel_scale_from_points

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/segmentation", tags=["segmentation"])


@router.post('/segment_image')
async def segment_image(request: SegmentationRequest, db: Session = Depends(get_session)):
    set_current_image_id(request.image_id)
    model = SAM2(config.ModelConfig.available_models[request.model]())
    embedding = db.query(ImageEmbeddings).filter_by(image_id=request.image_id, model=request.model).first()
    image_record = db.query(Images).filter_by(id=request.image_id).first()
    width, height = image_record.width, image_record.height
    use_crop = request.min_x > 0 or request.min_y > 0 or request.max_x < 1 or request.max_y < 1

    if use_crop:
        width = int((request.max_x - request.min_x) * width)
        height = int((request.max_y - request.min_y) * height)

    if request.use_prompts:
        prompts = Prompts()
        for point in request.point_prompts:
            prompts.add_point_annotation(point.x, point.y, point.label)
        for box in request.box_prompts:
            prompts.add_box_annotation(box.min_x, box.min_y, box.max_x, box.max_y)
        for polygon in request.polygon_prompts:
            prompts.add_polygon_annotation(polygon.vertices)
        for circle in request.circle_prompts:
            prompts.add_circle_annotation(circle.center_x, circle.center_y, circle.radius)

        if embedding is not None and not use_crop:
            embedding = load_embedding(embedding.id, request.model)
        else:
            image = load_image_as_array_from_disk(request.image_id)
            if image.shape[-1] != 3:
                logger.warning("Converting RGBA image to RGB.")
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            if use_crop:
                image = image[int(request.min_y * height):int(request.max_y * height),
                              int(request.min_x * width):int(request.max_x * width)]
            embedding = model.embed_image(image)
            if not use_crop:
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

    masks_response = []
    for mask, quality in zip(masks, quality):
        contours = get_contours(mask)
        contours_response = []
        for contour in contours:
            if len(contour) < 3:
                continue
            contour = Contour(contour)
            contours_response.append(ContourModel(
                x=[list_val[0] / width for list_val in contour.contour[..., 0].tolist()],
                y=[list_val[0] / height for list_val in contour.contour[..., 1].tolist()],
                label=request.label,
                quantifications=QuantificationsModel(
                    area=contour.area,
                    perimeter=contour.perimeter,
                    circularity=contour.circularity,
                    diameters=contour.get_diameters(100)
                )
            ))
        masks_response.append(SegmentationMaskModel(contours=contours_response, predicted_iou=quality))

    return SegmentationResponse(masks=masks_response, image_id=request.image_id, model=request.model)


@router.post('/set_scale')
def set_pixel_scale(scale_input: ScaleInput, db: Session = Depends(get_session)):
    image = db.query(Images).filter_by(id=scale_input.image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    scale_x, scale_y = compute_pixel_scale_from_points(
        (scale_input.x1, scale_input.y1),
        (scale_input.x2, scale_input.y2),
        scale_input.known_distance
    )

    image.scale_x = scale_x
    image.scale_y = scale_y
    image.unit = scale_input.unit or "mm"
    db.commit()

    return {"message": "Scale set successfully", "scale_x": scale_x, "scale_y": scale_y, "unit": image.unit}

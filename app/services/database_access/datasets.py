import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
import json
from logging import getLogger
from typing import Any, Literal

import numpy as np
import pandas as pd
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.image import Image
from iquana_toolbox.schemas.database.labels import LabelHierarchy
from iquana_toolbox.schemas.user import User
from sqlalchemy.orm import Session

from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.services.auth import get_current_user
from app.services.database_access.labels import get_hierarchical_label_name
from config import DATASETS_DIR

logger = getLogger(__name__)


async def create_new_dataset(
        name: str,
        description: str,
        owner_username: str,
        db: Session
):
    # Check if dataset with the same name already exists
    existing_dataset = db.query(Datasets).filter_by(name=name.strip()).first()
    if existing_dataset:
        return {"success": False,
                "message": f"Dataset with name '{name.strip()}' already exists.",
                "error": "Duplicate dataset name"}

    dataset_path = os.path.join(DATASETS_DIR, name.strip())
    # Use exist_ok=True to avoid FileExistsError if directory already exists
    os.makedirs(dataset_path, exist_ok=True)

    new_dataset = Datasets(
        name=name.strip(),
        description=description.strip(),
        folder_path=dataset_path,
        dataset_type="image",
        created_by=owner_username,
    )
    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)
    return new_dataset


async def share_dataset(
        dataset_id: int,
        share_with_username: str,
        sharing_username: str,
        db: Session
):
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if dataset.created_by != sharing_username:
        raise PermissionError("Only the owner can share a dataset")
    user_to_share = db.query(Users).filter_by(username=share_with_username).first()
    if not user_to_share:
        raise ValueError("User to share with not found")
    if not user_to_share in dataset.shared_with:
        dataset.shared_with.append(user_to_share)
        db.commit()


async def user_has_sharing_permission_for_dataset(
        dataset_id: int,
        sharing_username: str,
        db: Session
):
    """ Check whether a user can share a dataset. """
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    return sharing_username == dataset.created_by


async def get_dataset(
        dataset_id: int,
        db: Session
):
    return db.query(Datasets).filter_by(id=dataset_id).first()


async def get_num_of_images_in_dataset(
        dataset_id: int,
        db: Session
):
    return db.query(Images).filter_by(dataset_id=dataset_id).count()


async def get_annotation_progress_of_dataset(
        dataset_id: int,
        db: Session
):
    masks = (
        db.query(Masks)
        .join(Images, Masks.image_id == Images.id)
        .filter(Images.dataset_id == dataset_id).all()
    )
    status_dict = defaultdict(lambda: 0)
    for mask in masks:
        status_dict[mask.status] += 1
    return status_dict, len(masks)


async def get_datasets_of_user(
        user: User,
        db: Session
):
    datasets = db.query(Datasets).filter(Datasets.id.in_(user.available_datasets))
    return datasets


async def get_label_hierarchy_of_dataset(
        dataset_id: int,
        db: Session
) -> LabelHierarchy:
    labels = db.query(Labels).filter_by(dataset_id=dataset_id)
    return LabelHierarchy.from_query(labels)


async def has_dataset_deletion_permission(
        dataset_id: int,
        username: str,
        db: Session
):
    raise NotImplementedError


async def delete_dataset(
        dataset_id: int,
        db: Session
):
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        return {"success": False, "message": "Dataset not found."}
    dataset_folder = str(dataset.folder_path)
    # Delete the dataset
    db.delete(dataset)
    db.commit()
    # Delete disk directory, removes all image files.
    shutil.rmtree(dataset_folder, ignore_errors=True)


async def get_image_and_mask_ids_of_dataset(
        dataset_id: int,
        db: Session,
        filter_for_status: Literal["not_started", "in_progress", "reviewable", "finished"] | None = None,

):
    query = db.query(Images, Masks).join(Masks, Images.id == Masks.image_id).filter(Images.dataset_id == dataset_id)
    if filter_for_status:
        query = query.filter(Masks.status == filter_for_status)
    result = query.all()
    image_data = [
        {
            "image_id": img.id,
            "mask_id": mask.id,
            "status": mask.status
        } for img, mask in result
    ]
    return image_data


async def get_images_of_dataset(
        dataset_id: int,
        db: Session,
        limit: int = None,
        as_thumbnail: bool = False,
        as_base64: bool = False,

):
    response = {}
    images_query = db.query(Images).filter_by(dataset_id=dataset_id).limit(limit).all()
    images = [Image.from_db(img) for img in images_query]
    for img in images:
        if as_thumbnail:
            response[img.id] = img.load_thumbnail(as_base64=as_base64)
        else:
            response[img.id] = img.load_image(as_base64=as_base64)
    return response


async def get_dataset_as_df(
        dataset_id: int,
        exclude_not_fully_annotated: bool,
        exclude_unreviewed: bool,
        db: Session,
):
    query = (
        db.query(Contours, Images.file_name, Labels)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .join(Labels, Labels.id == Contours.label_id)
        .filter(Images.dataset_id == dataset_id)
    )
    if exclude_not_fully_annotated:
        query = query.filter(Masks.fully_annotated == True)
    if exclude_unreviewed:
        query = query.filter(Contours.reviewed_by.any())

    data = query.all()
    df_data = {}
    for row in data:
        contour: Contours = row[0]
        file_name: str = row[1]
        label_db: Labels = row[2]

        df_data.setdefault("file_name", []).append(file_name)
        df_data.setdefault("label", []).append(label_db.name)
        df_data.setdefault("label_id", []).append(contour.label_id)
        df_data.setdefault("contour_id", []).append(contour.id)
        df_data.setdefault("area", []).append(contour.area)
        df_data.setdefault("perimeter", []).append(contour.perimeter)
        df_data.setdefault("circularity", []).append(contour.circularity)
        df_data.setdefault("diameter_avg", []).append(contour.diameter)
        df_data.setdefault("coords_x", []).append(contour.x)
        df_data.setdefault("coords_y", []).append(contour.y)
    return pd.DataFrame(df_data)


def _build_coco_polygon(
        x_coords: list[float],
        y_coords: list[float],
        width: int,
        height: int,
) -> list[float]:
    """Convert contour points to a flat COCO polygon list."""
    if not x_coords or not y_coords:
        return []

    # Contours are usually stored normalized in [0, 1], but support legacy pixel coordinates.
    is_normalized = (
            max(abs(float(v)) for v in x_coords) <= 1.5
            and max(abs(float(v)) for v in y_coords) <= 1.5
    )
    scale_x = float(width) if is_normalized else 1.0
    scale_y = float(height) if is_normalized else 1.0

    polygon = []
    for x, y in zip(x_coords, y_coords):
        polygon.append(float(x) * scale_x)
        polygon.append(float(y) * scale_y)
    return polygon


async def export_dataset_contours_to_coco(
        dataset_id: int,
        db: Session,
        exclude_not_fully_annotated: bool = True,
        exclude_unreviewed: bool = True,
        output_file_path: str | None = None,
        log_to_mlflow: bool = False,
        mlflow_run_id: str | None = None,
) -> dict[str, Any]:
    """Export all contours of a dataset to COCO JSON and save to disk."""
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        return {"success": False, "message": "Dataset not found.", "dataset_id": dataset_id}

    query = (
        db.query(Contours, Images, Labels)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .outerjoin(Labels, Labels.id == Contours.label_id)
        .filter(Images.dataset_id == dataset_id)
    )
    if exclude_not_fully_annotated:
        query = query.filter(Masks.fully_annotated == True)
    if exclude_unreviewed:
        query = query.filter(Contours.reviewed_by.any())

    rows = query.all()

    images_by_id: dict[int, dict[str, Any]] = {}
    categories_by_id: dict[int, dict[str, Any]] = {}
    annotations: list[dict[str, Any]] = []

    for contour, image, label in rows:
        if contour.label_id is None or label is None:
            continue

        if image.id not in images_by_id:
            images_by_id[image.id] = {
                "id": image.id,
                "file_name": image.file_name,
                "width": image.width,
                "height": image.height,
            }

        if label.id not in categories_by_id:
            categories_by_id[label.id] = {
                "id": label.id,
                "name": label.name,
                "supercategory": "none",
            }

        polygon = _build_coco_polygon(contour.x, contour.y, image.width, image.height)
        if len(polygon) < 6:
            continue

        x_points = polygon[0::2]
        y_points = polygon[1::2]
        min_x, max_x = min(x_points), max(x_points)
        min_y, max_y = min(y_points), max(y_points)
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

        annotations.append({
            "id": contour.id,
            "image_id": image.id,
            "category_id": label.id,
            "segmentation": [polygon],
            "area": float(contour.area) if contour.area is not None else float(bbox[2] * bbox[3]),
            "bbox": bbox,
            "iscrowd": 0,
        })

    now_utc = datetime.now(timezone.utc)

    coco_payload: dict[str, Any] = {
        "info": {
            "description": dataset.description or f"COCO export for dataset {dataset.name}",
            "version": "1.0",
            "year": now_utc.year,
            "date_created": now_utc.isoformat(),
        },
        "licenses": [],
        "images": list(images_by_id.values()),
        "annotations": annotations,
        "categories": list(categories_by_id.values()),
    }

    if output_file_path is None:
        output_file_path = os.path.join(str(dataset.folder_path), f"{dataset.name.replace(' ', '_')}_coco.json")
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file_path, "w", encoding="utf-8") as fp:
        json.dump(coco_payload, fp, indent=2)

    mlflow_result = "not_requested"
    if log_to_mlflow:
        try:
            import mlflow

            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow.log_artifact(output_file_path, artifact_path="coco_exports")
            elif mlflow_run_id:
                with mlflow.start_run(run_id=mlflow_run_id):
                    mlflow.log_artifact(output_file_path, artifact_path="coco_exports")
            else:
                with mlflow.start_run(run_name=f"dataset_{dataset_id}_coco_export"):
                    mlflow.log_artifact(output_file_path, artifact_path="coco_exports")
            mlflow_result = "logged"
        except Exception as exc:
            logger.warning("Could not log COCO export to MLflow: %s", exc)
            mlflow_result = f"failed: {exc}"

    return {
        "success": True,
        "message": "COCO export written to disk.",
        "dataset_id": dataset_id,
        "output_file_path": output_file_path,
        "num_images": len(images_by_id),
        "num_annotations": len(annotations),
        "num_categories": len(categories_by_id),
        "mlflow": mlflow_result,
    }



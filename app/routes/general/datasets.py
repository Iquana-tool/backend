import io
import json
import os
import shutil
import zipfile
from collections import defaultdict
from io import StringIO
from logging import getLogger
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from iquana_toolbox.schemas.contour_hierarchy import ContourHierarchy
from iquana_toolbox.schemas.image import Image
from iquana_toolbox.schemas.labels import LabelHierarchy
from iquana_toolbox.schemas.user import User
from sqlalchemy.orm import Session
from starlette import status
from starlette.responses import StreamingResponse

from app.database import get_session
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.routes.general.masks import get_mask_annotation_status
from app.services.auth import get_current_user
from app.services.labels import get_hierarchical_label_name
from app.services.util import get_mask_path_from_image_path
from config import DATASETS_DIR

# Create a router for the export functionality
router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = getLogger(__name__)

@router.post("/create")
async def create_dataset(name: str,
                         description: str,
                         dataset_type: Literal["image", "scan", "DICOM"],
                         db: Session = Depends(get_session),
                         current_user=Depends(get_current_user)):
    """Create a new dataset.

    Args:
        name (str): The name of the dataset.
        description (str): A brief description of the dataset.
        dataset_type (Literal["image", "scan", "DICOM"]): The type of dataset.
        db (Session): The database session.
        current_user (Users): Auth bearer token.

    Returns:
        dict: A dictionary containing the success status and message, or error details.
    """
    try:
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
            dataset_type=dataset_type,
            created_by=current_user.username,
        )
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)
        return {"success": True,
                "message": "Dataset created successfully.",
                "dataset_id": new_dataset.id
                }
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        db.rollback()
        return {"success": False,
                "message": "Error creating dataset.",
                "error": str(e)}

@router.post("/{dataset_id}/share")
async def share_dataset(
    dataset_id: int,
    share_with_username: str,
    db: Session = Depends(get_session),
    user: "User" = Depends(get_current_user)
):
    """Share a dataset with another user by username.

    Args:
        dataset_id (int): The ID of the dataset to share.
        share_with_username (str): The username to share with.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.created_by != user.username:
        raise HTTPException(status_code=403, detail="Only the owner can share this dataset")
    user_to_share = db.query(Users).filter_by(username=share_with_username).first()
    if not user_to_share:
        raise HTTPException(status_code=404, detail="User to share with not found")
    if user_to_share in dataset.shared_with:
        return {"success": True, "message": "User already has access"}
    dataset.shared_with.append(user_to_share)
    db.commit()
    return {"success": True, "message": f"Dataset shared with {share_with_username}"}


@router.get("/all")
async def get_all_datasets(
    db: Session = Depends(get_session),
    user: "User" = Depends(get_current_user)
):
    """Get all datasets owned by or shared with the current user.

    Args:
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and the list of datasets.
    """
    available_datasets = user.available_datasets
    datasets = db.query(Datasets).filter(
        Datasets.id.in_(available_datasets)
    ).all()
    return {"success": True, "datasets": [
        {
            "id": ds.id,
            "name": ds.name,
            "description": ds.description,
            "dataset_type": ds.dataset_type,
            "folder_path": ds.folder_path,
            "created_by": ds.created_by,
            "shared_with": [u.username for u in ds.shared_with]
        }
        for ds in datasets
    ]}


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_session),
    user: "User" = Depends(get_current_user)
):
    """Get dataset information.

    Args:
        dataset_id (int): The ID of the dataset.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and dataset information.
    """
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        return {"success": False, "message": "Dataset not found."}
    return {"success": True, "message": "Dataset found.", "dataset": dataset}


@router.get("/{dataset_id}/images/count")
async def get_number_of_images(
    dataset_id: int,
    db: Session = Depends(get_session),
    user: "User" = Depends(get_current_user)
):
    """Get the number of images in a dataset.

    Args:
        dataset_id (int): The ID of the dataset.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the number of images.
    """
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        return {"success": False, "message": "Dataset not found."}
    number_of_images = db.query(Images).filter_by(dataset_id=dataset_id).count()
    return {"success": True, "number_of_images": number_of_images}


@router.get("/{dataset_id}/progress")
async def get_annotation_progress(dataset_id: int,
                                  user: User = Depends(get_current_user),
                                  db: Session = Depends(get_session)):
    """Get the annotation progress of a dataset.

    Args:
        dataset_id (int): The ID of the dataset to check.
        user (Users): Authentication bearer token.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the annotation progress details. The dict contains:
            - success (bool): Indicates if the operation was successful.
            - message (str): A message indicating the result of the operation.
            - manually_annotated (int): Number of images manually annotated.
            - auto_annotated_reviewed (int): Number of images auto-annotated with review.
            - auto_annotated_without_review (int): Number of images auto-annotated without review.
            - missing (int): Number of images missing annotations.
            - total_images (int): Total number of images in the dataset.
    """
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        return {"success": False, "message": "Dataset not found."}
    images = db.query(Images).filter_by(dataset_id=dataset_id).all()
    status_dict = defaultdict(lambda: 0)
    for image in images:
        mask = db.query(Masks).filter_by(image_id=image.id).first()
        status_response = await get_mask_annotation_status(mask.id, db, user)
        status_dict[status_response["status"]] += 1
    return {
        "success": True,
        "message": "Annotation progress retrieved successfully.",
        "total_images": len(images),
        "num_masks_with_status": status_dict,
    }


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_session),
    user: "User" = Depends(get_current_user)
):
    """Delete a dataset.

    Args:
        dataset_id (int): The ID of the dataset to delete.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    if dataset_id not in user.owned_datasets:
        return {"success": False, "message": "Dataset not owned by user. Only the owner can delete this dataset."}
    try:
        dataset = db.query(Datasets).filter_by(id=dataset_id).first()
        if not dataset:
            return {"success": False, "message": "Dataset not found."}
        # Delete disk directory, removes all image files.
        shutil.rmtree(dataset.folder_path, ignore_errors=True)
        # Delete the dataset
        db.delete(dataset)
        db.commit()
        return {"success": True, "message": "Dataset deleted successfully."}
    except Exception as e:
        logger.error(e)
        raise e
        return {"success": False, "message": "Error deleting dataset.", "error": str(e)}


@router.get("/{dataset_id}/images")
async def list_images(
    dataset_id: int,
    filter_for_status: Literal["not_started", "in_progress", "reviewable", "finished"] | None = None,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """List all images with masks of certain status for a given image ID.

    Args:
        dataset_id: Dataset ID to retrieve images from.
        filter_for_status: The status of the masks to filter by.
        db: Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A list of image IDs.
    """
    query = db.query(Images, Masks).join(Masks, Images.id == Masks.image_id).filter(Images.dataset_id == dataset_id)
    if filter_for_status:
        match filter_for_status:
            case "not_started":
                query = query.filter(Masks.status == "not_started")
            case "in_progress":
                query = query.filter(Masks.status == "in_progress")
            case "reviewable":
                query = query.filter(Masks.status == "reviewable")
            case "finished":
                query = query.filter(Masks.status == "finished")
            case _:
                raise HTTPException(status_code=403, detail="Unknown status.")
    result = query.all()
    image_data = [
        {
            "image_id": img.id,
            "mask_id": mask.id,
            "status": mask.status
        } for img, mask in result
    ]
    return {
        "success": True,
        "message": "Retrieved images successfully.",
        "image_data": image_data
    }


@router.get("/{dataset_id}/images/b64")
async def get_base64_images_of_dataset(
    dataset_id: int,
    limit: int = None,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """Get all images of a dataset.

    Args:
        dataset_id: ID of the dataset to retrieve images from.
        low_res: Whether to return low resolution images (thumbnails).
        limit: Optional limit on the number of images to return. If not provided, all images will be returned.
        db: Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A dict mapping from image ID to base64 encoded image.
    """
    try:
        response = {}
        images_query = db.query(Images).filter_by(dataset_id=dataset_id).limit(limit).all()
        images = [Image.from_db(img) for img in images_query]
        for img in images:
            response[id] = img.load_image(as_base64=True)
        return {
            "success": True,
            "message": f"Successfully retrieved {len(images)} images from dataset {dataset_id}.",
            "images": response
        }
    except Exception as e:
        logger.error(f"Get all images of dataset error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/thumbnails/b64")
async def get_base64_thumbnails_of_dataset(
    dataset_id: int,
    limit: int = None,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """Get all images of a dataset.

    Args:
        dataset_id: ID of the dataset to retrieve images from.
        low_res: Whether to return low resolution images (thumbnails).
        limit: Optional limit on the number of images to return. If not provided, all images will be returned.
        db: Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A dict mapping from image ID to base64 encoded image.
    """
    try:
        response = {}
        images_query = db.query(Images).filter_by(dataset_id=dataset_id).limit(limit).all()
        images = [Image.from_db(img) for img in images_query]
        for img in images:
            response[id] = img.load_image(as_base64=True)
        return {
            "success": True,
            "message": f"Successfully retrieved {len(images)} images from dataset {dataset_id}.",
            "images": response
        }
    except Exception as e:
        logger.error(f"Get all images of dataset error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/labels")
async def get_labels(
    dataset_id: int,
    db: Session = Depends(get_session),
    user: User = Depends(get_current_user)
):
    """Retrieve all labels for a given dataset.

    Args:
        dataset_id (int): The ID of the dataset.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and the labels hierarchy.
    """
    labels = db.query(Labels).filter_by(dataset_id=dataset_id)
    labels_hierarchy = LabelHierarchy.from_query(labels)
    return {
        "success": True,
        "message": f"Retrieved {len(labels_hierarchy)} labels for dataset {dataset_id}.",
        "labels": labels_hierarchy.model_dump()
    }


@router.get(
    "/{dataset_id}/quantification")
async def get_dataset_quantification(
        dataset_id: int,
        exclude_unreviewed: bool = True,
        exclude_not_fully_annotated: bool = True,
        as_download: bool = False,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """
    Export quantification data for the given dataset_id and labels.

    Args:
        dataset_id (int): The ID of the dataset to export.
        exclude_not_fully_annotated (bool): Whether to exclude not fully annotated masks.
        exclude_unreviewed (bool): Whether to exclude unreviewed contours.
        as_download (bool, optional): Whether to export as CSV. Defaults to False. If False, returns the data as a json.
        db (Session, optional): The database session. Defaults to Depends(get_session). This is a fastapi dependency.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and message if error, or a
        StreamingResponse with the CSV file.
    """
    if dataset_id not in user.available_datasets:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access to this dataset.")
    query = (db.query(Contours, Images.file_name)
    .join(Masks, Masks.id == Contours.mask_id).join(Images, Images.id == Masks.image_id).filter(
        Images.dataset_id == dataset_id
    ))
    if exclude_not_fully_annotated:
        query = query.filter(Masks.fully_annotated == True)
    if exclude_unreviewed:
        query = query.filter(Contours.reviewed_by.any())

    dataset_name = db.query(Datasets).filter_by(id=dataset_id).first().name

    data = query.all()
    df_data = {}
    for row in data:
        contour, file_name, finished, generated = row
        label_name = get_hierarchical_label_name(contour.label_id)
        diameters = json.loads(contour.diameters) if isinstance(contour.diameters, str) else contour.diameters
        coords = json.loads(contour.coords) if isinstance(contour.coords, str) else contour.coords

        df_data.setdefault("file_name", []).append(file_name)
        df_data.setdefault("label", []).append(label_name)
        df_data.setdefault("label_id", []).append(contour.label_id)
        df_data.setdefault("contour_id", []).append(contour.id)
        df_data.setdefault("area", []).append(contour.area)
        df_data.setdefault("perimeter", []).append(contour.perimeter)
        df_data.setdefault("circularity", []).append(contour.circularity)
        df_data.setdefault("diameter_avg", []).append(str(np.mean(diameters)) if diameters else None)
        df_data.setdefault("coords_x", []).append(str(coords["x"]) if coords else None)
        df_data.setdefault("coords_y", []).append(str(coords["y"]) if coords else None)
        df_data.setdefault("centroid_x", []).append(str(np.mean(coords["x"])) if coords else None)
        df_data.setdefault("centroid_y", []).append(str(np.mean(coords["y"])) if coords else None)
        df_data.setdefault("finished", []).append(finished)
        df_data.setdefault("generated", []).append(generated)
    df = pd.DataFrame(df_data)
    if df.empty:
        return {
            "success": False,
            "message": "No data found for the given dataset and filters."
        }
    else:
        if as_download:
            # Convert to CSV
            csv_content = df.to_csv(index=False)
            response = StreamingResponse(StringIO(csv_content), media_type="text/csv")
            response.headers[
                "Content-Disposition"] = f'attachment; filename="{dataset_name.replace(' ', '_')}_dataset.csv"'
            return response
        else:
            return {
                "success": True,
                "message": "Successfully exported the dataset as json.",
                "data": df.to_json(orient="records"),
            }


@router.get(
    "/get_dataset_object_quantifications/{dataset_id}?&exclude_unreviewed_objects={exclude_unreviewed_objects}",
    deprecated=True,
)
async def get_dataset_object_quantifications(dataset_id: int,
                                             exclude_unreviewed_objects: bool = False,
                                             db: Session = Depends(get_session),
                                             user: User = Depends(get_current_user),
                                             ):
    if dataset_id not in user.available_datasets:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access to this dataset.")
    images = db.query(Images).filter(Images.dataset_id == dataset_id).all()
    metrics_per_label = defaultdict(lambda: defaultdict(list))
    child_counts_per_label = defaultdict(lambda: defaultdict(list))
    for image in images:
        mask = db.query(Masks).filter(Masks.image_id == image.id).first()  # Only one should exist
        contours = db.query(Contours).filter_by(mask_id=mask.id)
        if exclude_unreviewed_objects:
            # Excludes contours that have no reviewer
            contours = contours.filter(Contours.reviewed_by.any())
        contour_hierarchy = ContourHierarchy.from_query(contours.all(),
                                                        height=image.height,
                                                        width=image.width)
        label_quants = contour_hierarchy.get_all_quantifications()
        for label, quants in label_quants.items():
            metrics = quants["metrics"]
            child_counts = quants["child_counts"]
            for k, v in metrics.items():
                metrics_per_label[label][k].extend(v)
            for k, v in child_counts.items():
                child_counts_per_label[label][k].extend(v)
    labels = db.query(Labels).filter_by(dataset_id=dataset_id)
    labels_hierarchy = LabelHierarchy.from_query(labels)
    return {
        "success": True,
        "message": "Successfully exported the object quantifications of this dataset as json.",
        "labels": labels_hierarchy.model_dump(),
        "metrics_per_label_id": metrics_per_label,
        "child_counts_per_label_id": child_counts_per_label,
    }


@router.get("/{dataset_id}/ml_dataset")
async def get_segmentation_dataset(
        dataset_id: int,
        include_label_ids: list[int] = None,
        exclude: list[Literal["unreviewed", "not_fully_annotated"]] = ["unreviewed", "not_fully_annotated"],
        as_download: bool = False,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """
    Download all images and masks as a ZIP file for the given dataset_id.

    Args:
        dataset_id (int): The ID of the dataset.
        exclude_unreviewed_annotations (bool): Whether to exclude unreviewed annotations.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        StreamingResponse: A ZIP file containing all images and masks.
    """
    dataset = db.query(Datasets).filter_by(id=dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    query = db.query(Images.file_path).join(Masks).filter(Images.dataset_id == dataset_id)

    if "not_fully_annotated" in exclude:
        query = query.filter(Masks.fully_annotated == True)
    if "unreviewed" in exclude:
        query = query.filter(Masks.fully_annotated == True, Masks.contours.any(Contours.reviewed_by.any())).all()
    masks = query.all()
    if not masks:
        raise HTTPException(status_code=404, detail="No finished masks found for this dataset.")

    zip_filename = f"{dataset.name.replace(' ', '_')}.zip"

    # Create a streaming response with a ZIP file
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for mask in masks:
            image = db.query(Images).filter_by(file_path=mask.file_path).first()
            if not image:
                continue

            mask_file_path = get_mask_path_from_image_path(image.file_path)
            if not os.path.exists(mask_file_path):
                logger.error(f"Mask file not found at {mask_file_path}.")
                continue

            zipf.write(mask_file_path, os.path.join("masks", os.path.basename(mask_file_path)))
            zipf.write(image.file_path, os.path.join("images", os.path.basename(image.file_path)))

    # Seek to the start of the buffer
    buffer.seek(0)

    # Create a streaming response
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )

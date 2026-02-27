import io
import os
import zipfile
from collections import defaultdict
from io import StringIO
from logging import getLogger
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from iquana_toolbox.schemas.contour_hierarchy import ContourHierarchy
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
from app.services.auth import get_current_user
from app.services.database_access import datasets as datasets_db
from app.services.database_access import labels as labels_db
from app.services.util import get_mask_path_from_image_path

# Create a router for the export functionality
router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = getLogger(__name__)


@router.post("/create")
async def create_dataset(name: str,
                         description: str,
                         dataset_type: Literal["image", "scan", "DICOM"],
                         current_user=Depends(get_current_user)):
    """Create a new dataset.

    Args:
        name (str): The name of the dataset.
        description (str): A brief description of the dataset.
        dataset_type (Literal["image", "scan", "DICOM"]): The type of dataset.
        current_user (Users): Auth bearer token.

    Returns:
        dict: A dictionary containing the success status and message, or error details.
    """

    return {"success": True,
            "message": "Dataset created successfully.",
            "dataset_id": await datasets_db.create_new_dataset(
                name=name,
                description=description,
                owner_username=current_user.username
            )
            }


@router.post("/{dataset_id}/share")
async def share_dataset(
        dataset_id: int,
        share_with_username: str,
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
    if not await datasets_db.user_has_sharing_permission_for_dataset(dataset_id, user.username):
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                             detail="User does not have permission to share this dataset.")
    await datasets_db.share_dataset(
        dataset_id,
        share_with_username,
        sharing_username=user.username
    )
    return {"success": True, "message": f"Dataset shared with {share_with_username}"}


@router.get("/all")
async def get_all_datasets(
        user: "User" = Depends(get_current_user)
):
    """Get all datasets owned by or shared with the current user.

    Args:
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and the list of datasets.
    """
    datasets = await datasets_db.get_datasets_of_user(user.username)
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
    dataset = await datasets_db.get_dataset(dataset_id)
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

    return {
        "success": True,
        "number_of_images": await datasets_db.get_num_of_images_in_dataset(dataset_id)
    }


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
    status_dict, num_masks = await datasets_db.get_annotation_progress_of_dataset(dataset_id)
    return {
        "success": True,
        "message": "Annotation progress retrieved successfully.",
        "total_images": num_masks,
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
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User cannot delete dataset.")
    await datasets_db.delete_dataset(dataset_id)
    return {"success": True, "message": "Dataset deleted successfully."}


@router.get("/{dataset_id}/images")
async def list_images(
        dataset_id: int,
        filter_for_status: Literal["not_started", "in_progress", "reviewable", "finished"] | None = None,
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
    image_data = await datasets_db.get_image_and_mask_ids_of_dataset(
        dataset_id,
        filter_for_status
    )
    return {
        "success": True,
        "message": "Retrieved images successfully.",
        "image_data": image_data
    }


@router.get("/{dataset_id}/images/b64")
async def get_base64_images_of_dataset(
        dataset_id: int,
        limit: int = None,
        user: User = Depends(get_current_user)
):
    """Get all images of a dataset.

    Args:
        dataset_id: ID of the dataset to retrieve images from.
        limit: Optional limit on the number of images to return. If not provided, all images will be returned.
        db: Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A dict mapping from image ID to base64 encoded image.
    """
    response = await datasets_db.get_images_of_dataset(
        dataset_id,
        limit,
        as_thumbnail=False,
        as_base64=True
    )
    return {
        "success": True,
        "message": f"Successfully retrieved {len(response)} images from dataset {dataset_id}.",
        "images": response
    }


@router.get("/{dataset_id}/thumbnails/b64")
async def get_base64_thumbnails_of_dataset(
        dataset_id: int,
        limit: int = None,
        user: User = Depends(get_current_user)
):
    """Get all images of a dataset.

    Args:
        dataset_id: ID of the dataset to retrieve images from.
        limit: Optional limit on the number of images to return. If not provided, all images will be returned.
        db: Database session dependency.
        user (User): The current authenticated user.

    Returns:
        A dict mapping from image ID to base64 encoded image.
    """
    response = await datasets_db.get_images_of_dataset(
        dataset_id,
        limit,
        as_thumbnail=True,
        as_base64=True
    )
    return {
        "success": True,
        "message": f"Successfully retrieved {len(response)} images from dataset {dataset_id}.",
        "images": response
    }


@router.get("/{dataset_id}/labels")
async def get_labels(
        dataset_id: int,
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
    labels_hierarchy = await labels_db.get_label_hierarchy(dataset_id)
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
    df = await datasets_db.get_dataset_as_df(dataset_id, exclude_not_fully_annotated, exclude_unreviewed, db)
    if df.empty:
        return {
            "success": False,
            "message": "No data found for the given dataset and filters.",
            "data": None
        }
    else:
        return {
            "success": True,
            "message": "Successfully exported the dataset as json.",
            "data": df.to_json(orient="records"),
        }


@router.get(
    "/{dataset_id}/quantification/download")
async def download_dataset_quantification(
        dataset_id: int,
        exclude_unreviewed: bool = True,
        exclude_not_fully_annotated: bool = True,
        file_format: Literal["json", "csv"] = "json",
        user: User = Depends(get_current_user)
):
    """
    Export quantification data for the given dataset_id and labels.

    Args:
        dataset_id (int): The ID of the dataset to export.
        exclude_not_fully_annotated (bool): Whether to exclude not fully annotated masks.
        exclude_unreviewed (bool): Whether to exclude unreviewed contours.
        file_format (Literal["json", "csv"]): File format to export to.
        db (Session, optional): The database session. Defaults to Depends(get_session). This is a fastapi dependency.
        user (User): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and message if error, or a
        StreamingResponse with the CSV file.
    """
    if dataset_id not in user.available_datasets:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access to this dataset.")

    dataset_name = (await datasets_db.get_dataset(dataset_id)).name
    df = await datasets_db.get_dataset_as_df(dataset_id, exclude_not_fully_annotated, exclude_unreviewed, db)
    if df.empty:
        return {
            "success": False,
            "message": "No data found for the given dataset and filters."
        }
    else:
        file_data = None
        match file_format:
            case "json":
                file_data = df.to_json(orient="records")
            case "csv":
                file_data = StringIO(df.to_csv(index=False))
            case _:
                raise ValueError(f"Invalid file format: {file_format}")
        response = StreamingResponse(file_data, media_type=f"text/{file_format}")
        response.headers[
            "Content-Disposition"] = f'attachment; filename="{dataset_name.replace(' ', '_')}_dataset.{file_format}"'
        return response


@router.get("/{dataset_id}/ml_dataset", deprecated=True)
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

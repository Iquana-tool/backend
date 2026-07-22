import io
import os
import zipfile
from io import StringIO
from logging import getLogger
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from starlette import status
from starlette.responses import JSONResponse, StreamingResponse

from app.database import get_session
from app.database.images import Images
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import DatasetRole, Permission
from app.services.auth import get_current_user
from app.services.database_access import datasets as datasets_db
from app.services.database_access import labels as labels_db
from app.services.database_access import members as members_db
from app.services.database_access.datasets import ContourSelection, export_dataset_contours_to_coco
from app.services.permissions import ensure_permission, require, require_global

# Create a router for the export functionality
router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = getLogger(__name__)


@router.post("/create")
async def create_dataset(name: str,
                         description: str,
                         dataset_type: Literal["image", "scan", "DICOM"],
                         db: Session = Depends(get_session),
                         current_user: AuthenticatedUser = Depends(
                             require_global(Permission.DATASET_CREATE))):
    """Create a new dataset. The creator becomes its owner.

    Args:
        name (str): The name of the dataset.
        description (str): A brief description of the dataset.
        dataset_type (Literal["image", "scan", "DICOM"]): The type of dataset.
        current_user (AuthenticatedUser): Caller, who must be allowed to create datasets.

    Returns:
        dict: A dictionary containing the success status and message, or error details.
    """
    dataset = await datasets_db.create_new_dataset(
        name=name,
        description=description,
        owner_username=current_user.username,
        db=db
    )
    return {"success": True,
            "message": "Dataset created successfully.",
            "dataset_id": dataset
            }


@router.post("/{dataset_id}/share")
async def share_dataset(
        dataset_id: int,
        share_with_username: str,
        role: str = "curator",
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.MEMBER_GRANT))
):
    """Share a dataset with another user by username.

    Kept for backwards compatibility; it now grants a role rather than blanket
    access. `PUT /datasets/{id}/members` is the fuller version, with per-member
    permission overrides.

    Args:
        dataset_id (int): The ID of the dataset to share.
        share_with_username (str): The username to share with.
        role (str): Dataset role to grant. Defaults to curator, matching the
            unrestricted access that sharing used to imply.
        db (Session): The database session.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    try:
        dataset_role = DatasetRole(role)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"Unknown role '{role}'. One of: "
                                   f"{', '.join(r.value for r in DatasetRole)}.")
    members_db.grant_role(dataset_id, share_with_username, dataset_role,
                          granted_by=user.username, db=db)
    return {"success": True,
            "message": f"Dataset shared with {share_with_username} as {dataset_role.value}."}


@router.get("/all")
async def get_all_datasets(
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(get_current_user)
):
    """Get all datasets the current user has any role on.

    Args:
        db (Session): The database session.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and the list of datasets.
    """
    datasets = await datasets_db.get_datasets_of_user(user, db=db)
    return {"success": True, "datasets": [
        {
            "id": ds.id,
            "name": ds.name,
            "description": ds.description,
            "dataset_type": ds.dataset_type,
            "folder_path": ds.folder_path,
            "created_by": ds.created_by,
            "shared_with": [u.username for u in ds.shared_with],
            # What *this* caller may do, so the UI can hide actions it would reject.
            "my_role": user.role_for(ds.id).value if user.role_for(ds.id) else None,
            "my_permissions": sorted(p.value for p in user.permissions_for(ds.id)),
        }
        for ds in datasets
    ]}


@router.get("/{dataset_id}")
async def get_dataset(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.DATASET_READ))
):
    """Get dataset information.

    Args:
        dataset_id (int): The ID of the dataset.
        db (Session): The database session.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and dataset information.
    """
    dataset = await datasets_db.get_dataset(dataset_id, db=db)
    return {
        "success": True,
        "message": "Dataset found.",
        "dataset": dataset,
        "my_role": user.role_for(dataset_id).value if user.role_for(dataset_id) else None,
        "my_permissions": sorted(p.value for p in user.permissions_for(dataset_id)),
    }


@router.patch("/{dataset_id}/settings")
async def update_dataset_settings(
        dataset_id: int,
        require_independent_review: bool | None = None,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.DATASET_UPDATE))
):
    """Update per-dataset review policy.

    With `require_independent_review` on, a contour cannot be approved by whoever
    created it, so `finished` means a second pair of eyes actually saw the work.
    Off by default, because a single owner annotating their own dataset would
    otherwise never be able to finish it.
    """
    dataset = await datasets_db.get_dataset(dataset_id, db=db)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
    if require_independent_review is not None:
        dataset.require_independent_review = require_independent_review
        db.commit()
    return {
        "success": True,
        "message": "Dataset settings updated.",
        "require_independent_review": dataset.require_independent_review,
    }


@router.get("/{dataset_id}/images/count")
async def get_number_of_images(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.DATASET_READ))
):
    """Get the number of images in a dataset.

    Args:
        dataset_id (int): The ID of the dataset.
        db (Session): The database session.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        dict: A dictionary containing the number of images.
    """

    return {
        "success": True,
        "number_of_images": await datasets_db.get_num_of_images_in_dataset(dataset_id, db=db)
    }


@router.get("/{dataset_id}/progress")
async def get_annotation_progress(dataset_id: int,
                                  user: AuthenticatedUser = Depends(require(Permission.DATASET_READ)),
                                  db: Session = Depends(get_session)):
    """Get the annotation progress of a dataset.

    Args:
        dataset_id (int): The ID of the dataset to check.
        user (AuthenticatedUser): The current authenticated user.
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
    status_dict, num_masks = await datasets_db.get_annotation_progress_of_dataset(dataset_id, db=db, )
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
        user: AuthenticatedUser = Depends(require(Permission.DATASET_DELETE))
):
    """Delete a dataset.

    Args:
        dataset_id (int): The ID of the dataset to delete.
        db (Session): The database session.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and message.
    """
    await datasets_db.delete_dataset(dataset_id, db=db, )
    return {"success": True, "message": "Dataset deleted successfully."}


@router.get("/{dataset_id}/images")
async def list_images(
        dataset_id: int,
        filter_for_status: Literal["not_started", "in_progress", "rejected", "reviewable", "finished"] | None = None,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ))
):
    """List all images with masks of certain status for a given image ID.

    Args:
        dataset_id: Dataset ID to retrieve images from.
        filter_for_status: The status of the masks to filter by.
        db: Database session dependency.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        A list of image IDs.
    """
    image_data = await datasets_db.get_image_and_mask_ids_of_dataset(
        dataset_id,
        filter_for_status=filter_for_status,
        db=db,
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
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_READ))
):
    """Get all images of a dataset.

    Args:
        dataset_id: ID of the dataset to retrieve images from.
        limit: Optional limit on the number of images to return. If not provided, all images will be returned.
        db: Database session dependency.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        A dict mapping from image ID to base64 encoded image.
    """
    response = await datasets_db.get_images_of_dataset(
        dataset_id,
        limit=limit,
        db=db,
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
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.IMAGE_READ))
):
    """Get all images of a dataset.

    Args:
        dataset_id: ID of the dataset to retrieve images from.
        limit: Optional limit on the number of images to return. If not provided, all images will be returned.
        db: Database session dependency.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        A dict mapping from image ID to base64 encoded image.
    """
    response = await datasets_db.get_images_of_dataset(
        dataset_id,
        db=db,
        limit=limit,
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
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.LABEL_READ))
):
    """Retrieve all labels for a given dataset.

    Args:
        dataset_id (int): The ID of the dataset.
        db (Session): The database session.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and the labels hierarchy.
    """
    labels_hierarchy = await labels_db.get_label_hierarchy(dataset_id, db=db, )
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
        user: AuthenticatedUser = Depends(require(Permission.EXPORT_QUANTIFICATION))
):
    """
    Export quantification data for the given dataset_id and labels.

    Args:
        dataset_id (int): The ID of the dataset to export.
        exclude_not_fully_annotated (bool): Whether to exclude not fully annotated masks.
        exclude_unreviewed (bool): Whether to exclude unreviewed contours.
        as_download (bool, optional): Whether to export as CSV. Defaults to False. If False, returns the data as a json.
        db (Session, optional): The database session. Defaults to Depends(get_session). This is a fastapi dependency.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and message if error, or a
        StreamingResponse with the CSV file.
    """
    df: pd.DataFrame = await datasets_db.get_dataset_as_df(dataset_id, exclude_not_fully_annotated, exclude_unreviewed, db)
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
            "data": df.to_json(orient="records", default_handler=str),
        }


@router.get(
    "/{dataset_id}/quantification/download")
async def download_dataset_quantification(
        dataset_id: int,
        exclude_unreviewed: bool = True,
        exclude_not_fully_annotated: bool = True,
        file_format: Literal["json", "csv"] = "json",
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.EXPORT_QUANTIFICATION))
):
    """
    Export quantification data for the given dataset_id and labels.

    Args:
        dataset_id (int): The ID of the dataset to export.
        exclude_not_fully_annotated (bool): Whether to exclude not fully annotated masks.
        exclude_unreviewed (bool): Whether to exclude unreviewed contours.
        file_format (Literal["json", "csv"]): File format to export to.
        db (Session, optional): The database session. Defaults to Depends(get_session). This is a fastapi dependency.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        dict: A dictionary containing the success status and message if error, or a
        StreamingResponse with the CSV file.
    """

    dataset_name = (await datasets_db.get_dataset(dataset_id, db=db, )).name
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
                file_data = df.to_json(orient="records", default_handler=str)
            case "csv":
                file_data = StringIO(df.to_csv(index=False))
            case _:
                raise ValueError(f"Invalid file format: {file_format}")
        response = StreamingResponse(file_data, media_type=f"text/{file_format}")
        response.headers[
            "Content-Disposition"] = f'attachment; filename="{dataset_name.replace(' ', '_')}_dataset.{file_format}"'
        return response


@router.get("/{dataset_id}/coco/annotations")
async def get_coco_annotations(
        dataset_id: int,
        exclude_not_fully_annotated: bool = True,
        exclude_unreviewed: bool = True,
        contour_selection: ContourSelection = "all",
        log_to_mlflow: bool = False,
        mlflow_run_id: str | None = None,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.EXPORT_ANNOTATIONS))
):
    """
    Return the dataset annotations as a COCO JSON document, without any images.

    Shares the COCO-building logic with the ZIP export, so the annotations are
    identical to what `GET /{dataset_id}/coco` bundles.
    Args:
        dataset_id (int): The ID of the dataset to export.
        exclude_not_fully_annotated (bool): Whether to exclude not fully annotated masks.
        exclude_unreviewed (bool): Whether to exclude unreviewed contours.
        contour_selection ("all" | "leaves" | "top_level"): Which contours of the
            annotation hierarchy to emit. "all" keeps every contour (parents overlap
            their children), "leaves" keeps only the innermost contours, "top_level"
            keeps only contours without a parent.
        log_to_mlflow (bool): Whether to log the export to MLflow.
        mlflow_run_id (str | None): The MLflow run ID.
        db (Session, optional): The database session. Defaults to Depends(get_session).
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        JSONResponse: The COCO annotations document.
    """
    # Check user access to the dataset

    dataset = await datasets_db.get_dataset(dataset_id, db=db)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")

    result = await export_dataset_contours_to_coco(
        dataset_id,
        db,
        exclude_not_fully_annotated,
        exclude_unreviewed,
        contour_selection=contour_selection,
        write_to_disk=False,
        log_to_mlflow=log_to_mlflow,
        mlflow_run_id=mlflow_run_id,
    )

    if not result.get("success"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("message"))

    file_name = f"{dataset.name.replace(' ', '_')}_coco.json"
    return JSONResponse(
        content=result["coco_payload"],
        headers={"Content-Disposition": f"attachment; filename={file_name}"},
    )


@router.get("/{dataset_id}/coco")
async def get_coco_dataset(
        dataset_id: int,
        exclude_not_fully_annotated: bool = True,
        exclude_unreviewed: bool = True,
        contour_selection: ContourSelection = "all",
        include_images: bool = True,
        log_to_mlflow: bool = False,
        mlflow_run_id: str | None = None,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.EXPORT_ANNOTATIONS))
):
    """
    Download the dataset in COCO format as a ZIP file. The ZIP file will contain a JSON file with the annotations and
    optionally the images.
    Args:
        dataset_id (int): The ID of the dataset to download.
        exclude_not_fully_annotated (bool): Whether to exclude not fully annotated masks.
        exclude_unreviewed (bool): Whether to exclude unreviewed contours.
        contour_selection ("all" | "leaves" | "top_level"): Which contours of the
            annotation hierarchy to emit. "all" keeps every contour (parents overlap
            their children), "leaves" keeps only the innermost contours, "top_level"
            keeps only contours without a parent.
        include_images (bool): Whether to include images in the dataset. Bundling the
            raw imagery needs `export.images` on top of `export.annotations`, so
            collaborators can be given the annotations without the pixels.
        log_to_mlflow (bool): Whether to log the dataset to MLflow.
        mlflow_run_id (str | None): The MLflow run ID.
        db (Session, optional): The database session. Defaults to Depends(get_session).
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        StreamingResponse: A StreamingResponse object containing the dataset as a zip file.
    """
    if include_images:
        ensure_permission(user, dataset_id, Permission.EXPORT_IMAGES)

    dataset = await datasets_db.get_dataset(dataset_id, db=db)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")

    # Export contours to COCO format
    result = await export_dataset_contours_to_coco(
        dataset_id,
        db,
        exclude_not_fully_annotated,
        exclude_unreviewed,
        contour_selection=contour_selection,
        log_to_mlflow=log_to_mlflow,
        mlflow_run_id=mlflow_run_id,
    )

    if not result.get("success"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result.get("message"))

    coco_json_path = result["output_file_path"]

    # Create ZIP file with COCO JSON and optionally images
    zip_filename = f"{dataset.name.replace(' ', '_')}_coco.zip"
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add the COCO JSON file
        zipf.write(coco_json_path, arcname=os.path.basename(coco_json_path))

        # Add only the images referenced by the COCO JSON, so the bundle stays in sync
        # with the annotations (same filters, no duplicates).
        if include_images and result["image_ids"]:
            images = db.query(Images).filter(Images.id.in_(list(result["image_ids"]))).all()

            for image in images:
                if os.path.exists(image.file_path):
                    zipf.write(image.file_path, arcname=os.path.join("images", os.path.basename(image.file_path)))
                else:
                    logger.warning(f"Image file not found at {image.file_path}.")

    # Seek to the start of the buffer
    buffer.seek(0)

    # Create and return streaming response
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )


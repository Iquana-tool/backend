import io
import os
import zipfile
from logging import getLogger
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from iquana_toolbox.quantification import list_metrics
from iquana_toolbox.schemas.database.quantification_profile import QuantificationProfile
from iquana_toolbox.schemas.user import User
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette import status
from starlette.responses import JSONResponse, Response, StreamingResponse

from app.database import get_session
from app.database.images import Images
from app.exceptions import InvalidMetadataError
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.permissions import DatasetRole, Permission
from app.services.auth import get_current_user
from app.services.database_access import datasets as datasets_db
from app.services.database_access import image_metadata as metadata_db
from app.services.database_access import labels as labels_db
from app.services.database_access import members as members_db
from app.services.database_access import quantification_profiles as profiles_db
from app.services.database_access.datasets import ContourSelection, export_dataset_contours_to_coco
from app.services.quantification import (
    APPEARANCE_METRIC_KEYS,
    CONTEXTUAL_METRIC_KEYS,
    GEOMETRY_METRIC_KEYS,
    RELATIONAL_METRIC_KEYS,
    compute_appearance_metrics_for_dataset,
    compute_contextual_metrics_for_dataset,
    compute_geometry_metrics_for_dataset,
    compute_relational_metrics_for_dataset,
)
from app.services.util import get_mask_path_from_image_path
from app.services.permissions import ensure_permission, require, require_global

# Create a router for the export functionality
router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = getLogger(__name__)


class ProfileEntryBody(BaseModel):
    """Request body for a single metric selection in a profile (see ProfileEntry)."""
    metric_key: str = Field(..., description="Registry key of the metric.")
    params: dict = Field(default_factory=dict, description="Per-metric parameter dict.")
    label_ids: list[int] | None = Field(default=None, description="Label ids the metric is "
                                                                  "scoped to; null means all labels.")


class ProfileBody(BaseModel):
    """Request body for creating/updating a quantification profile."""
    name: str = Field(..., description="Human-readable profile name.")
    is_default: bool = Field(default=False, description="Mark this profile as the dataset default.")
    entries: list[ProfileEntryBody] = Field(default_factory=list,
                                            description="Ordered list of metric selections.")


def _assert_image_in_dataset(db: Session, dataset_id: int, image_id: int | None) -> None:
    """Refuse an ``image_id`` that is not part of ``dataset_id``.

    The quantification reads scope by ``Images.dataset_id`` anyway, so a foreign id would
    quietly aggregate nothing rather than fail - an empty per-image page that looks like an
    image with no objects. Checked up front so the answer is a 404 instead.

    :raises HTTPException: 404 if the image does not exist in this dataset.
    """
    if image_id is None:
        return
    exists = (
        db.query(Images.id)
        .filter(Images.id == image_id, Images.dataset_id == dataset_id)
        .first()
    )
    if exists is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} is not part of dataset {dataset_id}.",
        )


def _resolve_profile_scoping(
        db: Session,
        dataset_id: int,
        profile_id: int | None,
) -> tuple[dict[str, list[int] | None] | None, set[str]]:
    """Resolve a profile id into (metric_scoping, tiers_to_compute).

    Returns ``(None, {"geometry", "appearance", "contextual", "relational"})`` when ``profile_id`` is
    None so the legacy no-profile path aggregates every metric and lazily computes all
    tiers, exactly as before. When a profile is given, ``metric_scoping`` maps
    each of the profile's metric keys to its label scope (last entry wins if a key repeats),
    and the tier set contains only the tiers the profile actually references, so e.g. a
    geometry-only profile never pays the appearance image-decode or relational compute cost.

    :raises HTTPException: 404 if ``profile_id`` does not belong to the dataset.
    """
    if profile_id is None:
        return None, {"appearance", "contextual", "relational"}

    row = profiles_db.get_profile(db, dataset_id, profile_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found.")

    profile = row.to_schema()
    metric_scoping: dict[str, list[int] | None] = {}
    for entry in profile.entries:
        metric_scoping[entry.metric_key] = entry.label_ids

    tiers: set[str] = set()
    if any(key in GEOMETRY_METRIC_KEYS for key in metric_scoping):
        tiers.add("geometry")
    if any(key in APPEARANCE_METRIC_KEYS for key in metric_scoping):
        tiers.add("appearance")
    if any(key in CONTEXTUAL_METRIC_KEYS for key in metric_scoping):
        tiers.add("contextual")
    if any(key in RELATIONAL_METRIC_KEYS for key in metric_scoping):
        tiers.add("relational")
    return metric_scoping, tiers


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
    if isinstance(dataset, dict):
        return dataset

    return {"success": True,
            "message": "Dataset created successfully.",
            "dataset_id": dataset.id
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
    """Get the per-phase progress of a dataset.

    Args:
        dataset_id (int): The ID of the dataset to check.
        user (AuthenticatedUser): The current authenticated user.
        db (Session): The database session.

    Returns:
        dict: A dictionary containing the progress details. The dict contains:
            - success (bool): Indicates if the operation was successful.
            - message (str): A message indicating the result of the operation.
            - phases (dict): ``{phase: {state: count}}`` for ``calibrate``,
              ``annotate`` and ``review``, each state being one of ``not_started``,
              ``in_progress`` or ``finished`` — plus ``blocked`` on ``review``,
              which counts images with nothing drawn to review. This is what the
              three progress bars are drawn from.
            - overall (dict): The same three counts for the combined status, which
              is ``finished`` only when all three phases are.
            - total_images (int): Total number of images in the dataset.
    """
    counts, total_images = await datasets_db.get_annotation_progress_of_dataset(dataset_id, db=db)
    overall = counts.pop("overall")
    return {
        "success": True,
        "message": "Annotation progress retrieved successfully.",
        "total_images": total_images,
        "phases": counts,
        "overall": overall,
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
        filter_for_status: Literal["blocked", "not_started", "in_progress", "finished"] | None = None,
        filter_for_phase: Literal["calibrate", "annotate", "review"] | None = None,
        db: Session = Depends(get_session),
        user: AuthenticatedUser = Depends(require(Permission.ANNOTATION_READ))
):
    """List a dataset's images with their workflow status.

    Args:
        dataset_id: Dataset ID to retrieve images from.
        filter_for_status: Keep only images in this state.
        filter_for_phase: Which phase ``filter_for_status`` applies to
            (``calibrate`` / ``annotate`` / ``review``). Omit it to filter on the
            overall status.
        db: Database session dependency.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        ``image_data``: one ``{image_id, file_name, mask_id, status, phases,
        metadata}`` entry per image, where ``metadata`` is the image's grouping
        key/values (see ``GET /metadata/dataset/{id}`` for the dataset's whole
        metadata vocabulary).
    """
    image_data = await datasets_db.get_image_and_mask_ids_of_dataset(
        dataset_id,
        filter_for_status=filter_for_status,
        filter_for_phase=filter_for_phase,
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


@router.get("/{dataset_id}/quantification/summary")
async def get_dataset_quantification_summary(
        dataset_id: int,
        exclude_unreviewed: bool = True,
        exclude_not_fully_annotated: bool = True,
        include_appearance: bool = True,
        include_contextual: bool = True,
        include_relational: bool = True,
        include_distribution: bool = False,
        profile_id: int | None = None,
        group_by: str | None = None,
        image_id: int | None = None,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """Server-side aggregated quantification summary for a dataset.

    Aggregates the tall ``contour_metrics`` table per label / metric / component / unit
    into count / mean / std / min / max (no per-contour rows are shipped to the client),
    plus the parent->child label counts the quantification page shows. This is the Step-2
    endpoint that the frontend migrates to in Step 5; the legacy ``/quantification``
    endpoints are left untouched.

    Perf tradeoff (Step 3): appearance metrics (mean color / intensity) need the image
    pixels decoded, which never happens on the contour write path. When
    ``include_appearance`` is True (default), this endpoint calls
    :func:`compute_appearance_metrics_for_dataset` with ``only_stale=True`` BEFORE
    aggregating, so new/edited contours get their color computed on demand and the
    summary always reflects it without a separate manual step. This means the first
    summary request after bulk edits/imports pays the one-time image-decode cost for
    every touched image (subsequent requests are cheap: only_stale skips anything
    already computed and fresh). Pass ``include_appearance=False`` to skip this and get
    the fast, aggregation-only path (e.g. for polling loops that don't render color).

    Perf tradeoff (Step 4): contextual metrics (nearest-neighbour distance) are
    RELATIONAL, so ``include_contextual=True`` (default) calls
    :func:`compute_contextual_metrics_for_dataset` with ``only_stale=True`` first, same
    idea as appearance - the first request after edits/imports pays the cost of
    rebuilding the affected parent groups' KDTrees, subsequent requests are cheap.

    Relational metrics (``n_children``) are computed the same lazy way via
    ``include_relational=True`` (default): missing/stale rows are recomputed by
    :func:`compute_relational_metrics_for_dataset` before aggregating.

    Distribution (opt-in): with ``include_distribution=True`` the response also carries a
    ``distribution`` object mirroring ``metrics`` but with box/violin stats for each
    value_dim-1 numeric metric (``str(label_id) -> metric_key -> {"0": {count, mean, std,
    min, max, q1, median, q3, whisker_low, whisker_high, outliers, outlier_count, kde,
    histogram}}``). This is heavier than the pure-SQL summary (it fetches stored per-contour
    values and reduces them with numpy - SQLite has no percentile function), so it is off by
    default and only computed for the plotted numeric metrics; COLOR metrics are excluded.
    The bar view never needs it, so the frontend only requests it for box/violin.

    Args:
        dataset_id (int): The ID of the dataset to summarize.
        exclude_unreviewed (bool): Whether to exclude unreviewed contours.
        exclude_not_fully_annotated (bool): Whether to exclude not fully annotated masks.
        include_appearance (bool): Whether to lazily compute missing/stale appearance
            metrics (mean color / intensity) before aggregating. Defaults to True.
        include_contextual (bool): Whether to lazily compute missing/stale contextual
            metrics (nearest-neighbour distance) before aggregating. Defaults to True.
        include_relational (bool): Whether to lazily compute missing/stale relational
            metrics (number of children) before aggregating. Defaults to True.
        include_distribution (bool): Whether to also compute and return per-(label, metric)
            box/violin distribution stats for the numeric metrics. Defaults to False.
        group_by (str | None): An image-metadata key to additionally break the results
            down by, so the page can compare the same label across sites / treatments
            rather than only across labels. Metadata is effectively an image-wide label
            that every object on the image inherits, which is why this is a grouping key
            on the existing aggregation rather than a new kind of metric.

            Only *groupable* key types are accepted (category, yes/no); a number or a
            date is near-unique per image and would draw one band per image, so it is
            refused with a 422 rather than rendered. See ``GET /metadata/types``.
        image_id (int | None): Narrow every aggregation to a single image of the dataset -
            the per-image inspection view. The response keeps exactly the same shape, so
            the client renders one image's numbers with the components it already has, and
            gets the dataset baseline to compare against from a second, unscoped call.

            Note ``scale_status`` then describes that one image: an image with a scale
            reports physical units even inside a dataset whose images disagree, because
            "every contributing image shares one unit" is trivially true of one image.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: ``{success, metrics, child_counts_per_label_id, object_counts_per_label_id,
        scale_status, labels}`` where ``metrics`` maps ``str(label_id) -> metric_key ->
        {unit, components: [{count, mean, std, min, max}, ...]}`` and ``labels`` is the label
        hierarchy dump (as the labels endpoint). ``scale_status`` reports whether the
        dataset's images share one scale unit and which unit the numbers are in (pixels when
        the scales are mixed), so the client can warn when quantifications fall back to
        pixels. When ``include_distribution`` is set, a ``distribution`` key is added.

        With ``group_by``, three more keys appear: ``groups`` (``group_value ->`` the same
        shape as ``metrics``), ``group_by`` and the display-ordered ``group_values``.
        ``metrics`` itself stays dataset-wide and unchanged, so a client that ignores the
        grouping sees exactly what it saw before. A ``distribution`` requested alongside
        ``group_by`` gains the same extra level.
    """
    if dataset_id not in user.available_datasets:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access to this dataset.")

    _assert_image_in_dataset(db, dataset_id, image_id)

    # A profile (if given) both restricts which metrics are aggregated and which tiers we
    # bother lazily computing, so a geometry-only profile skips the appearance/contextual/
    # relational compute cost entirely.
    metric_scoping, profile_tiers = _resolve_profile_scoping(db, dataset_id, profile_id)

    # Refuse an ungroupable key before doing any of the expensive lazy compute below.
    group_by_key = None
    if group_by:
        try:
            group_by_key = metadata_db.assert_groupable(db, dataset_id, group_by)
        except InvalidMetadataError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail=str(exc))

    if "geometry" in profile_tiers:
        compute_geometry_metrics_for_dataset(db, dataset_id, only_stale=True)
    if include_appearance and "appearance" in profile_tiers:
        compute_appearance_metrics_for_dataset(db, dataset_id, only_stale=True)
    if include_contextual and "contextual" in profile_tiers:
        compute_contextual_metrics_for_dataset(db, dataset_id, only_stale=True)
    if include_relational and "relational" in profile_tiers:
        compute_relational_metrics_for_dataset(db, dataset_id, only_stale=True)

    summary = await datasets_db.get_quantification_summary(
        dataset_id, exclude_not_fully_annotated, exclude_unreviewed, db,
        metric_scoping=metric_scoping, group_by_key=group_by_key, image_id=image_id,
    )
    labels_hierarchy = await labels_db.get_label_hierarchy(dataset_id, db=db)
    response = {
        "success": True,
        "message": "Successfully aggregated the dataset quantification.",
        "metrics": summary["metrics"],
        "child_counts_per_label_id": summary["child_counts_per_label_id"],
        "object_counts_per_label_id": summary["object_counts_per_label_id"],
        "scale_status": summary["scale_status"],
        "labels": labels_hierarchy.model_dump(),
    }
    if group_by_key:
        response["groups"] = summary["groups"]
        response["group_by"] = summary["group_by"]
        response["group_values"] = summary["group_values"]
    if include_distribution:
        response["distribution"] = await datasets_db.get_quantification_distribution(
            dataset_id, exclude_not_fully_annotated, exclude_unreviewed, db,
            metric_scoping=metric_scoping, group_by_key=group_by_key, image_id=image_id,
        )
    return response


@router.post("/{dataset_id}/quantification/appearance/recompute")
async def recompute_dataset_appearance_metrics(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """Explicitly (re)compute appearance-tier metrics (mean color / intensity) for a dataset.

    Runs :func:`compute_appearance_metrics_for_dataset` with ``only_stale=True``: only
    contours with no fresh row yet (new contours) or a ``stale=True`` row (geometry
    changed since the last compute, see ``mark_appearance_stale``) are recomputed. Useful
    to pre-warm the cache (e.g. right after a bulk import) rather than paying the decode
    cost lazily on the next ``GET .../summary`` call.

    Args:
        dataset_id (int): The ID of the dataset to recompute appearance metrics for.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: ``{success, computed_rows, message}``.
    """
    if dataset_id not in user.available_datasets:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access to this dataset.")

    computed_rows = compute_appearance_metrics_for_dataset(db, dataset_id, only_stale=True)
    return {
        "success": True,
        "computed_rows": computed_rows,
        "message": f"Computed {computed_rows} appearance metric rows.",
    }


@router.post("/{dataset_id}/quantification/contextual/recompute")
async def recompute_dataset_contextual_metrics(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """Explicitly (re)compute contextual-tier metrics (nearest-neighbour distance) for a dataset.

    Runs :func:`compute_contextual_metrics_for_dataset` with ``only_stale=True``: parent
    groups with at least one contour missing a fresh row (new contour, or a row marked
    stale by ``mark_contextual_stale_for_group`` after a move/re-parent/delete) are fully
    recomputed - see that function's docstring for why the whole group, not just the
    changed contour, is recomputed. Useful to pre-warm the cache (e.g. right after a bulk
    import) rather than paying the cost lazily on the next ``GET .../summary`` call.

    Args:
        dataset_id (int): The ID of the dataset to recompute contextual metrics for.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: ``{success, computed_rows, message}``.
    """
    if dataset_id not in user.available_datasets:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access to this dataset.")

    computed_rows = compute_contextual_metrics_for_dataset(db, dataset_id, only_stale=True)
    return {
        "success": True,
        "computed_rows": computed_rows,
        "message": f"Computed {computed_rows} contextual metric rows.",
    }


@router.post("/{dataset_id}/quantification/relational/recompute")
async def recompute_dataset_relational_metrics(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """Explicitly (re)compute relational-tier metrics (number of children) for a dataset.

    Runs :func:`compute_relational_metrics_for_dataset` with ``only_stale=True``: images
    with at least one contour missing a fresh row (new contour, or a parent row marked stale
    by ``mark_relational_stale_for_parent`` after a child was added / removed / re-parented)
    are recomputed - the whole image, so children are countable in the context (see that
    function's docstring). Useful to pre-warm the cache (e.g. right after a bulk import)
    rather than paying the cost lazily on the next ``GET .../summary`` call.

    Args:
        dataset_id (int): The ID of the dataset to recompute relational metrics for.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: ``{success, computed_rows, message}``.
    """
    if dataset_id not in user.available_datasets:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access to this dataset.")

    computed_rows = compute_relational_metrics_for_dataset(db, dataset_id, only_stale=True)
    return {
        "success": True,
        "computed_rows": computed_rows,
        "message": f"Computed {computed_rows} relational metric rows.",
    }


@router.get("/quantification/metrics/catalog")
async def get_metrics_catalog(
        user: User = Depends(get_current_user)
):
    """Serializable catalog of every registered quantification metric.

    Dataset-independent: returns :func:`iquana_toolbox.quantification.list_metrics` so the
    frontend can render metric pickers and know each metric's tier / unit_kind / value_dim
    / component names to render it generically. Placed under the datasets prefix (the
    router this file owns) but requires no dataset, so no per-dataset access check applies.

    Args:
        user (User): The current authenticated user.

    Returns:
        dict: ``{success, metrics}`` where ``metrics`` is the metric catalog list.
    """
    return {"success": True, "metrics": list_metrics()}


@router.get("/{dataset_id}/quantification/profiles")
async def get_quantification_profiles(
        dataset_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """List the dataset's quantification profiles, auto-creating a default if none exist.

    The auto-created default is the four geometry metrics on all labels, so an existing
    dataset renders exactly as before profiles existed.

    Args:
        dataset_id (int): The dataset whose profiles to list.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: ``{success, profiles}`` where each profile is a QuantificationProfile dump.
    """
    if dataset_id not in user.available_datasets:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access to this dataset.")

    profiles = profiles_db.list_profiles(db, dataset_id)
    return {"success": True, "profiles": [p.model_dump() for p in profiles]}


@router.post("/{dataset_id}/quantification/profiles")
async def create_quantification_profile(
        dataset_id: int,
        body: ProfileBody,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """Create a new quantification profile for the dataset.

    Metric keys in the entries are validated against the registry (unknown keys are
    rejected with a 422 by the schema). Marking the profile default unsets any previous
    default for the dataset.

    Args:
        dataset_id (int): The dataset to create the profile in.
        body (ProfileBody): The profile name / default flag / entries.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: ``{success, profile}`` with the created profile dump.
    """
    if dataset_id not in user.available_datasets:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access to this dataset.")

    try:
        schema = QuantificationProfile(
            dataset_id=dataset_id,
            name=body.name,
            is_default=body.is_default,
            entries=[e.model_dump() for e in body.entries],
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    created = profiles_db.create_profile(db, schema)
    return {"success": True, "profile": created.model_dump()}


@router.put("/{dataset_id}/quantification/profiles/{profile_id}")
async def update_quantification_profile(
        dataset_id: int,
        profile_id: int,
        body: ProfileBody,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """Update a profile's name / entries / default flag.

    Setting ``is_default`` True unsets it on every other profile of the dataset.

    Args:
        dataset_id (int): The dataset the profile belongs to.
        profile_id (int): The profile to update.
        body (ProfileBody): The new name / default flag / entries.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: ``{success, profile}`` with the updated profile dump.
    """
    if dataset_id not in user.available_datasets:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access to this dataset.")

    row = profiles_db.get_profile(db, dataset_id, profile_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found.")

    try:
        schema = QuantificationProfile(
            id=profile_id,
            dataset_id=dataset_id,
            name=body.name,
            is_default=body.is_default,
            entries=[e.model_dump() for e in body.entries],
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    updated = profiles_db.update_profile(db, row, schema)
    return {"success": True, "profile": updated.model_dump()}


@router.delete("/{dataset_id}/quantification/profiles/{profile_id}")
async def delete_quantification_profile(
        dataset_id: int,
        profile_id: int,
        db: Session = Depends(get_session),
        user: User = Depends(get_current_user)
):
    """Delete a quantification profile.

    If the deleted profile was the default, the lowest-id remaining profile is promoted to
    default. Deleting the last profile is allowed - the next profiles listing re-creates
    the geometry default, so the dataset always has a usable default.

    Args:
        dataset_id (int): The dataset the profile belongs to.
        profile_id (int): The profile to delete.
        db (Session): The database session.
        user (User): The current authenticated user.

    Returns:
        dict: ``{success, message}``.
    """
    if dataset_id not in user.available_datasets:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not have access to this dataset.")

    row = profiles_db.get_profile(db, dataset_id, profile_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found.")

    profiles_db.delete_profile(db, row)
    return {"success": True, "message": "Profile deleted successfully."}


@router.get(
    "/{dataset_id}/quantification/download")
async def download_dataset_quantification(
        dataset_id: int,
        exclude_unreviewed: bool = True,
        exclude_not_fully_annotated: bool = True,
        file_format: Literal["json", "csv"] = "json",
        profile_id: int | None = None,
        image_id: int | None = None,
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
        image_id (int | None): Restrict the export to one image of the dataset - the same
            columns, only that image's rows. This is what the per-image view both tabulates
            and offers as "Export this image", so the table on screen and the file that
            comes out of it are produced by one code path.
        db (Session, optional): The database session. Defaults to Depends(get_session). This is a fastapi dependency.
        user (AuthenticatedUser): The current authenticated user.

    Returns:
        dict: ``{success: False, message}`` when the dataset has no rows, otherwise a
        ``Response`` carrying the whole document as JSON or CSV.
    """

    _assert_image_in_dataset(db, dataset_id, image_id)

    # When a profile is given, the export emits one column per profile metric/component
    # from contour_metrics; otherwise it keeps the legacy four-geometry-column shape.
    metric_scoping, profile_tiers = _resolve_profile_scoping(db, dataset_id, profile_id)
    if "geometry" in profile_tiers:
        compute_geometry_metrics_for_dataset(db, dataset_id, only_stale=True)
    if profile_id is not None:
        if "appearance" in profile_tiers:
            compute_appearance_metrics_for_dataset(db, dataset_id, only_stale=True)
        if "contextual" in profile_tiers:
            compute_contextual_metrics_for_dataset(db, dataset_id, only_stale=True)
        if "relational" in profile_tiers:
            compute_relational_metrics_for_dataset(db, dataset_id, only_stale=True)

    dataset_name = (await datasets_db.get_dataset(dataset_id, db=db, )).name
    df = await datasets_db.get_dataset_as_df(
        dataset_id, exclude_not_fully_annotated, exclude_unreviewed, db,
        metric_scoping=metric_scoping, image_id=image_id,
    )
    if df.empty:
        return {
            "success": False,
            "message": "No data found for the given dataset and filters."
        }
    else:
        # Sent as one whole response rather than streamed.
        #
        # `to_json` / `to_csv` build the entire document in memory before this point, so
        # there is nothing left to stream and a `StreamingResponse` only costs. It costs a
        # lot in the JSON case: given a plain `str` it iterates the *string*, which yields
        # one character at a time, so a 150 KB export left here as one ASGI body message
        # per character — 150,000 chunked-transfer frames, each with its own framing
        # overhead and socket write. Measured on a real dataset that was 1.3 minutes of
        # "content download" for a few hundred rows. (The CSV branch happened to escape it
        # by wrapping in `StringIO`, which iterates by line.)
        #
        # A plain `Response` also sets `Content-Length`, which lets the client show real
        # progress and lets `GZipMiddleware` compress the body in one pass.
        match file_format:
            case "json":
                content = df.to_json(orient="records", default_handler=str)
                media_type = "application/json"
            case "csv":
                content = df.to_csv(index=False)
                media_type = "text/csv"
            case _:
                raise ValueError(f"Invalid file format: {file_format}")
        response = Response(content=content, media_type=media_type)
        stem = f"{dataset_name.replace(' ', '_')}_dataset"
        if image_id is not None:
            image_row = db.query(Images.file_name).filter(Images.id == image_id).first()
            file_stem = os.path.splitext(image_row.file_name)[0] if image_row else str(image_id)
            stem = f"{dataset_name.replace(' ', '_')}_{file_stem.replace(' ', '_')}"
        response.headers[
            "Content-Disposition"] = f'attachment; filename="{stem}.{file_format}"'
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

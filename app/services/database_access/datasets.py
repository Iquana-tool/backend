import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
import json
from logging import getLogger
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from iquana_toolbox.quantification import METRIC_REGISTRY, UnitKind, get_metric, resolve_unit
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.image import Image
from iquana_toolbox.schemas.database.labels import LabelHierarchy
from iquana_toolbox.schemas.user import User
from sqlalchemy import case, func
from sqlalchemy.orm import Session, aliased

from app.database.contour_metrics import ContourMetrics
from app.database.contours import Contours
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.users import Users
from app.services import image_status
from app.services.database_access.labels import get_hierarchical_label_name
from app.services.database_access.members import ensure_owner_membership
from config import DATASETS_DIR
from PIL import Image as PILImage

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

    # Ownership is a membership row so it can later be transferred; `created_by`
    # stays as the immutable record of who made the dataset.
    ensure_owner_membership(new_dataset.id, owner_username, db)
    return new_dataset


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
    """Per-phase progress of a dataset: Calibrate, Annotate, Review and overall.

    Counted over *images*, not masks. An image that has never been opened has no
    mask row at all, and counting masks silently dropped it from the denominator —
    a dataset of 100 images with 3 opened reported "3 total, 3 not started".

    Returns:
        ``(counts, total_images)`` where ``counts`` is
        ``{phase: {state: n}}`` for each of ``calibrate`` / ``annotate`` / ``review``
        plus an ``overall`` row. See :mod:`app.services.image_status`.
    """
    images = db.query(Images).filter(Images.dataset_id == dataset_id).all()
    statuses = image_status.status_for_images(db, images)
    counts = image_status.count_phases(statuses.values())
    return counts, len(images)


async def get_datasets_of_user(
        user: User,
        db: Session
):
    """Every dataset the user holds a role on. Admins see all of them."""
    if getattr(user, "is_admin", False):
        return db.query(Datasets)
    return db.query(Datasets).filter(Datasets.id.in_(user.available_datasets))


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
) -> bool:
    """Whether a user may delete a dataset, i.e. whether they own it.

    Route handlers should use `require(Permission.DATASET_DELETE)` instead; this
    exists for callers outside the request cycle.
    """
    from app.database.dataset_members import DatasetMembers
    from app.schemas.permissions import DatasetRole

    role = (
        db.query(DatasetMembers.role)
        .filter_by(dataset_id=dataset_id, username=username)
        .scalar()
    )
    if role is not None:
        return role == DatasetRole.OWNER.value
    # Fall back to creator for datasets predating membership rows.
    return db.query(Datasets.created_by).filter_by(id=dataset_id).scalar() == username


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
        filter_for_status: Literal["blocked", "not_started", "in_progress", "finished"] | None = None,
        filter_for_phase: Literal["calibrate", "annotate", "review"] | None = None,
):
    """Every image of a dataset with its mask id and its three phase statuses.

    Args:
        dataset_id: Dataset to list.
        db: SQLAlchemy session.
        filter_for_status: Keep only images in this state. Applied to the overall
            status, or to one phase when ``filter_for_phase`` is given. ``blocked``
            only ever matches the review phase.
        filter_for_phase: Which phase ``filter_for_status`` refers to. Without it
            the filter is on the overall status.

    Returns:
        A list of ``{image_id, mask_id, status, phases}`` dicts. ``mask_id`` is
        None for an image that has never been opened for annotation.
    """
    # LEFT join: an image with no mask row is not started, not absent. The old
    # inner join hid every untouched image from the gallery's status filters.
    rows = (
        db.query(Images, Masks)
        .outerjoin(Masks, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
        .all()
    )
    images = [image for image, _ in rows]
    masks_by_image = {image.id: mask for image, mask in rows if mask is not None}
    statuses = image_status.status_for_images(db, images, masks_by_image=masks_by_image)

    image_data = []
    for image in images:
        entry = statuses[image.id]
        if filter_for_status:
            actual = (entry["phases"][filter_for_phase] if filter_for_phase
                      else entry["status"])
            if actual != filter_for_status:
                continue
        image_data.append({
            "image_id": image.id,
            "mask_id": entry["mask_id"],
            "status": entry["status"],
            "phases": entry["phases"],
        })
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
        metric_scoping: dict[str, list[int] | None] | None = None,
):
    """Flat per-contour export dataframe.

    Without ``metric_scoping`` (the default, legacy path) emits the four legacy geometry
    columns straight off the ``contours`` row, unchanged. With ``metric_scoping`` (a
    profile's ``{metric_key: label_ids | None}`` map) it instead emits one column per
    profile metric, pulled from the tall ``contour_metrics`` table so appearance /
    contextual / multi-component metrics are exportable too: a multi-component metric
    becomes one column per component (e.g. ``mean_color_rgb_r`` / ``_g`` / ``_b``), and a
    metric only fills a row when that row's label is in the metric's scope.
    """
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

    if metric_scoping is not None:
        return _dataset_df_from_profile(
            data, dataset_id, metric_scoping, db,
            exclude_not_fully_annotated, exclude_unreviewed,
        )

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


def _metric_column_name(metric_key: str, component: int) -> str:
    """Column name for a metric component in the profile export.

    Single-component metrics keep their bare key (``area``); multi-component metrics get a
    per-component suffix from the registry's component names when available
    (``mean_color_rgb_r``), falling back to the numeric index (``mean_color_rgb_0``).
    """
    from iquana_toolbox.quantification import METRIC_REGISTRY

    metric = METRIC_REGISTRY.get(metric_key)
    if metric is None or metric.value_dim <= 1:
        return metric_key
    if metric.components and component < len(metric.components):
        return f"{metric_key}_{metric.components[component].lower()}"
    return f"{metric_key}_{component}"


def _dataset_df_from_profile(
        rows,
        dataset_id: int,
        metric_scoping: dict[str, list[int] | None],
        db: Session,
        exclude_not_fully_annotated: bool,
        exclude_unreviewed: bool,
) -> pd.DataFrame:
    """Build the flat export dataframe for a profile from the tall ``contour_metrics`` table.

    One row per contour; one column per (profile metric, component). A cell is filled only
    when the contour's label is in that metric's scope AND a metric row exists for it.

    Stored metric values are pixel-native; they are converted to the dataset display unit
    (per each contour's image scale) on the same all-or-nothing rule as the summary, so the
    exported numbers match the quantification page: physical units only when every
    contributing image shares one unit, pixels otherwise.
    """
    contour_ids = [row[0].id for row in rows]
    metric_keys = list(metric_scoping)

    scale_display = _resolve_scale_display(
        dataset_id, exclude_not_fully_annotated, exclude_unreviewed, db
    )
    display_physical = scale_display["display_physical"]

    # Per-contour image scale, for the pixel -> physical conversion below.
    scale_by_contour: dict[int, tuple[float, float]] = {}
    if display_physical and contour_ids:
        for cid, sx, sy in (
            db.query(Contours.id, Images.scale_x, Images.scale_y)
            .join(Masks, Masks.id == Contours.mask_id)
            .join(Images, Images.id == Masks.image_id)
            .filter(Contours.id.in_(contour_ids))
            .all()
        ):
            scale_by_contour[cid] = (sx, sy)

    # Pull all relevant metric rows in one query, index by (contour_id, metric_key, comp).
    values_by_key: dict[tuple[int, str, int], float] = {}
    if contour_ids and metric_keys:
        metric_rows = (
            db.query(ContourMetrics)
            .filter(
                ContourMetrics.contour_id.in_(contour_ids),
                ContourMetrics.metric_key.in_(metric_keys),
            )
            .all()
        )
        for mr in metric_rows:
            value = mr.value
            if display_physical:
                sx, sy = scale_by_contour.get(mr.contour_id, (1.0, 1.0))
                value *= _pixel_to_physical_factor(mr.metric_key, sx, sy)
            values_by_key[(mr.contour_id, mr.metric_key, mr.component)] = value

    # Determine the component columns to emit per metric from the registry's value_dim.
    from iquana_toolbox.quantification import METRIC_REGISTRY

    metric_components: dict[str, list[int]] = {}
    for key in metric_keys:
        metric = METRIC_REGISTRY.get(key)
        value_dim = metric.value_dim if metric is not None else 1
        metric_components[key] = list(range(value_dim))

    df_data: dict[str, list] = {}
    for row in rows:
        contour: Contours = row[0]
        file_name: str = row[1]
        label_db: Labels = row[2]

        df_data.setdefault("file_name", []).append(file_name)
        df_data.setdefault("label", []).append(label_db.name)
        df_data.setdefault("label_id", []).append(contour.label_id)
        df_data.setdefault("contour_id", []).append(contour.id)

        for key in metric_keys:
            allowed_labels = metric_scoping.get(key)
            in_scope = allowed_labels is None or contour.label_id in allowed_labels
            for component in metric_components[key]:
                col = _metric_column_name(key, component)
                value = values_by_key.get((contour.id, key, component)) if in_scope else None
                df_data.setdefault(col, []).append(value)

    return pd.DataFrame(df_data)


def _scope_contour_query(
        query,
        dataset_id: int,
        exclude_not_fully_annotated: bool,
        exclude_unreviewed: bool,
):
    """Join a ``Contours``-based query to its dataset and apply the two exclude filters.

    Shared by :func:`get_quantification_summary` and
    :func:`get_quantification_distribution` so the summary and the box/violin plots are
    always computed over exactly the same contour set - if they diverged, a median could
    fall outside the min/max the summary reports for the same metric.

    Args:
        query: A query selecting from (or joined to) ``Contours``.
        dataset_id: The dataset to scope to.
        exclude_not_fully_annotated: Drop contours on masks not marked fully annotated.
        exclude_unreviewed: Drop contours nobody has reviewed.

    Returns:
        The query with the dataset joins and filters applied.
    """
    query = (
        query.join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
    )
    if exclude_not_fully_annotated:
        query = query.filter(Masks.fully_annotated == True)
    if exclude_unreviewed:
        query = query.filter(Contours.reviewed_by.any())
    return query


# Metric keys grouped by how a pixel value converts to a physical value, derived once from
# the registry: LENGTH values scale by the (isotropic) pixel scale, AREA values by its
# square, everything else (ratio / count / colour / intensity) is scale-independent.
_LENGTH_METRIC_KEYS: set[str] = {k for k, m in METRIC_REGISTRY.items() if m.unit_kind == UnitKind.LENGTH}
_AREA_METRIC_KEYS: set[str] = {k for k, m in METRIC_REGISTRY.items() if m.unit_kind == UnitKind.AREA}


def _resolve_scale_display(
        dataset_id: int,
        exclude_not_fully_annotated: bool,
        exclude_unreviewed: bool,
        db: Session,
) -> dict[str, Any]:
    """Decide which unit the aggregated numbers should be reported in for a dataset.

    Stored ``contour_metrics`` values are PIXEL-native (see
    :func:`app.services.quantification.build_pixel_quant_context`); the per-image physical
    scale is applied here, at read time. It is only meaningful to report physical units when
    every image contributing objects to the summary shares ONE unit - otherwise you would be
    pooling, say, mm² and px² into a single mean. So:

      * all contributing images share one physical unit (e.g. all "mm")  -> report physical
        (``consistent=True``, values converted per image via its own scale),
      * all contributing images are unscaled ("px")                       -> report pixels
        (``consistent=True``, the normal all-pixel case, no warning),
      * a MIX (some scaled, some not, or different physical units)         -> report pixels
        (``consistent=False``); the frontend shows a warning and the numbers stay in px.

    Returns a dict describing the decision (also surfaced to the client as ``scale_status``):
        display_unit: The image length unit the numbers are in ("px" or e.g. "mm").
        display_physical: Whether a physical conversion should be applied on read.
        consistent: Whether the contributing images share one unit (drives the warning).
        distinct_units: Sorted distinct units among contributing images.
        images_scaled / images_unscaled: Contributing-image counts by scaled/unscaled.
    """
    unit_rows = (
        db.query(Images.id, Images.unit)
        .select_from(Contours)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
    )
    if exclude_not_fully_annotated:
        unit_rows = unit_rows.filter(Masks.fully_annotated == True)
    if exclude_unreviewed:
        unit_rows = unit_rows.filter(Contours.reviewed_by.any())

    units = [(row.unit or "px") for row in unit_rows.distinct().all()]
    distinct_units = sorted(set(units))
    images_scaled = sum(1 for u in units if u != "px")
    images_unscaled = sum(1 for u in units if u == "px")

    consistent = len(distinct_units) == 1
    display_physical = consistent and distinct_units[0] != "px"
    display_unit = distinct_units[0] if display_physical else "px"

    return {
        "display_unit": display_unit,
        "display_physical": display_physical,
        "consistent": consistent,
        "distinct_units": distinct_units,
        "images_scaled": images_scaled,
        "images_unscaled": images_unscaled,
    }


def _pixel_to_physical_factor(metric_key: str, scale_x: float, scale_y: float) -> float:
    """Multiplier converting a metric's pixel value to physical units for one image.

    Area metrics scale by the product of the per-axis pixel scale (always exact). Length
    metrics scale by ``scale_x``: exact for the isotropic scales the calibration UI produces
    (``scale_x == scale_y``), and a well-defined choice for the rare anisotropic case (a
    single-axis pixel length cannot be recovered from one stored scalar anyway). Everything
    else (ratio / count / colour / intensity) is scale-independent. Kept in lockstep with the
    SQL conversion in :func:`get_quantification_summary` so the summary and the box/violin
    distributions never report the same metric in different units.
    """
    if metric_key in _AREA_METRIC_KEYS:
        return float(scale_x) * float(scale_y)
    if metric_key in _LENGTH_METRIC_KEYS:
        return float(scale_x)
    return 1.0


async def get_quantification_summary(
        dataset_id: int,
        exclude_not_fully_annotated: bool,
        exclude_unreviewed: bool,
        db: Session,
        metric_scoping: dict[str, list[int] | None] | None = None,
) -> dict[str, Any]:
    """Aggregate the tall ``contour_metrics`` rows of a dataset server-side.

    Groups the metric rows by ``(label_id, metric_key, component, unit)`` and computes
    count / mean / std / min / max entirely in SQL (SQLite has no ``stddev``, so the
    population standard deviation is derived in python from ``E[x]`` and ``E[x²]``). Also
    computes the parent-label -> child-label counts the legacy page shows. The two exclude
    filters mirror :func:`get_dataset_as_df` so the summary matches the flat export.

    Args:
        dataset_id: The dataset to summarize.
        exclude_not_fully_annotated: Drop contours on masks not marked fully annotated.
        exclude_unreviewed: Drop contours nobody has reviewed.
        db: The database session.
        metric_scoping: Optional profile scoping. When given, only the metric keys present
            in this mapping are aggregated, and each key's value (a list of label ids, or
            ``None`` for all labels) restricts which labels that metric is reported for.
            ``None`` (the default) keeps the legacy behavior: every metric, every label.

    Returns:
        A dict with ``metrics`` (nested label_id -> metric_key -> {unit, components}),
        ``child_counts_per_label_id`` (parent label_id -> child label_id -> count),
        ``object_counts_per_label_id`` (label_id -> total / reviewed / unreviewed) and
        ``scale_status`` (see :func:`_resolve_scale_display`). Metric values are reported in
        the dataset's physical unit only when every contributing image shares one unit;
        otherwise they are reported pixel-native and ``scale_status["consistent"]`` is False
        so the frontend can warn. The object counts are a full per-class census and
        intentionally ignore both exclude filters (see ``_compute_object_counts``).
    """
    # Base join scoping to the dataset; the same for both aggregations below.
    def _apply_filters(query):
        return _scope_contour_query(query, dataset_id, exclude_not_fully_annotated,
                                    exclude_unreviewed)

    def _compute_object_counts() -> dict[str, dict[str, int]]:
        # Per-label census of annotated objects: total / reviewed / unreviewed. A contour
        # counts as reviewed iff at least one user has reviewed it (matching the semantics
        # used elsewhere, e.g. Masks.review_status and Contours.reviewed_by.any()).
        #
        # NB: this deliberately ignores both exclude filters. Applying exclude_unreviewed
        # would force "unreviewed" to always be 0, and applying exclude_not_fully_annotated
        # would hide in-progress annotation work; the whole point of this breakdown is to
        # show the full class census and how much of it still needs review.
        reviewed_flag = case((Contours.reviewed_by.any(), 1), else_=0)
        count_query = (
            db.query(
                Contours.label_id.label("label_id"),
                func.count(Contours.id).label("total"),
                func.sum(reviewed_flag).label("reviewed"),
            )
            .join(Masks, Masks.id == Contours.mask_id)
            .join(Images, Images.id == Masks.image_id)
            .filter(Images.dataset_id == dataset_id)
            .group_by(Contours.label_id)
        )

        result: dict[str, dict[str, int]] = {}
        for row in count_query.all():
            total = int(row.total)
            reviewed = int(row.reviewed or 0)
            result[str(row.label_id)] = {
                "total": total,
                "reviewed": reviewed,
                "unreviewed": total - reviewed,
            }
        return result

    def _compute_child_counts() -> dict[str, dict[str, int]]:
        # Count, per parent label, how many child contours of each child label exist. This
        # is the flat form of ContourHierarchy.get_label_quantification's child_counts:
        # keyed by the PARENT's label id -> {child label id: total child contours}.
        parent_contours = aliased(Contours)
        child_query = _apply_filters(
            db.query(
                parent_contours.label_id.label("parent_label_id"),
                Contours.label_id.label("child_label_id"),
                func.count(Contours.id).label("count"),
            ).join(parent_contours, parent_contours.id == Contours.parent_id)
        ).group_by(parent_contours.label_id, Contours.label_id)

        result: dict[str, dict[str, int]] = {}
        for row in child_query.all():
            parent_key = str(row.parent_label_id)
            result.setdefault(parent_key, {})[str(row.child_label_id)] = int(row.count)
        return result

    def _compute_object_counts() -> dict[str, dict[str, int]]:
        # How many contours exist per label, split by whether anyone has reviewed them.
        #
        # Deliberately NOT run through _apply_filters: this is a census of the dataset, and
        # the two exclude filters are exactly what it is meant to report on. Filtering here
        # would make "unreviewed" always read 0 under exclude_unreviewed=True and hide the
        # work still sitting on not-fully-annotated masks - so the counts are identical
        # whatever the filters are set to, which is what makes them a useful denominator
        # for the filtered metrics above.
        reviewed_flag = case((Contours.reviewed_by.any(), 1), else_=0)
        count_query = (
            db.query(
                Contours.label_id.label("label_id"),
                func.count(Contours.id).label("total"),
                func.sum(reviewed_flag).label("reviewed"),
            )
            .join(Masks, Masks.id == Contours.mask_id)
            .join(Images, Images.id == Masks.image_id)
            .filter(Images.dataset_id == dataset_id)
            .group_by(Contours.label_id)
        )

        result: dict[str, dict[str, int]] = {}
        for row in count_query.all():
            total = int(row.total)
            reviewed = int(row.reviewed or 0)
            result[str(row.label_id)] = {
                "total": total,
                "reviewed": reviewed,
                "unreviewed": total - reviewed,
            }
        return result

    # --- Scale display decision ---------------------------------------------
    # Stored metric values are pixel-native; decide here whether to convert them to a
    # physical unit on read (only when every contributing image shares one unit) or report
    # pixels (the default, and the fallback when a dataset mixes scaled/unscaled images).
    scale_display = _resolve_scale_display(
        dataset_id, exclude_not_fully_annotated, exclude_unreviewed, db
    )
    display_unit = scale_display["display_unit"]

    # When reporting physical units, convert each pixel value to physical IN SQL by
    # multiplying by the row's image scale (area by scale_x*scale_y, length by scale_x -
    # see _pixel_to_physical_factor for the isotropy note). Deliberately multiplication-only
    # so the expression runs on SQLite (which has no sqrt). Otherwise values are already px.
    if scale_display["display_physical"]:
        value_expr = ContourMetrics.value * case(
            (ContourMetrics.metric_key.in_(_AREA_METRIC_KEYS), Images.scale_x * Images.scale_y),
            (ContourMetrics.metric_key.in_(_LENGTH_METRIC_KEYS), Images.scale_x),
            else_=1.0,
        )
    else:
        value_expr = ContourMetrics.value

    # --- Metric aggregation -------------------------------------------------
    # Grouped WITHOUT unit now: stored units are uniformly pixel-native, and the reported
    # unit is derived in python from the metric's unit kind and the display unit, so a mix of
    # scaled/unscaled images can no longer collide two units into one metric entry.
    metric_query = _apply_filters(
        db.query(
            Contours.label_id.label("label_id"),
            ContourMetrics.metric_key.label("metric_key"),
            ContourMetrics.component.label("component"),
            func.count(value_expr).label("count"),
            func.avg(value_expr).label("mean"),
            func.avg(value_expr * value_expr).label("mean_sq"),
            func.min(value_expr).label("min"),
            func.max(value_expr).label("max"),
        ).join(ContourMetrics, ContourMetrics.contour_id == Contours.id)
    ).group_by(
        Contours.label_id,
        ContourMetrics.metric_key,
        ContourMetrics.component,
    )

    # Profile scoping: restrict to the profile's metric keys up front so the aggregation
    # itself is cheaper, then honor each metric's per-label scoping row-by-row below.
    if metric_scoping is not None:
        if not metric_scoping:
            # A profile with no entries -> no metrics to report.
            return {
                "metrics": {},
                "child_counts_per_label_id": _compute_child_counts(),
                "object_counts_per_label_id": _compute_object_counts(),
                "scale_status": scale_display,
            }
        metric_query = metric_query.filter(ContourMetrics.metric_key.in_(list(metric_scoping)))

    metrics: dict[str, dict[str, Any]] = {}
    for row in metric_query.all():
        # Honor per-metric label scoping: skip rows for labels not in this metric's scope.
        if metric_scoping is not None:
            allowed_labels = metric_scoping.get(row.metric_key)
            if allowed_labels is not None and row.label_id not in allowed_labels:
                continue
        mean = float(row.mean) if row.mean is not None else 0.0
        mean_sq = float(row.mean_sq) if row.mean_sq is not None else 0.0
        # Population std = sqrt(max(0, E[x^2] - E[x]^2)); clamp to guard float noise.
        variance = max(0.0, mean_sq - mean * mean)
        std = float(np.sqrt(variance))

        # The reported unit is derived (not read from the stored row) from the metric's unit
        # kind and the dataset display unit: "mm"/"mm²" when reporting physical, else px.
        unit = resolve_unit(get_metric(row.metric_key).unit_kind, display_unit)

        label_key = str(row.label_id)
        metric_entry = metrics.setdefault(label_key, {}).setdefault(
            row.metric_key, {"unit": unit, "components": {}}
        )
        metric_entry["components"][int(row.component)] = {
            "count": int(row.count),
            "mean": mean,
            "std": std,
            "min": float(row.min) if row.min is not None else 0.0,
            "max": float(row.max) if row.max is not None else 0.0,
        }

    # Flatten each metric's component dict into an ordered list (component 0, 1, ...).
    for label_metrics in metrics.values():
        for metric_entry in label_metrics.values():
            components = metric_entry.pop("components")
            metric_entry["components"] = [components[i] for i in sorted(components)]

    # --- Child-count / object-count aggregation ------------------------------
    return {
        "metrics": metrics,
        "child_counts_per_label_id": _compute_child_counts(),
        "object_counts_per_label_id": _compute_object_counts(),
        "scale_status": scale_display,
    }


# --- Distribution (box / violin) statistics ----------------------------------------

# Number of points on the KDE evaluation grid. The curve is sent to the frontend as
# plain arrays, so this directly bounds the payload size per metric/component.
_KDE_GRID_POINTS = 128

# Upper bound on how many individual outlier VALUES are returned. The true count is
# always reported in full as ``outlier_count``; only the sample is capped, so a metric
# with thousands of outliers cannot blow up the response.
_MAX_OUTLIER_SAMPLES = 50

# Bin count for the histogram fallback used when a KDE cannot be fitted.
_HISTOGRAM_BINS = 30

# Tukey's constant: points more than 1.5 IQR beyond a quartile are outliers, and the
# whiskers stop at the most extreme point that is NOT an outlier.
_WHISKER_IQR_FACTOR = 1.5


def _empty_distribution_stats() -> dict[str, Any]:
    """Zeroed stats payload for a metric/component that has no values at all."""
    return {
        "count": 0,
        "min": 0.0, "max": 0.0, "mean": 0.0,
        "q1": 0.0, "median": 0.0, "q3": 0.0,
        "whisker_low": 0.0, "whisker_high": 0.0,
        "outlier_count": 0, "outliers": [],
        "kde": None, "histogram": None,
    }


def _compute_distribution_stats(values: np.ndarray) -> dict[str, Any]:
    """Reduce a 1-D array of metric values to a box/violin-plot payload.

    Pure numpy/scipy - no database access - so it is cheap to test directly. Computes:

      * the five-number summary (min / q1 / median / q3 / max) plus the mean, using
        numpy's default linear-interpolation percentiles,
      * Tukey whiskers and outliers (see :data:`_WHISKER_IQR_FACTOR`): the whiskers stop
        at the most extreme value INSIDE the fences, so they never reach an outlier,
      * a bounded sample of the outlier values, and
      * a smooth density curve for the violin: a Gaussian KDE, falling back to a
        histogram when a KDE cannot be fitted (too few points, or scipy unavailable).

    A distribution with no spread (every value identical, or a single value) gets neither
    a KDE nor a histogram: there is no shape to draw, and a KDE would be singular.

    Args:
        values: The raw metric values. Non-finite entries are dropped.

    Returns:
        A JSON-serializable stats dict; see :func:`_empty_distribution_stats` for the keys.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    count = int(values.size)
    if count == 0:
        return _empty_distribution_stats()

    q1, median, q3 = (float(v) for v in np.percentile(values, [25, 50, 75]))
    iqr = q3 - q1
    lower_fence = q1 - _WHISKER_IQR_FACTOR * iqr
    upper_fence = q3 + _WHISKER_IQR_FACTOR * iqr

    is_outlier = (values < lower_fence) | (values > upper_fence)
    inliers = values[~is_outlier]
    # Every point being an "outlier" is impossible for a real IQR, but guard anyway so a
    # degenerate input cannot produce whiskers from an empty array.
    whisker_source = inliers if inliers.size else values
    whisker_low = float(whisker_source.min())
    whisker_high = float(whisker_source.max())

    outliers = np.sort(values[is_outlier])
    if outliers.size > _MAX_OUTLIER_SAMPLES:
        # Evenly spaced picks over the SORTED outliers, so the sample keeps both extremes
        # and stays representative of the tail - and is deterministic (no RNG).
        indices = np.unique(
            np.linspace(0, outliers.size - 1, _MAX_OUTLIER_SAMPLES).round().astype(int)
        )
        outlier_sample = outliers[indices]
    else:
        outlier_sample = outliers

    kde, histogram = _compute_density_curve(values)

    return {
        "count": count,
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
        "outlier_count": int(outliers.size),
        "outliers": [float(v) for v in outlier_sample],
        "kde": kde,
        "histogram": histogram,
    }


def _compute_density_curve(values: np.ndarray) -> tuple[dict | None, dict | None]:
    """Build the violin curve for ``values`` as ``(kde, histogram)``; at most one is set.

    Returns ``(None, None)`` when the values have no spread - a KDE of a zero-variance
    sample is singular, and a single-bin histogram carries no information.
    """
    low, high = float(values.min()), float(values.max())
    if values.size < 2 or high <= low:
        return None, None

    grid = np.linspace(low, high, _KDE_GRID_POINTS)
    try:
        from scipy.stats import gaussian_kde

        density = gaussian_kde(values)(grid)
    except (ImportError, ValueError, np.linalg.LinAlgError) as exc:
        # Singular covariance (near-duplicate points) or no scipy: fall back to a
        # histogram so the frontend still has a shape to draw.
        logger.debug("KDE unavailable for %d values (%s); using a histogram.", values.size, exc)
        counts, edges = np.histogram(values, bins=min(_HISTOGRAM_BINS, values.size))
        return None, {
            "bin_edges": [float(e) for e in edges],
            "counts": [int(c) for c in counts],
        }

    # Clamp: a KDE evaluated on a coarse grid can return tiny negative values.
    return {
        "x": [float(v) for v in grid],
        "density": [float(max(d, 0.0)) for d in density],
    }, None


def _distribution_metric_keys(metric_keys: Iterable[str] | None = None) -> set[str]:
    """The registry metrics a distribution can be drawn for, optionally intersected.

    Only single-component metrics are eligible: a box or violin plot summarizes ONE
    ordered numeric axis, which a multi-component value (an RGB or LAB colour, whose
    channels are not comparable to each other) does not provide.

    Args:
        metric_keys: Optional keys to restrict to, e.g. a profile's scoped metrics.
            ``None`` returns every eligible metric.

    Returns:
        The set of eligible metric keys.
    """
    eligible = {key for key, metric in METRIC_REGISTRY.items() if metric.value_dim == 1}
    if metric_keys is None:
        return eligible
    return eligible & set(metric_keys)


async def get_quantification_distribution(
        dataset_id: int,
        exclude_not_fully_annotated: bool,
        exclude_unreviewed: bool,
        db: Session,
        metric_scoping: dict[str, list[int] | None] | None = None,
) -> dict[str, Any]:
    """Compute per-label box/violin distributions for a dataset's metrics.

    Complements :func:`get_quantification_summary`: that one aggregates to scalars in SQL,
    this one needs the raw values (percentiles and a KDE cannot be expressed as SQL
    aggregates), so the in-scope values ARE loaded into memory. The eligible metrics are
    restricted to single-component ones (see :func:`_distribution_metric_keys`) and the
    per-row payload is bounded (see :data:`_MAX_OUTLIER_SAMPLES`,
    :data:`_KDE_GRID_POINTS`), which keeps both the query and the response bounded in
    practice.

    Args:
        dataset_id: The dataset to summarize.
        exclude_not_fully_annotated: Drop contours on masks not marked fully annotated.
        exclude_unreviewed: Drop contours nobody has reviewed.
        db: The database session.
        metric_scoping: Optional profile scoping, interpreted exactly as in
            :func:`get_quantification_summary`: only the listed metric keys are computed,
            and each key's label list (or ``None`` for all labels) restricts which labels
            it is reported for.

    Returns:
        ``{label_id: {metric_key: {component: stats}}}`` with every key stringified for
        JSON. Labels and metrics with no values are omitted entirely, so an empty dict
        means nothing in scope had a stored value.
    """
    eligible_keys = _distribution_metric_keys(
        None if metric_scoping is None else metric_scoping.keys()
    )
    if not eligible_keys:
        return {}

    # Same pixel-native -> physical decision as the summary, so a metric's box/violin plot
    # and its scalar card always agree on the unit (and thus the values).
    scale_display = _resolve_scale_display(
        dataset_id, exclude_not_fully_annotated, exclude_unreviewed, db
    )
    display_unit = scale_display["display_unit"]
    display_physical = scale_display["display_physical"]

    value_query = _scope_contour_query(
        db.query(
            Contours.label_id.label("label_id"),
            ContourMetrics.metric_key.label("metric_key"),
            ContourMetrics.component.label("component"),
            ContourMetrics.value.label("value"),
            Images.scale_x.label("scale_x"),
            Images.scale_y.label("scale_y"),
        ).join(ContourMetrics, ContourMetrics.contour_id == Contours.id),
        dataset_id, exclude_not_fully_annotated, exclude_unreviewed,
    ).filter(ContourMetrics.metric_key.in_(sorted(eligible_keys)))

    # Bucket the values by (label, metric, component), converting each pixel value to the
    # display unit per its own image scale (a no-op factor of 1 when reporting pixels).
    buckets: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for row in value_query.all():
        if metric_scoping is not None:
            allowed_labels = metric_scoping.get(row.metric_key)
            if allowed_labels is not None and row.label_id not in allowed_labels:
                continue
        key = (row.label_id, row.metric_key, int(row.component))
        value = float(row.value)
        if display_physical:
            value *= _pixel_to_physical_factor(row.metric_key, row.scale_x, row.scale_y)
        buckets[key].append(value)

    result: dict[str, Any] = {}
    for (label_id, metric_key, component), values in buckets.items():
        stats = _compute_distribution_stats(np.asarray(values, dtype=np.float64))
        stats["unit"] = resolve_unit(get_metric(metric_key).unit_kind, display_unit)
        result.setdefault(str(label_id), {}).setdefault(metric_key, {})[str(component)] = stats
    return result


def _native_image_dimensions(image: "Images") -> tuple[int, int]:
    """Resolve an image's native pixel dimensions for the COCO export.

    The COCO geometry is produced by scaling the normalized [0, 1] contour
    coordinates by the image size, so this MUST be the full-resolution size of the
    original file — never the thumbnail/preview size. We read it straight from the
    file on disk (only the header is decoded, so this is cheap) which is authoritative
    even if the stored ``Images.width/height`` columns are stale, and fall back to the
    stored columns only when the file cannot be read.
    """
    file_path = getattr(image, "file_path", None)
    if file_path and os.path.exists(file_path):
        try:
            with PILImage.open(file_path) as img:
                return img.width, img.height
        except (OSError, ValueError):
            logger.warning(
                "Could not read native size for image %s from %s; "
                "falling back to stored dimensions %sx%s.",
                image.id, file_path, image.width, image.height,
            )
    return int(image.width), int(image.height)


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


# Which contours of a (possibly nested) annotation tree to emit. COCO is flat, so
# the caller decides how the hierarchy collapses:
#   - "all":       every contour becomes its own annotation (parents overlap children)
#   - "leaves":    only contours that are not a parent within the result set
#   - "top_level": only contours without a parent
ContourSelection = Literal["all", "leaves", "top_level"]


def _filter_contour_rows(
        rows: list[tuple[Any, Any, Any]],
        contour_selection: ContourSelection,
) -> list[tuple[Any, Any, Any]]:
    """Filter (contour, image, label) rows according to the hierarchy selection."""
    if contour_selection == "leaves":
        parent_ids = {contour.parent_id for contour, _, _ in rows if contour.parent_id is not None}
        return [row for row in rows if row[0].id not in parent_ids]
    if contour_selection == "top_level":
        return [row for row in rows if row[0].parent_id is None]
    return rows


def build_coco_payload(
        dataset: "Datasets",
        rows: list[tuple[Any, Any, Any]],
) -> tuple[dict[str, Any], set[int]]:
    """Build a COCO JSON payload from pre-fetched (contour, image, label) rows.

    This is the single source of truth for the COCO document, shared by the ZIP
    export and the annotations-only endpoint. Returns the payload and the set of
    image ids it actually references, so callers can keep bundled images in sync.
    """
    images_by_id: dict[int, dict[str, Any]] = {}
    native_size_by_id: dict[int, tuple[int, int]] = {}
    categories_by_id: dict[int, dict[str, Any]] = {}
    annotations: list[dict[str, Any]] = []

    for contour, image, label in rows:
        if contour.label_id is None or label is None:
            continue

        if image.id not in images_by_id:
            # Native (full-resolution) dimensions are the single source of truth for
            # all geometry below; resolved once per image and reused for every contour.
            native_width, native_height = _native_image_dimensions(image)
            native_size_by_id[image.id] = (native_width, native_height)
            images_by_id[image.id] = {
                "id": image.id,
                "file_name": image.file_name,
                "width": native_width,
                "height": native_height,
            }
        width, height = native_size_by_id[image.id]

        if label.id not in categories_by_id:
            categories_by_id[label.id] = {
                "id": label.id,
                "name": label.name,
                "supercategory": "none",
            }

        polygon = _build_coco_polygon(contour.x, contour.y, width, height)
        if len(polygon) < 6:
            continue

        x_points = polygon[0::2]
        y_points = polygon[1::2]
        min_x, max_x = min(x_points), max(x_points)
        min_y, max_y = min(y_points), max(y_points)
        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

        # contour.area is computed from the normalized [0, 1] coordinates, so it is a
        # fraction of the image. COCO expects pixel^2, so scale by the native image
        # size to stay consistent with the (pixel) segmentation and bbox above. Using
        # native width*height recomputes the area in native space (it scales by
        # sx*sy, not a single factor).
        if contour.area is not None:
            area = float(contour.area) * float(width) * float(height)
        else:
            area = float(bbox[2] * bbox[3])

        annotations.append({
            "id": contour.id,
            "image_id": image.id,
            "category_id": label.id,
            "segmentation": [polygon],
            "area": area,
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
    return coco_payload, set(images_by_id.keys())


async def export_dataset_contours_to_coco(
        dataset_id: int,
        db: Session,
        exclude_not_fully_annotated: bool = True,
        exclude_unreviewed: bool = True,
        contour_selection: ContourSelection = "all",
        output_file_path: str | None = None,
        write_to_disk: bool = True,
        log_to_mlflow: bool = False,
        mlflow_run_id: str | None = None,
) -> dict[str, Any]:
    """Export all contours of a dataset to a COCO payload.

    Builds the COCO JSON via :func:`build_coco_payload` and, when ``write_to_disk``
    (or ``log_to_mlflow``) is set, persists it to disk and optionally logs it to
    MLflow. The returned dict always carries the in-memory ``coco_payload`` and the
    set of referenced ``image_ids``.
    """
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

    rows = _filter_contour_rows(query.all(), contour_selection)
    coco_payload, image_ids = build_coco_payload(dataset, rows)

    result: dict[str, Any] = {
        "success": True,
        "message": "COCO export created.",
        "dataset_id": dataset_id,
        "coco_payload": coco_payload,
        "image_ids": image_ids,
        "output_file_path": None,
        "num_images": len(coco_payload["images"]),
        "num_annotations": len(coco_payload["annotations"]),
        "num_categories": len(coco_payload["categories"]),
        "mlflow": "not_requested",
    }

    # The JSON is only persisted when a caller needs a file on disk (e.g. the ZIP
    # export bundles it, or MLflow logging requires an artifact path).
    if not (write_to_disk or log_to_mlflow):
        return result

    if output_file_path is None:
        output_file_path = os.path.join(str(dataset.folder_path), f"{dataset.name.replace(' ', '_')}_coco.json")
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file_path, "w", encoding="utf-8") as fp:
        json.dump(coco_payload, fp, indent=2)
    result["output_file_path"] = output_file_path
    result["message"] = "COCO export written to disk."

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
            result["mlflow"] = "logged"
        except Exception as exc:
            logger.warning("Could not log COCO export to MLflow: %s", exc)
            result["mlflow"] = f"failed: {exc}"

    return result



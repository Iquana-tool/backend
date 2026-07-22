"""Shared machinery for computing and persisting contour quantification metrics.

This module is the single seam Steps 3-4 (batch / lazy computation) reuse. It builds a
per-image :class:`~iquana_toolbox.quantification.context.QuantContext` and writes metric
values into the tall :class:`~app.database.contour_metrics.ContourMetrics` table via the
metric registry. Both the dual-write path in ``save_contour_tree`` and the backfill script
go through here so metric semantics live in exactly one place.
"""
import os
from logging import getLogger
from typing import Iterable

import numpy as np
from PIL import Image as PILImage
from iquana_toolbox.quantification import QuantContext, get_metric
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.database.quantification import QuantificationModel
from sqlalchemy import Select
from sqlalchemy.orm import Session

from app.database.contour_metrics import ContourMetrics
from app.database.contours import Contours
from app.database.images import Images
from app.database.masks import Masks

logger = getLogger(__name__)

# The four geometry metrics that are dual-written today (kept in sync with the legacy
# columns on ``contours``). Later steps extend this by passing explicit metric keys.
GEOMETRY_METRIC_KEYS: tuple[str, ...] = ("area", "perimeter", "circularity", "max_diameter")

# The appearance-tier metrics (Step 3): need the decoded image pixels, so they are
# computed lazily/batched via compute_appearance_metrics_for_dataset, never on the
# synchronous contour write path.
APPEARANCE_METRIC_KEYS: tuple[str, ...] = ("mean_color_rgb", "mean_color_lab", "mean_intensity")

# The contextual-tier metrics (Step 4): RELATIONAL - a contour's value depends on where
# its same-parent siblings are, so they are computed lazily/batched via
# compute_contextual_metrics_for_dataset, same as appearance, but with the additional
# wrinkle that invalidating one contour must invalidate its whole parent group (see
# mark_contextual_stale_for_group in app.services.database_access.contours).
CONTEXTUAL_METRIC_KEYS: tuple[str, ...] = ("nn_distance", "mean_knn_distance")

# The relational-tier metrics: a contour's value depends on its CHILDREN (contours naming
# it as parent), so like contextual they are computed lazily/batched, but their staleness
# is PARENT-TARGETED rather than sibling-group-wide - n_children only changes when a child
# is added/removed/re-parented, which affects exactly the (old/new) parent, never the
# parent's siblings (see mark_relational_stale_for_parent in
# app.services.database_access.contours).
RELATIONAL_METRIC_KEYS: tuple[str, ...] = ("n_children",)

# Number of images processed per commit in the batch appearance compute below (mirrors
# the backfill script's BATCH_SIZE, but per-IMAGE here since work is dominated by decode).
_APPEARANCE_BATCH_SIZE = 50

# Same batching idea for contextual metrics; no image decode here, but kept per-image so
# a large dataset does not hold one giant transaction open.
_CONTEXTUAL_BATCH_SIZE = 50

# Same batching idea for relational metrics (n_children); pure counting, no image decode.
_RELATIONAL_BATCH_SIZE = 50


def load_image_rgb(image: "Images") -> np.ndarray | None:
    """Load ``image.file_path`` from disk as an RGB ``uint8`` array.

    Mirrors the file-reading + fallback/logging pattern of
    ``app.services.database_access.datasets._native_image_dimensions``: reads the real
    file on disk (authoritative even if stored ``Images`` columns are stale) and logs a
    warning rather than raising if the file is missing or unreadable, so a single bad
    image cannot abort a whole batch.

    Args:
        image: An ``Images`` ORM row.

    Returns:
        An ``(H, W, 3)`` ``uint8`` RGB array, or ``None`` if the file could not be read.
    """
    file_path = getattr(image, "file_path", None)
    if not file_path or not os.path.exists(file_path):
        logger.warning("Image file not found at %s (image id=%s); cannot compute "
                       "appearance metrics.", file_path, getattr(image, "id", None))
        return None
    try:
        with PILImage.open(file_path) as img:
            return np.array(img.convert("RGB"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read image %s (id=%s) for appearance metrics: %s",
                       file_path, getattr(image, "id", None), exc)
        return None


def build_quant_context(image, contours: list[Contour], image_loader=None) -> QuantContext:
    """Build a :class:`QuantContext` for one image from an ``Images`` row and its contours.

    Args:
        image: An ``Images`` ORM row (or any object exposing width/height/scale_x/
            scale_y/unit). ``None`` falls back to a 1x1 px unit-scale context.
        contours: The :class:`Contour` schema objects to compute metrics for. Each must
            carry an ``id`` (they are the keys of the returned metric dicts).
        image_loader: Optional callable returning the image as an RGB array, forwarded to
            the context for appearance metrics (Step 3+). Left ``None`` for geometry.

    Returns:
        A :class:`QuantContext` scoped to the image.
    """
    if image is not None:
        width = int(image.width)
        height = int(image.height)
        scale_x = float(image.scale_x)
        scale_y = float(image.scale_y)
        unit = image.unit or "px"
    else:
        logger.warning("build_quant_context called without an image; using 1px unit scale.")
        width = height = 1
        scale_x = scale_y = 1.0
        unit = "px"
    return QuantContext(
        contours=contours,
        width=width,
        height=height,
        scale_x=scale_x,
        scale_y=scale_y,
        unit=unit,
        image_loader=image_loader,
    )


def quantify_contour_row(contour, image) -> QuantificationModel:
    """Recompute a stored contour's geometry in physical units.

    The database keeps NORMALIZED ([0, 1]) coordinates, so they are projected back to
    pixels with the image's dimensions before any geometry is computed - computing on the
    normalized values directly would anisotropically distort shapes on non-square images.
    Shared by the dual-write path in ``app.database.contours.save_contour_tree`` and
    ``scripts/backfill_contour_metrics`` so the legacy columns and the tall
    ``contour_metrics`` rows are always filled from exactly the same math.

    Args:
        contour: A ``Contours`` ORM row (normalized ``x`` / ``y`` coordinate lists).
        image: The ``Images`` row the contour belongs to (dimensions + physical scale).

    Returns:
        The recomputed :class:`QuantificationModel`, in the image's physical length unit.
    """
    x = contour.x if isinstance(contour.x, list) else list(contour.x or [])
    y = contour.y if isinstance(contour.y, list) else list(contour.y or [])
    if len(x) == 0:
        points_px = np.empty((0, 2), dtype=np.float64)
    else:
        points_px = np.stack([
            np.asarray(x, dtype=np.float64) * image.width,
            np.asarray(y, dtype=np.float64) * image.height,
        ], axis=-1)
    return QuantificationModel.from_contour(
        points_px,
        scale_x=image.scale_x,
        scale_y=image.scale_y,
        unit=image.unit or "px",
    )


def _upsert_metric_rows(
        session: Session,
        rows: list[dict],
        target_pairs: set[tuple[int, str]] | None = None,
) -> int:
    """Delete-then-insert the given metric rows keyed by (contour_id, metric_key).

    Deletes any existing rows for ``target_pairs`` first (so a metric whose ``value_dim``
    shrank does not leave stale higher components behind), then bulk-inserts the fresh
    rows. Idempotent: re-running yields the same table state.

    ``target_pairs`` defaults to the (contour_id, metric_key) pairs present in ``rows``,
    but callers whose metric OMITS some contours from its result (e.g. a CONTEXTUAL
    metric that has no meaningful value for an only-child contour, see
    ``metrics/contextual.py``) must pass the FULL set of (contour_id, metric_key) pairs
    that were considered, not just the ones with a value. Otherwise a contour that used
    to have a row (e.g. it had a neighbor before the neighbor was deleted) but is no
    longer in the returned dict would keep its stale row forever, corrupting aggregation.
    """
    pairs = target_pairs if target_pairs is not None else {(r["contour_id"], r["metric_key"]) for r in rows}
    if not pairs:
        return 0
    for contour_id, metric_key in pairs:
        session.query(ContourMetrics).filter(
            ContourMetrics.contour_id == contour_id,
            ContourMetrics.metric_key == metric_key,
        ).delete(synchronize_session=False)
    if rows:
        session.bulk_insert_mappings(ContourMetrics, rows)
    return len(rows)


def compute_and_store_metrics(
        session: Session,
        metric_keys: Iterable[str],
        contours: list[Contour],
        image,
        image_loader=None,
) -> int:
    """Compute the given metrics for ``contours`` and upsert them into ``contour_metrics``.

    Builds one :class:`QuantContext` for the image, runs each metric's ``compute_batch``,
    and writes one row per (contour, metric, component) with the unit resolved from the
    metric's unit kind and the image's length unit. Contours without an ``id`` are skipped
    (they cannot be keyed). Does not commit — the caller controls the transaction.

    A metric's ``compute_batch`` may OMIT a contour from its returned dict when it has no
    meaningful value for it (e.g. an only-child contour has no nearest neighbor, see
    ``metrics/contextual.py``). The delete step below always targets the FULL
    (contour, metric_key) set - every contour passed in, for every requested metric key -
    not just the ones present in the result, so an omitted contour's stale/pre-existing
    row (e.g. left over from when it still had a neighbor) is correctly removed rather
    than left to linger and corrupt aggregation.

    Args:
        session: The database session.
        metric_keys: Registry keys of the metrics to compute.
        contours: The contour schemas to compute for (must all belong to ``image``).
        image: The ``Images`` ORM row the contours belong to.
        image_loader: Optional RGB image loader for appearance metrics.

    Returns:
        The number of metric rows written.
    """
    metric_keys = list(metric_keys)
    contours = [c for c in contours if c.id is not None]
    if not contours or not metric_keys:
        return 0

    ctx = build_quant_context(image, contours, image_loader=image_loader)
    rows: list[dict] = []
    target_pairs: set[tuple[int, str]] = set()
    for metric_key in metric_keys:
        metric = get_metric(metric_key)
        unit = ctx.resolve_unit(metric.unit_kind)
        values_by_contour = metric.compute_batch(ctx)
        for contour in contours:
            target_pairs.add((contour.id, metric_key))
        for contour_id, values in values_by_contour.items():
            for component, value in enumerate(values):
                rows.append({
                    "contour_id": contour_id,
                    "metric_key": metric_key,
                    "component": component,
                    "value": float(value),
                    "unit": unit,
                    "stale": False,
                })
    return _upsert_metric_rows(session, rows, target_pairs=target_pairs)


def mark_appearance_stale(session: Session, contour_id: int) -> int:
    """Mark a contour's APPEARANCE-tier metric rows as ``stale=True``.

    Call this whenever a contour's geometry changes (its filled pixels may now differ),
    so :func:`compute_appearance_metrics_for_dataset` (with ``only_stale=True``) knows to
    recompute it. Geometry-tier rows are intentionally left untouched — those are
    recomputed synchronously on the same write, so they are never stale.

    Args:
        session: The database session (caller controls commit).
        contour_id: The contour whose appearance rows should be invalidated.

    Returns:
        The number of rows marked stale.
    """
    return session.query(ContourMetrics).filter(
        ContourMetrics.contour_id == contour_id,
        ContourMetrics.metric_key.in_(APPEARANCE_METRIC_KEYS),
    ).update({ContourMetrics.stale: True}, synchronize_session=False)


def _id_filter(contour_ids: "Iterable[int] | Select"):
    """Normalize a contour-id argument for use with ``IN``.

    Accepts either a concrete collection of ids or a ``SELECT`` that yields them. The
    subquery form matters on the bulk write path: resolving a sibling group in SQL keeps
    the group from being materialized into python once per saved contour, which is what
    turns a large mask import into quadratic work.

    Returns:
        A list of ids, a :class:`Select`, or ``None`` when there is nothing to match.
    """
    if isinstance(contour_ids, Select):
        return contour_ids
    ids = list(contour_ids)
    return ids or None


def mark_contextual_stale(session: Session, contour_ids: "Iterable[int] | Select") -> int:
    """Mark the CONTEXTUAL-tier metric rows of ``contour_ids`` as ``stale=True``.

    Low-level primitive shared by the group-invalidation helper in
    ``app.services.database_access.contours`` (``mark_contextual_stale_for_group``): that
    function is responsible for figuring out WHICH contours are affected (the whole
    parent group, see its docstring), this one just flips the flag for a given set of ids.

    Args:
        session: The database session (caller controls commit).
        contour_ids: The contours whose contextual rows should be invalidated, either as
            a collection of ids or as a ``SELECT`` yielding them (see :func:`_id_filter`).

    Returns:
        The number of rows marked stale.
    """
    id_filter = _id_filter(contour_ids)
    if id_filter is None:
        return 0
    return session.query(ContourMetrics).filter(
        ContourMetrics.contour_id.in_(id_filter),
        ContourMetrics.metric_key.in_(CONTEXTUAL_METRIC_KEYS),
    ).update({ContourMetrics.stale: True}, synchronize_session=False)


def mark_relational_stale(session: Session, contour_ids: Iterable[int]) -> int:
    """Mark the RELATIONAL-tier metric rows of ``contour_ids`` as ``stale=True``.

    Low-level primitive shared by the parent-invalidation helper in
    ``app.services.database_access.contours`` (``mark_relational_stale_for_parent``): that
    function decides WHICH contour (the parent that gained/lost a child) is affected, this
    one just flips the flag for a given set of ids.

    Note the difference from :func:`mark_contextual_stale`: contextual staleness fans out to
    a whole sibling GROUP (every member is a potential neighbor of every other), whereas a
    relational ``n_children`` change is confined to the single parent whose child set
    changed - so callers pass only the parent id(s), never the parent's siblings.

    Args:
        session: The database session (caller controls commit).
        contour_ids: The contours (parents) whose relational rows should be invalidated,
            either as a collection of ids or as a ``SELECT`` yielding them.

    Returns:
        The number of rows marked stale.
    """
    id_filter = _id_filter(contour_ids)
    if id_filter is None:
        return 0
    return session.query(ContourMetrics).filter(
        ContourMetrics.contour_id.in_(id_filter),
        ContourMetrics.metric_key.in_(RELATIONAL_METRIC_KEYS),
    ).update({ContourMetrics.stale: True}, synchronize_session=False)


def _images_needing_appearance_compute(
        session: Session,
        dataset_id: int,
        metric_keys: tuple[str, ...],
        only_stale: bool,
        image_ids: Iterable[int] | None,
):
    """Query (Contours, Images) rows scoped to the dataset that need appearance compute.

    When ``only_stale`` is True, a contour "needs compute" if, for at least one of
    ``metric_keys``, it has no fresh (``stale=False``) row — i.e. it either has no row at
    all yet (brand new contour) or its row was marked stale (geometry changed since the
    last compute). Expressed as an OR of per-key NOT EXISTS subqueries so it stays a
    single SQL query regardless of how many metric keys are requested. Ordered by image
    id so the caller can group consecutive rows into one :class:`QuantContext` per image.
    """
    from sqlalchemy import or_

    query = (
        session.query(Contours, Images)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
    )
    if image_ids is not None:
        query = query.filter(Images.id.in_(list(image_ids)))

    if only_stale:
        missing_or_stale_conditions = []
        for metric_key in metric_keys:
            has_fresh_row = (
                session.query(ContourMetrics.contour_id)
                .filter(
                    ContourMetrics.contour_id == Contours.id,
                    ContourMetrics.metric_key == metric_key,
                    ContourMetrics.stale.is_(False),
                )
                .exists()
            )
            missing_or_stale_conditions.append(~has_fresh_row)
        query = query.filter(or_(*missing_or_stale_conditions))

    return query.order_by(Images.id).all()


def compute_appearance_metrics_for_dataset(
        db: Session,
        dataset_id: int,
        metric_keys: Iterable[str] = APPEARANCE_METRIC_KEYS,
        only_stale: bool = True,
        image_ids: Iterable[int] | None = None,
) -> int:
    """Compute and store appearance-tier metrics for a dataset, grouped per image.

    Appearance metrics need the decoded image pixels, which is too expensive for the
    synchronous contour write path (``save_contour_tree``), so this is the batch/lazy
    entry point: it groups contours by image (one DB round trip via a join, then a
    dict grouping), and for each image builds exactly one
    :class:`~iquana_toolbox.quantification.context.QuantContext` (via
    :func:`build_quant_context` with :func:`load_image_rgb` as the loader) and calls
    :func:`compute_and_store_metrics`, so the image is decoded at most once per image
    regardless of how many metrics or contours it has (see ``QuantContext.image``).

    Commits are batched every :data:`_APPEARANCE_BATCH_SIZE` images so a large dataset
    does not hold one giant transaction open.

    Args:
        db: The database session.
        dataset_id: The dataset to compute for.
        metric_keys: Registry keys of the appearance metrics to compute. Defaults to
            all of :data:`APPEARANCE_METRIC_KEYS`.
        only_stale: If True (default), skip contours that already have a fresh
            (non-stale) row for every requested metric key. New contours (no rows yet)
            and contours whose rows were marked stale (via :func:`mark_appearance_stale`)
            are always recomputed. Set False to force recomputation of everything in
            scope (e.g. after adding a new appearance metric).
        image_ids: Optional subset of image ids to restrict the computation to
            (e.g. to recompute just the images touched by one edit).

    Returns:
        The total number of ``contour_metrics`` rows written.
    """
    metric_keys = tuple(metric_keys)
    if not metric_keys:
        return 0

    rows = _images_needing_appearance_compute(db, dataset_id, metric_keys, only_stale, image_ids)
    if not rows:
        return 0

    contours_by_image: dict[int, list] = {}
    image_by_id: dict[int, Images] = {}
    for contour_db, image_db in rows:
        image_by_id[image_db.id] = image_db
        contours_by_image.setdefault(image_db.id, []).append(Contour.from_db(contour_db))

    total_rows = 0
    images_since_commit = 0
    for image_id, contour_schemas in contours_by_image.items():
        image = image_by_id[image_id]
        total_rows += compute_and_store_metrics(
            db, metric_keys, contour_schemas, image,
            image_loader=lambda img=image: load_image_rgb(img),
        )
        images_since_commit += 1
        if images_since_commit >= _APPEARANCE_BATCH_SIZE:
            db.commit()
            images_since_commit = 0

    if images_since_commit > 0:
        db.commit()

    logger.info("Computed appearance metrics for dataset %s: %d images, %d rows (only_stale=%s).",
               dataset_id, len(contours_by_image), total_rows, only_stale)
    return total_rows


def _contours_needing_contextual_compute(
        session: Session,
        dataset_id: int,
        metric_keys: tuple[str, ...],
        only_stale: bool,
        image_ids: Iterable[int] | None,
):
    """Query (Contours, Images) rows scoped to the dataset that need contextual compute.

    CONTEXTUAL metrics are RELATIONAL (see ``metrics/contextual.py``): a contour's
    nearest-neighbour distance depends on where its same-parent siblings are, so adding,
    moving or deleting ONE contour changes the correct value for every sibling in its
    parent group, not just itself. This function therefore does the same missing-or-stale
    query as ``_images_needing_appearance_compute`` and then EXPANDS the result to the
    full parent group of every matched contour: for each ``(mask_id, parent_id)`` pair
    with at least one stale/missing contour, every contour sharing that ``(mask_id,
    parent_id)`` is included, because the KDTree for that group must be rebuilt anyway
    once any member of the group changes.

    Root-level contours (``parent_id is None``) are grouped by ``mask_id`` alone (see
    ``metrics/contextual.py``: all root contours of an image are siblings of each other).

    Args:
        session: The database session.
        dataset_id: The dataset to scope the query to.
        metric_keys: Registry keys of the contextual metrics to compute.
        only_stale: If True, restrict to contours missing a fresh row for at least one of
            ``metric_keys`` (before group expansion). If False, every contour in scope
            "needs" compute (no filtering) - the expansion step is then a no-op.
        image_ids: Optional subset of image ids to scope to.

    Returns:
        A list of ``(Contours, Images)`` rows for every contour in every affected parent
        group, ordered by image id (so the caller can group into one QuantContext per
        image, same as the appearance path).
    """
    from sqlalchemy import or_

    base_query = (
        session.query(Contours, Images)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
    )
    if image_ids is not None:
        base_query = base_query.filter(Images.id.in_(list(image_ids)))

    if not only_stale:
        return base_query.order_by(Images.id).all()

    seed_query = base_query
    missing_or_stale_conditions = []
    for metric_key in metric_keys:
        has_fresh_row = (
            session.query(ContourMetrics.contour_id)
            .filter(
                ContourMetrics.contour_id == Contours.id,
                ContourMetrics.metric_key == metric_key,
                ContourMetrics.stale.is_(False),
            )
            .exists()
        )
        missing_or_stale_conditions.append(~has_fresh_row)
    seed_query = seed_query.filter(or_(*missing_or_stale_conditions))

    seed_rows = seed_query.all()
    if not seed_rows:
        return []

    # Expand every seed contour to its full parent group: (mask_id, parent_id) pairs.
    group_keys = {(contour_db.mask_id, contour_db.parent_id) for contour_db, _image_db in seed_rows}

    group_filters = [
        (Contours.mask_id == mask_id) & (
            Contours.parent_id.is_(None) if parent_id is None else Contours.parent_id == parent_id
        )
        for mask_id, parent_id in group_keys
    ]
    expanded_query = base_query.filter(or_(*group_filters))
    return expanded_query.order_by(Images.id).all()


def compute_contextual_metrics_for_dataset(
        db: Session,
        dataset_id: int,
        metric_keys: Iterable[str] = CONTEXTUAL_METRIC_KEYS,
        only_stale: bool = True,
        image_ids: Iterable[int] | None = None,
) -> int:
    """Compute and store contextual-tier metrics for a dataset, grouped per image.

    Mirrors :func:`compute_appearance_metrics_for_dataset`'s per-image grouping (all of a
    parent's children live in the same image/mask, so grouping by image is correct: the
    KDTree in ``metrics/contextual.py`` is built per PARENT group, which is always a
    subset of one image's contours). No image pixels are needed here, so no
    ``image_loader`` is passed.

    Critical difference from appearance: contextual metrics are RELATIONAL, so "needs
    compute" is expanded from individual stale/missing contours to their FULL parent
    group by :func:`_contours_needing_contextual_compute` before this function ever sees
    them - by the time this function groups rows by image, the group-expansion has
    already happened.

    Commits are batched every :data:`_CONTEXTUAL_BATCH_SIZE` images.

    Args:
        db: The database session.
        dataset_id: The dataset to compute for.
        metric_keys: Registry keys of the contextual metrics to compute. Defaults to all
            of :data:`CONTEXTUAL_METRIC_KEYS`.
        only_stale: If True (default), skip parent groups that are entirely fresh (no
            missing/stale row for any of their members). Set False to force
            recomputation of everything in scope.
        image_ids: Optional subset of image ids to restrict the computation to.

    Returns:
        The total number of ``contour_metrics`` rows written.
    """
    metric_keys = tuple(metric_keys)
    if not metric_keys:
        return 0

    rows = _contours_needing_contextual_compute(db, dataset_id, metric_keys, only_stale, image_ids)
    if not rows:
        return 0

    contours_by_image: dict[int, list] = {}
    image_by_id: dict[int, Images] = {}
    for contour_db, image_db in rows:
        image_by_id[image_db.id] = image_db
        contours_by_image.setdefault(image_db.id, []).append(Contour.from_db(contour_db))

    total_rows = 0
    images_since_commit = 0
    for image_id, contour_schemas in contours_by_image.items():
        image = image_by_id[image_id]
        total_rows += compute_and_store_metrics(db, metric_keys, contour_schemas, image)
        images_since_commit += 1
        if images_since_commit >= _CONTEXTUAL_BATCH_SIZE:
            db.commit()
            images_since_commit = 0

    if images_since_commit > 0:
        db.commit()

    logger.info("Computed contextual metrics for dataset %s: %d images, %d rows (only_stale=%s).",
               dataset_id, len(contours_by_image), total_rows, only_stale)
    return total_rows


def _contours_needing_relational_compute(
        session: Session,
        dataset_id: int,
        metric_keys: tuple[str, ...],
        only_stale: bool,
        image_ids: Iterable[int] | None,
):
    """Query (Contours, Images) rows scoped to the dataset that need relational compute.

    RELATIONAL metrics (``n_children``) are computed by counting, per contour, how many
    OTHER contours name it as their parent. That count is derived from the WHOLE image's
    contours (a parent and all its children live in the same image/mask), so the
    :class:`QuantContext` for an image must contain every contour of that image to count
    children correctly. This function therefore does the same missing-or-stale seed query
    as the appearance/contextual paths, then EXPANDS the result to the full IMAGE of every
    matched contour (whereas the contextual path expands only to the parent GROUP): if any
    contour in an image is stale/missing for a relational metric, the whole image is
    recomputed, so the counts are always taken over the complete contour set.

    Args:
        session: The database session.
        dataset_id: The dataset to scope the query to.
        metric_keys: Registry keys of the relational metrics to compute.
        only_stale: If True, restrict to images that contain at least one contour missing a
            fresh row for one of ``metric_keys`` (before image expansion). If False, every
            contour in scope needs compute (no filtering) - the expansion is a no-op.
        image_ids: Optional subset of image ids to scope to.

    Returns:
        A list of ``(Contours, Images)`` rows for every contour in every affected image,
        ordered by image id (so the caller can group into one QuantContext per image).
    """
    from sqlalchemy import or_

    base_query = (
        session.query(Contours, Images)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id)
    )
    if image_ids is not None:
        base_query = base_query.filter(Images.id.in_(list(image_ids)))

    if not only_stale:
        return base_query.order_by(Images.id).all()

    seed_query = base_query
    missing_or_stale_conditions = []
    for metric_key in metric_keys:
        has_fresh_row = (
            session.query(ContourMetrics.contour_id)
            .filter(
                ContourMetrics.contour_id == Contours.id,
                ContourMetrics.metric_key == metric_key,
                ContourMetrics.stale.is_(False),
            )
            .exists()
        )
        missing_or_stale_conditions.append(~has_fresh_row)
    seed_query = seed_query.filter(or_(*missing_or_stale_conditions))

    seed_rows = seed_query.all()
    if not seed_rows:
        return []

    # Expand every seed contour to its full IMAGE so children are countable in the context.
    affected_image_ids = {image_db.id for _contour_db, image_db in seed_rows}
    expanded_query = base_query.filter(Images.id.in_(affected_image_ids))
    return expanded_query.order_by(Images.id).all()


def compute_relational_metrics_for_dataset(
        db: Session,
        dataset_id: int,
        metric_keys: Iterable[str] = RELATIONAL_METRIC_KEYS,
        only_stale: bool = True,
        image_ids: Iterable[int] | None = None,
) -> int:
    """Compute and store relational-tier metrics (``n_children``) for a dataset, per image.

    Mirrors :func:`compute_contextual_metrics_for_dataset`'s per-image grouping, but no
    image pixels are needed (no ``image_loader``) and the "needs compute" set is expanded to
    the full IMAGE rather than the parent group (see
    :func:`_contours_needing_relational_compute`): ``n_children`` counts a contour's
    children, which requires the whole image's contours in the context.

    Commits are batched every :data:`_RELATIONAL_BATCH_SIZE` images.

    Args:
        db: The database session.
        dataset_id: The dataset to compute for.
        metric_keys: Registry keys of the relational metrics to compute. Defaults to all of
            :data:`RELATIONAL_METRIC_KEYS`.
        only_stale: If True (default), skip images entirely fresh for the requested metrics.
            Set False to force recomputation of everything in scope.
        image_ids: Optional subset of image ids to restrict the computation to.

    Returns:
        The total number of ``contour_metrics`` rows written.
    """
    metric_keys = tuple(metric_keys)
    if not metric_keys:
        return 0

    rows = _contours_needing_relational_compute(db, dataset_id, metric_keys, only_stale, image_ids)
    if not rows:
        return 0

    contours_by_image: dict[int, list] = {}
    image_by_id: dict[int, Images] = {}
    for contour_db, image_db in rows:
        image_by_id[image_db.id] = image_db
        contours_by_image.setdefault(image_db.id, []).append(Contour.from_db(contour_db))

    total_rows = 0
    images_since_commit = 0
    for image_id, contour_schemas in contours_by_image.items():
        image = image_by_id[image_id]
        total_rows += compute_and_store_metrics(db, metric_keys, contour_schemas, image)
        images_since_commit += 1
        if images_since_commit >= _RELATIONAL_BATCH_SIZE:
            db.commit()
            images_since_commit = 0

    if images_since_commit > 0:
        db.commit()

    logger.info("Computed relational metrics for dataset %s: %d images, %d rows (only_stale=%s).",
               dataset_id, len(contours_by_image), total_rows, only_stale)
    return total_rows

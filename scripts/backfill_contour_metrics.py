"""Backfill / repair the quantification stores for existing contours.

For every contour (optionally scoped to one dataset) this recomputes the geometry, then:
  * UPDATEs the legacy ``contours`` columns (area / perimeter / circularity / diameter) in
    the image's PHYSICAL unit, and
  * upserts the tall ``contour_metrics`` rows for area / perimeter / circularity /
    max_diameter PIXEL-native (image scale ignored; applied at read time).

Run this once after upgrading to the pixel-native quantification storage so any
pre-existing ``contour_metrics`` rows (which were stored in physical units) are rewritten
in pixels; otherwise the dataset summary would mislabel those stale physical values.

Idempotent (safe to re-run), batches commits, logs progress and prints a summary.

Usage (cwd = backend/):
    backend/.venv/Scripts/python.exe scripts/backfill_contour_metrics.py [--dataset-id N] [--dry-run]
"""
import argparse
import logging
import os
import sys

# Make ``import app...`` and ``import config`` work regardless of where python is invoked
# from: put the backend project root (parent of this scripts/ dir) on sys.path.
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from iquana_toolbox.schemas.database.quantification import QuantificationModel  # noqa: E402

from app.database import get_context_session  # noqa: E402
from app.database.contours import Contours  # noqa: E402
from app.database.datasets import Datasets  # noqa: E402
from app.database.images import Images  # noqa: E402
from app.database.masks import Masks  # noqa: E402
from app.services.quantification import (  # noqa: E402
    GEOMETRY_METRIC_KEYS,
    compute_contextual_metrics_for_dataset,
    quantify_contour_row,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_contour_metrics")

BATCH_SIZE = 500


def _iter_contour_rows(session, dataset_id: int | None):
    """Yield (Contours, Images) rows, optionally scoped to one dataset."""
    query = (
        session.query(Contours, Images)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
    )
    if dataset_id is not None:
        query = query.filter(Images.dataset_id == dataset_id)
    return query.yield_per(BATCH_SIZE)


def _recompute(contour: Contours, image: Images) -> QuantificationModel:
    """Recompute the quantification for a DB contour in pixel space (scaled to units).

    Delegates to the shared helper so this repair path and the synchronous dual-write in
    ``save_contour_tree`` can never drift apart.
    """
    return quantify_contour_row(contour, image)


def backfill(dataset_id: int | None = None, dry_run: bool = False) -> dict[str, int]:
    """Recompute and re-store metrics for all in-scope contours.

    Recomputes the GEOMETRY tier per contour (legacy physical columns + pixel-native tall
    rows) and then force-recomputes the CONTEXTUAL tier (nn_distance / mean_knn_distance) to
    pixels for each in-scope dataset - contextual metrics are the only other tier that used
    to be stored in physical units, so both must be rewritten for a complete pixel-native
    migration. Appearance / relational metrics are unitless (scale-independent) and need no
    migration.

    Returns a summary dict: processed / updated / skipped_degenerate / contextual_rows.
    """
    # Import here so create_all has already registered contour_metrics.
    from app.database.contour_metrics import ContourMetrics
    from app.database.contours import _resolve_metric_unit

    processed = updated = skipped_degenerate = contextual_rows = 0

    with get_context_session() as session:
        for contour, image in _iter_contour_rows(session, dataset_id):
            processed += 1
            quant = _recompute(contour, image)

            # A contour whose recomputed area is 0 is degenerate (collinear / <3 points).
            if quant.area == 0.0 and quant.perimeter == 0.0:
                skipped_degenerate += 1

            if not dry_run:
                # 1. Legacy columns: PHYSICAL units (image scale applied).
                contour.area = quant.area
                contour.perimeter = quant.perimeter
                contour.circularity = quant.circularity
                contour.diameter = quant.max_diameter

                # 2. Tall table: PIXEL-native (image scale ignored), delete-then-insert per
                #    metric key -> idempotent. The physical scale is applied at read time.
                quant_px = quantify_contour_row(contour, image, pixel=True)
                values_by_key = {
                    "area": quant_px.area,
                    "perimeter": quant_px.perimeter,
                    "circularity": quant_px.circularity,
                    "max_diameter": quant_px.max_diameter,
                }
                for metric_key in GEOMETRY_METRIC_KEYS:
                    session.query(ContourMetrics).filter(
                        ContourMetrics.contour_id == contour.id,
                        ContourMetrics.metric_key == metric_key,
                        ContourMetrics.component == 0,
                    ).delete(synchronize_session=False)
                    session.add(ContourMetrics(
                        contour_id=contour.id,
                        metric_key=metric_key,
                        component=0,
                        value=float(values_by_key[metric_key]),
                        unit=_resolve_metric_unit(metric_key, "px"),
                        stale=False,
                    ))
            updated += 0 if dry_run else 1

            if processed % BATCH_SIZE == 0:
                if not dry_run:
                    session.commit()
                logger.info("Processed %d contours (updated=%d, degenerate=%d)...",
                            processed, updated, skipped_degenerate)

        if not dry_run:
            session.commit()

        # Contextual tier: recompute to pixels for every in-scope dataset (only_stale=False
        # forces a rewrite of any lingering physical-unit rows). Skipped on a dry run.
        if not dry_run:
            if dataset_id is not None:
                scoped_ids = [dataset_id]
            else:
                scoped_ids = [d.id for d in session.query(Datasets.id).all()]
            for did in scoped_ids:
                contextual_rows += compute_contextual_metrics_for_dataset(
                    session, did, only_stale=False
                )

    logger.info(
        "Backfill complete. processed=%d updated=%d skipped_degenerate=%d "
        "contextual_rows=%d dry_run=%s",
        processed, updated, skipped_degenerate, contextual_rows, dry_run,
    )
    return {
        "processed": processed,
        "updated": updated,
        "skipped_degenerate": skipped_degenerate,
        "contextual_rows": contextual_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill contour_metrics and legacy quantification columns.")
    parser.add_argument("--dataset-id", type=int, default=None,
                        help="Only backfill contours of this dataset. Default: all datasets.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and report but do not write to the database.")
    args = parser.parse_args()

    summary = backfill(dataset_id=args.dataset_id, dry_run=args.dry_run)
    print(f"Backfill summary: {summary}")


if __name__ == "__main__":
    main()

"""Backfill / repair the quantification stores for existing contours.

For every contour (optionally scoped to one dataset) this recomputes the geometry in
pixel space with the Step-1 code, then:
  * UPDATEs the legacy ``contours`` columns (area / perimeter / circularity / diameter), and
  * upserts the tall ``contour_metrics`` rows for area / perimeter / circularity / max_diameter.

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

import numpy as np  # noqa: E402

from iquana_toolbox.schemas.database.quantification import QuantificationModel  # noqa: E402

from app.database import get_context_session  # noqa: E402
from app.database.contours import Contours  # noqa: E402
from app.database.images import Images  # noqa: E402
from app.database.masks import Masks  # noqa: E402
from app.services.quantification import GEOMETRY_METRIC_KEYS  # noqa: E402

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
    """Recompute the quantification for a DB contour in pixel space (scaled to units)."""
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


def backfill(dataset_id: int | None = None, dry_run: bool = False) -> dict[str, int]:
    """Recompute and re-store metrics for all in-scope contours.

    Returns a summary dict: processed / updated / skipped_degenerate.
    """
    # Import here so create_all has already registered contour_metrics.
    from app.database.contour_metrics import ContourMetrics
    from app.database.contours import _resolve_metric_unit

    processed = updated = skipped_degenerate = 0

    with get_context_session() as session:
        for contour, image in _iter_contour_rows(session, dataset_id):
            processed += 1
            quant = _recompute(contour, image)

            # A contour whose recomputed area is 0 is degenerate (collinear / <3 points).
            if quant.area == 0.0 and quant.perimeter == 0.0:
                skipped_degenerate += 1

            if not dry_run:
                # 1. Legacy columns.
                contour.area = quant.area
                contour.perimeter = quant.perimeter
                contour.circularity = quant.circularity
                contour.diameter = quant.max_diameter

                # 2. Tall table (delete-then-insert per metric key -> idempotent).
                unit = image.unit or "px"
                values_by_key = {
                    "area": quant.area,
                    "perimeter": quant.perimeter,
                    "circularity": quant.circularity,
                    "max_diameter": quant.max_diameter,
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
                        unit=_resolve_metric_unit(metric_key, unit),
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

    logger.info("Backfill complete. processed=%d updated=%d skipped_degenerate=%d dry_run=%s",
                processed, updated, skipped_degenerate, dry_run)
    return {
        "processed": processed,
        "updated": updated,
        "skipped_degenerate": skipped_degenerate,
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

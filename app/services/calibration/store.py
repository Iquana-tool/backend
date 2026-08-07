"""Row-level persistence for calibrations. No knowledge of any particular kind.

Split out from ``service`` so the orchestration layer (which does know about
kinds, pipelines and staleness) reads as policy and this reads as plumbing.
Nothing here commits — the caller owns the transaction, matching the convention
in ``app.services.quantification``.
"""
from datetime import datetime, timezone
from logging import getLogger
from typing import Iterable

from sqlalchemy.orm import Session

from app.database.contour_metrics import ContourMetrics
from app.database.contours import Contours
from app.database.image_calibrations import ImageCalibrations
from app.database.masks import Masks

logger = getLogger(__name__)


def get_row(db: Session, image_id: int, kind: str) -> ImageCalibrations | None:
    """The current calibration row of ``kind`` for ``image_id``, if any."""
    return (
        db.query(ImageCalibrations)
        .filter_by(image_id=image_id, kind=kind)
        .first()
    )


def list_rows(db: Session, image_id: int) -> list[ImageCalibrations]:
    """Every calibration row stored for an image, unordered."""
    return db.query(ImageCalibrations).filter_by(image_id=image_id).all()


def upsert(
        db: Session,
        image_id: int,
        kind: str,
        params: dict,
        source: str,
        username: str | None = None,
) -> ImageCalibrations:
    """Create or replace the calibration of ``kind`` for ``image_id``.

    ``params`` must already have been normalised by the kind's descriptor — this
    layer does no validation. Does not commit.
    """
    row = get_row(db, image_id, kind)
    if row is None:
        row = ImageCalibrations(
            image_id=image_id,
            kind=kind,
            params=params,
            source=source,
            created_by=username,
        )
        db.add(row)
        return row

    row.params = params
    row.source = source
    row.created_by = username
    # `onupdate` only fires for UPDATE statements SQLAlchemy builds from a dirty
    # instance; setting it here keeps the value right even when params happen to
    # be identical and no column actually changes.
    row.updated_at = datetime.now(timezone.utc)
    return row


def delete_row(db: Session, image_id: int, kind: str) -> bool:
    """Remove a calibration. Returns whether a row was actually deleted."""
    row = get_row(db, image_id, kind)
    if row is None:
        return False
    db.delete(row)
    return True


def mark_metrics_stale_for_image(
        db: Session,
        image_id: int,
        metric_keys: Iterable[str],
) -> int:
    """Flag every stored metric row of ``metric_keys`` on an image for recompute.

    A calibration change does not rewrite any measurement — it marks the affected
    ones stale and lets the existing batch recompute (``only_stale=True``) pick
    them up. That keeps a recalibration cheap regardless of how many contours the
    image carries, and means the numbers on screen are never silently wrong: they
    are either fresh or flagged.

    Args:
        db: SQLAlchemy session (caller commits).
        image_id: The image whose contours' metrics should be invalidated.
        metric_keys: Registry keys to invalidate, from the kind's descriptor.

    Returns:
        The number of metric rows marked stale.
    """
    metric_keys = list(metric_keys)
    if not metric_keys:
        return 0

    contour_ids = [
        row[0]
        for row in (
            db.query(Contours.id)
            .join(Masks, Masks.id == Contours.mask_id)
            .filter(Masks.image_id == image_id)
            .all()
        )
    ]
    if not contour_ids:
        return 0

    updated = (
        db.query(ContourMetrics)
        .filter(
            ContourMetrics.contour_id.in_(contour_ids),
            ContourMetrics.metric_key.in_(metric_keys),
        )
        .update({ContourMetrics.stale: True}, synchronize_session=False)
    )
    logger.debug("Marked %d metric row(s) stale on image %d for keys %s.",
                 updated, image_id, metric_keys)
    return updated

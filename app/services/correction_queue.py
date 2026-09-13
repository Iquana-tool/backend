"""Building correction queues: which sent-back annotations an annotator has to
address, in what order.

The mirror image of ``app.services.review_queue``. Where the review queue collects
*pending* contours for a reviewer, the correction queue collects *open rejections*
(the reviewer's send-backs) for the annotator who has to rework them. An item points
at the mask — and, for a per-object complaint, the contour — to load in the editor,
and carries the reviewer's reason and note so the annotator knows what to change.

Every open rejection in the dataset qualifies (the send-back is dataset-wide work,
not scoped to one author — same visibility as ``RejectionBanner``). Ordering is a
flat oldest/newest-first sort, not a scoring registry: corrections are a backlog to
clear, not a ranked review pass. Items are grouped so an image's rejections stay
consecutive, keeping the editor's per-image session and caches warm across them.

Queues are snapshots, not reservations: nothing is locked, and an item someone else
resolves mid-session simply no-ops when acted on.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.database.images import Images
from app.database.masks import Masks
from app.database.rejections import AnnotationRejections
from app.schemas.review import (
    REJECTION_REASON_LABELS,
    CorrectionQueueItem,
    CorrectionQueueRead,
    CorrectionQueueRequest,
    CorrectionSortOrder,
    CorrectionSummary,
    RejectionReason,
)

#: Sort sentinel for a rejection with no timestamp (never expected on an open row).
_MIN_TIME = datetime.min


def _open_rejections_query(dataset_id: int, db: Session):
    """Open rejections of the dataset, with each one's image id.

    Joined through ``Masks`` to ``Images`` both to scope to the dataset and to
    surface the image id: a rejection names a mask, and the editor loads by image.
    """
    return (
        db.query(
            AnnotationRejections.id,
            AnnotationRejections.mask_id,
            AnnotationRejections.contour_id,
            AnnotationRejections.reason,
            AnnotationRejections.note,
            AnnotationRejections.created_by,
            AnnotationRejections.created_at,
            Masks.image_id.label("image_id"),
        )
        .join(Masks, Masks.id == AnnotationRejections.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Images.dataset_id == dataset_id,
                AnnotationRejections.resolved_at.is_(None))
    )


def summarize(dataset_id: int, db: Session) -> CorrectionSummary:
    """The numbers behind "There are x instances sent back for correction"."""
    rows = _open_rejections_query(dataset_id, db).all()
    return CorrectionSummary(
        open_rejections=len(rows),
        affected_instances=len({row.contour_id for row in rows
                                if row.contour_id is not None}),
        affected_images=len({row.image_id for row in rows}),
    )


def build_queue(dataset_id: int, request: CorrectionQueueRequest,
                db: Session) -> CorrectionQueueRead:
    """Build the ordered work list for one correction session."""
    rows = _open_rejections_query(dataset_id, db).all()

    if request.reasons:
        wanted = {reason.value for reason in request.reasons}
        rows = [row for row in rows if row.reason in wanted]

    newest_first = request.order is CorrectionSortOrder.NEWEST

    # Group an image's rejections together, but order the images themselves by the
    # age of their leading item so "oldest first" still walks the backlog in age
    # order across images (not by image id). `created_at` is None-guarded for
    # safety — an open rejection always has one — with a stable minimum sentinel.
    def age(row):
        return row.created_at or _MIN_TIME

    image_key: dict[int, object] = {}
    for row in rows:
        current = image_key.get(row.image_id)
        candidate = age(row)
        if current is None or (candidate > current if newest_first else candidate < current):
            image_key[row.image_id] = candidate

    rows.sort(key=lambda row: (image_key[row.image_id], row.image_id, age(row)),
              reverse=newest_first)

    items = [
        CorrectionQueueItem(
            rejection_id=row.id,
            mask_id=row.mask_id,
            image_id=row.image_id,
            contour_id=row.contour_id,
            reason=RejectionReason(row.reason),
            reason_label=REJECTION_REASON_LABELS[RejectionReason(row.reason)],
            note=row.note,
            created_by=row.created_by,
            created_at=row.created_at,
        )
        for row in rows
    ]
    return CorrectionQueueRead(order=request.order, total=len(items), items=items)

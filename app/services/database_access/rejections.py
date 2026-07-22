"""Creating, listing and resolving review rejections."""
from datetime import datetime, timezone
from logging import getLogger

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.database.contours import Contours
from app.database.masks import Masks
from app.database.rejections import AnnotationRejections
from app.schemas.review import (
    REJECTION_REASON_LABELS,
    RejectionCreate,
    RejectionRead,
    RejectionReason,
    RejectionReasonOption,
)

logger = getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def reason_options() -> list[RejectionReasonOption]:
    """The reviewer's reason dropdown, wording included.

    Served from the backend so the frontend does not have to keep its own copy of
    the vocabulary in sync.
    """
    return [
        RejectionReasonOption(
            value=reason,
            label=REJECTION_REASON_LABELS[reason],
            requires_note=reason is RejectionReason.OTHER,
        )
        for reason in RejectionReason
    ]


def to_read(rejection: AnnotationRejections) -> RejectionRead:
    reason = RejectionReason(rejection.reason)
    return RejectionRead(
        id=rejection.id,
        mask_id=rejection.mask_id,
        contour_id=rejection.contour_id,
        reason=reason,
        reason_label=REJECTION_REASON_LABELS[reason],
        note=rejection.note,
        created_by=rejection.created_by,
        created_at=_as_utc(rejection.created_at),
        resolved_at=_as_utc(rejection.resolved_at),
        resolved_by=rejection.resolved_by,
    )


async def reject(mask_id: int,
                 body: RejectionCreate,
                 username: str,
                 db: Session) -> AnnotationRejections:
    """Send a mask (or one of its contours) back to the annotator.

    Rejecting also clears `fully_annotated`, so the mask leaves the reviewer's
    queue and reappears in the annotator's; without that the same mask would keep
    showing up as awaiting review.
    """
    mask = db.query(Masks).filter_by(id=mask_id).first()
    if mask is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mask not found.")

    if body.contour_id is not None:
        belongs = (
            db.query(Contours.id)
            .filter(Contours.id == body.contour_id, Contours.mask_id == mask_id)
            .scalar()
        )
        if belongs is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Contour {body.contour_id} does not belong to mask {mask_id}.")

    rejection = AnnotationRejections(
        mask_id=mask_id,
        contour_id=body.contour_id,
        reason=body.reason.value,
        note=(body.note or "").strip() or None,
        created_by=username,
        created_at=_utcnow(),
    )
    db.add(rejection)
    mask.fully_annotated = False
    db.commit()
    db.refresh(rejection)
    return rejection


async def resolve(rejection_id: int, username: str, db: Session) -> AnnotationRejections:
    """Mark one rejection as dealt with. Resolving is idempotent."""
    rejection = db.query(AnnotationRejections).filter_by(id=rejection_id).first()
    if rejection is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rejection not found.")
    if rejection.resolved_at is None:
        rejection.resolved_at = _utcnow()
        rejection.resolved_by = username
        db.commit()
        db.refresh(rejection)
    return rejection


async def resolve_all_for_mask(mask_id: int, username: str, db: Session) -> int:
    """Clear every open rejection on a mask; returns how many were resolved."""
    open_rejections = (
        db.query(AnnotationRejections)
        .filter(AnnotationRejections.mask_id == mask_id,
                AnnotationRejections.resolved_at.is_(None))
        .all()
    )
    now = _utcnow()
    for rejection in open_rejections:
        rejection.resolved_at = now
        rejection.resolved_by = username
    if open_rejections:
        db.commit()
    return len(open_rejections)


async def list_for_mask(mask_id: int, db: Session, open_only: bool = False) -> list[RejectionRead]:
    """Rejections recorded against a mask, newest first."""
    query = db.query(AnnotationRejections).filter(AnnotationRejections.mask_id == mask_id)
    if open_only:
        query = query.filter(AnnotationRejections.resolved_at.is_(None))
    rejections = query.order_by(AnnotationRejections.created_at.desc()).all()
    return [to_read(rejection) for rejection in rejections]


async def count_open_for_mask(mask_id: int, db: Session) -> int:
    return (
        db.query(AnnotationRejections)
        .filter(AnnotationRejections.mask_id == mask_id,
                AnnotationRejections.resolved_at.is_(None))
        .count()
    )

"""Review rejections: how a reviewer sends annotation work back to its author.

A rejection always names a mask (so "what is still open on this image?" is one
indexed query) and optionally a contour. A row with `contour_id IS NULL` is a
mask-level complaint such as "objects are missing"; a row with a contour is about
that one object, e.g. "bad outline".

Rejections are resolved rather than deleted, so the history of a mask survives for
the per-user metrics planned for the user study.
"""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import relationship

from app.database import database
from app.schemas.review import RejectionReason


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnnotationRejections(database):
    """One reviewer complaint against a mask or a single contour."""

    __tablename__ = "annotation_rejections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mask_id = Column(Integer, ForeignKey("masks.id", ondelete="CASCADE"), nullable=False, index=True)
    # NULL => the rejection is about the mask as a whole.
    contour_id = Column(Integer, ForeignKey("contours.id", ondelete="CASCADE"), nullable=True)
    reason = Column(String(32), nullable=False, default=RejectionReason.OTHER.value)
    note = Column(String(1000), nullable=True)
    created_by = Column(String, ForeignKey("users.username", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String, ForeignKey("users.username", ondelete="SET NULL"), nullable=True)

    mask = relationship("Masks", back_populates="rejections")

    @property
    def is_open(self) -> bool:
        return self.resolved_at is None


# Open rejections are looked up on every mask status computation, so make the
# "unresolved for this mask" lookup a covered index scan.
Index(
    "ix_annotation_rejections_open",
    AnnotationRejections.mask_id,
    AnnotationRejections.resolved_at,
)

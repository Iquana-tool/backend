"""Completeness checks: a reviewer's answer to "is everything on this image annotated?".

Reviewing objects one at a time says nothing about the objects nobody drew, and a
measurement such as polyp density is only right when every polyp is outlined. So at
the end of an image the reviewer answers per label: complete, incomplete or unsure.
A row may also be about the image's relationships rather than a label.

Rows are only ever added. A later answer from the same reviewer supersedes an
earlier one, and an undo marks a row withdrawn instead of deleting it, so the
history stays available for the user studies and for a model-quality report.
"""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import relationship

from app.database import database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CompletenessChecks(database):
    """One reviewer's answer about one label (or the relationships) on one image."""

    __tablename__ = "completeness_checks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mask_id = Column(Integer, ForeignKey("masks.id", ondelete="CASCADE"), nullable=False, index=True)
    # "label" rows name a label; a "relationships" row has none.
    item = Column(String(16), nullable=False)
    label_id = Column(Integer, ForeignKey("labels.id", ondelete="CASCADE"), nullable=True)
    answer = Column(String(16), nullable=False)
    note = Column(String(1000), nullable=True)
    created_by = Column(String, ForeignKey("users.username", ondelete="SET NULL", onupdate="CASCADE"),
                       nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    # The send-back an "incomplete" answer created, so the two can be traced and undone together.
    rejection_id = Column(Integer, ForeignKey("annotation_rejections.id", ondelete="SET NULL"),
                          nullable=True)
    withdrawn_at = Column(DateTime, nullable=True)

    rejection = relationship("AnnotationRejections")


# "The current answers on this image" filters on mask and skips withdrawn rows.
Index("ix_completeness_checks_current", CompletenessChecks.mask_id, CompletenessChecks.withdrawn_at)

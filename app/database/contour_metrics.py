"""Tall (long-format) table storing one row per contour / metric / component.

This is the structural core of the reworked quantification system: instead of a fixed
set of float columns on ``contours`` (area / perimeter / circularity / diameter), every
metric value is a row here, keyed by ``(contour_id, metric_key, component)``. This lets
new metrics (appearance, relational, multi-component colors, ...) be added without schema
migrations. The legacy columns on ``contours`` are dual-written for now and dropped in a
later step.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
)
from sqlalchemy.orm import Mapped

from app.database import database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ContourMetrics(database):
    """A single scalar metric component computed for one contour.

    Composite primary key ``(contour_id, metric_key, component)`` makes upserts (delete
    + insert, or merge) trivial and guarantees at most one row per component. Rows are
    removed automatically when the contour is deleted (ON DELETE CASCADE).
    """
    __tablename__ = "contour_metrics"

    contour_id: Mapped[int] = Column(
        Integer, ForeignKey("contours.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    metric_key: Mapped[str] = Column(String(64), primary_key=True, nullable=False)
    # Component index within a multi-dimensional metric (e.g. 0/1/2 for LAB color).
    component: Mapped[int] = Column(SmallInteger, primary_key=True, nullable=False, default=0)

    value: Mapped[float] = Column(Float, nullable=False)
    # Resolved unit for this row ("mm", "mm²", "" for unitless), see registry.resolve_unit.
    unit: Mapped[str] = Column(String(16), nullable=True)
    computed_at: Mapped[datetime] = Column(DateTime, nullable=False, default=_utcnow)
    # Marks a value as needing recomputation (e.g. after the contour geometry changed).
    stale: Mapped[bool] = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_contour_metrics_metric_key", "metric_key"),
    )

    def __repr__(self) -> str:
        return (f"<ContourMetrics(contour_id={self.contour_id}, metric_key='{self.metric_key}', "
                f"component={self.component}, value={self.value}, unit='{self.unit}')>")

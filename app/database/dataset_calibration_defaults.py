"""Per-dataset calibration defaults: how a dataset calibrates, chosen once.

A calibration strategy is almost never a per-image decision. A dataset is shot in
one session, with one camera, with the same reference card in frame — so which
strategy to use, and against which card, is a property of the dataset. Asking per
image would put a dialog on the path of every single image, which is the same
friction that kept the Calibrate tab from being a gate.

So the choice lives here, and a per-image calibration inherits it unless that
image needs something else — the one that lost its card overboard, say, and has to
fall back to a two-patch estimate.

Stored as a row per (dataset, kind) rather than a column on ``datasets`` for two
reasons: a new table is created by ``metadata.create_all`` where an added column
would need a dialect-specific ALTER, and a row carries its own provenance
(``updated_by`` / ``updated_at``) without widening the datasets table each time a
kind gains a setting.

Note what this is *not*: a calibration. It holds no measurements and corrects
nothing. It is the starting configuration a new calibration of that kind adopts.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped

from app.database import database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DatasetCalibrationDefaults(database):
    """The strategy configuration new calibrations of one kind start from."""

    __tablename__ = "dataset_calibration_defaults"

    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = Column(
        Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: Registry key of the calibration kind these defaults apply to.
    kind: Mapped[str] = Column(String(32), nullable=False)

    #: Strategy configuration, e.g. ``{"strategy": "gray_wedge", "card": "kodak_q13"}``.
    #: Validated against the strategy registry before it is written.
    defaults: Mapped[dict] = Column(JSON, nullable=False, default=dict)

    updated_by: Mapped[str] = Column(String, nullable=True)
    updated_at: Mapped[datetime] = Column(
        DateTime, nullable=False, default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        UniqueConstraint("dataset_id", "kind", name="uq_dataset_calibration_defaults"),
    )

    def __repr__(self) -> str:
        return (f"<DatasetCalibrationDefaults(dataset_id={self.dataset_id}, "
                f"kind='{self.kind}', defaults={self.defaults})>")

"""Per-image calibration records, one row per (image, calibration kind).

Calibration is the general form of what ``images.scale_x`` / ``scale_y`` / ``unit``
already did for one case: a typed transform, measured once against a reference in
the frame, that makes some family of measurements comparable across images and
sessions. Scale makes geometry comparable; intensity and colour make the
appearance metrics comparable.

Rather than adding a column per calibration kind (the pattern the tall
``contour_metrics`` table already rejected for metrics), the parameters live in a
JSON blob whose shape is owned by the kind's descriptor in
``app.services.calibration.registry``. Adding a kind is then a registry entry plus
a UI card, with no schema migration.

Two deliberate design points:

* **Non-destructive.** Only the parameters are stored — never a corrected copy of
  the image. The transform is applied at compute time (see
  ``app.services.calibration.service.load_calibrated_image_rgb``), so a
  calibration can be revised or removed at any point and provenance survives.
* **Scale is dual-written.** ``images.scale_x`` / ``scale_y`` / ``unit`` stay the
  read path for everything that predates this table (quantification contexts,
  COCO export, the status bar). The row here is the record with provenance; the
  columns are its mirror. See ``registry._persist_scale``.
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


class ImageCalibrations(database):
    """One calibration of one kind for one image.

    ``UniqueConstraint(image_id, kind)`` makes the row the single current
    calibration of that kind: setting a calibration again overwrites it rather
    than accumulating history. Rows are removed with the image (ON DELETE CASCADE).
    """
    __tablename__ = "image_calibrations"

    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = Column(
        Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: Registry key of the calibration kind ("scale", "intensity", "colour"...).
    kind: Mapped[str] = Column(String(32), nullable=False)

    #: Kind-specific parameters, validated and normalised by the kind's descriptor.
    #: Always a JSON object; never null (an absent calibration is an absent row).
    params: Mapped[dict] = Column(JSON, nullable=False, default=dict)

    #: How the calibration was obtained — see ``CalibrationSource``. Provenance is
    #: the point of the table: "measured from a reference in this frame" and
    #: "copied from the dataset default" carry very different confidence.
    source: Mapped[str] = Column(String(24), nullable=False, default="manual")

    #: Account that set it, for the same reason contours carry author_username.
    created_by: Mapped[str] = Column(String, nullable=True)
    created_at: Mapped[datetime] = Column(DateTime, nullable=False, default=_utcnow)
    updated_at: Mapped[datetime] = Column(
        DateTime, nullable=False, default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        UniqueConstraint("image_id", "kind", name="uq_image_calibrations_image_kind"),
    )

    def __repr__(self) -> str:
        return (f"<ImageCalibration(image_id={self.image_id}, kind='{self.kind}', "
                f"source='{self.source}', params={self.params})>")

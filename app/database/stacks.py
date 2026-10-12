"""A stack: one dataset item made of ordered 2D frames.

An OCT volume (B-scans along ``z``), a microscopy z-stack and a video (frames along
``t``) are the same thing to the tool: an ordered list of images that belong
together. The frames themselves are ``Images`` rows (``kind = 'frame'``), so
annotation, review, metadata and the AI models work on them as on any image; this
row holds what is true of the stack as a whole.

Replaces the earlier ``scans`` table, which kept a folder path and a slice count but
could not carry annotations because nothing pointed from an image back to it.

Stack-level metadata (eye, visit date, device) is stored once, on the stack, in
``image_metadata`` with ``stack_id`` set, and every frame inherits it; see
``app.services.database_access.image_metadata``.
"""
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.database import database


class Stacks(database):
    __tablename__ = "stacks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)

    #: What the stack is, which drives how it is presented: ``oct_volume``,
    #: ``video`` or ``z_stack``.
    kind = Column(String(32), nullable=False)
    #: The axis the frame index walks along: ``z`` (space) or ``t`` (time).
    axis = Column(String(1), nullable=False)
    frame_count = Column(Integer, nullable=False)
    #: Distance between neighbouring frames (mm for ``z``, seconds for ``t``).
    #: Null when the file does not say.
    frame_spacing = Column(Float, nullable=True)
    frame_spacing_unit = Column(String(10), nullable=True)

    #: An en-face overview of the whole stack -- the IR-SLO of an OCT volume. Each
    #: frame's ``overview_geometry`` says where on it the frame was taken.
    overview_file_path = Column(String, nullable=True)
    overview_width = Column(Integer, nullable=True)
    overview_height = Column(Integer, nullable=True)
    overview_scale_x = Column(Float, nullable=True)
    overview_scale_y = Column(Float, nullable=True)
    overview_unit = Column(String(10), nullable=True)

    #: Gallery preview: the overview if there is one, else the middle frame.
    thumbnail_file_path = Column(String, nullable=False)

    #: Format the stack was read from, e.g. ``heidelberg_vol`` or ``heidelberg_e2e``.
    source_format = Column(String(32), nullable=False)
    #: Name of the uploaded file, kept for provenance. One file can hold several
    #: stacks (an E2E export of both eyes), so this is not unique.
    source_file_name = Column(String, nullable=False)

    created_by = Column(String, ForeignKey("users.username", ondelete="SET NULL", onupdate="CASCADE"),
                        nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    # passive_deletes: the database's ON DELETE CASCADE removes the frames (and
    # through them their masks, contours and metadata).
    frames = relationship("Frames", order_by="Frames.frame_index", passive_deletes=True,
                          foreign_keys="Frames.stack_id")

    def __repr__(self) -> str:
        return f"<Stack(id={self.id}, kind='{self.kind}', frames={self.frame_count})>"

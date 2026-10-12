from sqlalchemy import JSON, CheckConstraint, Column, Float, ForeignKey, Integer, String, UniqueConstraint

from . import database

#: ``Images.kind`` of a standalone 2D image.
KIND_IMAGE = "image"
#: ``Images.kind`` of one 2D frame of a stack (an OCT slice, a video frame).
KIND_FRAME = "frame"


class Images(database):
    """
    Represents an image in the database. An image is part of a dataset and can be associated with masks and
    contours.

    A frame of a stack is an image too (see :class:`Frames`), so everything that hangs off ``images.id`` --
    masks, contours, metadata, calibrations, inference jobs, embeddings -- works on frames unchanged. The
    frame columns live on this table rather than a joined ``frames`` table so that deleting a stack cascades
    straight to its frames and nothing that reads images needs a join.
    """
    __tablename__ = 'images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False, index=True)  # Foreign key to the datasets table

    file_name = Column(String, nullable=False)  # Image file name
    file_path = Column(String, nullable=False)  # Full path to the image file on disk
    thumbnail_file_path = Column(String, nullable=False)  # Full path to the thumbnail image file on disk
    description = Column(String, nullable=True)  # Optional description of the image

    width = Column(Integer, nullable=False)  # Width of the image in pixels
    height = Column(Integer, nullable=False)  # Height of the image in pixels
    color_mode = Column(String, nullable=False, default=3)  # Color mode of the image, eg. 'RGB', 'RGBA', 'L'

    scale_x = Column(Float, nullable=False, default=1)  # mm per pixel in X
    scale_y = Column(Float, nullable=False, default=1)  # mm per pixel in Y
    unit = Column(String(10), default="px")  # Default unit: px (for pixels)

    #: ``image`` or ``frame``; the polymorphic discriminator. Lists of a dataset's
    #: own images (the gallery) filter on ``kind == 'image'``, since a stack is
    #: listed as one item rather than as its frames.
    kind = Column(String(16), nullable=False, default=KIND_IMAGE, server_default=KIND_IMAGE, index=True)

    # -- Frame columns: set exactly when kind == 'frame' -----------------------
    stack_id = Column(Integer, ForeignKey('stacks.id', ondelete='CASCADE'), nullable=True, index=True)
    #: 0-based position of the frame in its stack.
    frame_index = Column(Integer, nullable=True)
    #: Physical position along the stack axis, in the stack's ``frame_spacing_unit``
    #: (mm for an OCT volume, seconds for a video). Optional: a stack with no known
    #: spacing still has an order.
    frame_position = Column(Float, nullable=True)
    #: Where this frame lies on the stack's overview image, in overview pixels:
    #: ``{"type": "line", "start": [x, y], "end": [x, y]}`` for an OCT B-scan.
    overview_geometry = Column(JSON, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "(kind = 'frame') = (stack_id IS NOT NULL AND frame_index IS NOT NULL)",
            name="ck_images_frame_columns",
        ),
        UniqueConstraint("stack_id", "frame_index", name="uq_images_stack_frame_index"),
    )
    __mapper_args__ = {"polymorphic_on": kind, "polymorphic_identity": KIND_IMAGE}

    def __repr__(self):
        return (f"<Image(id='{self.id}', "
                f"path='{self.file_name}', "
                f"width='{self.width}',"
                f"height='{self.height}')>"
                f"scale_x='{self.scale_x}', "
                f"scale_y='{self.scale_y}', "
                f"unit='{self.unit}')>")


class Frames(Images):
    """One 2D frame of a stack: an image that knows its stack and its position in it."""
    __mapper_args__ = {"polymorphic_identity": KIND_FRAME}

from sqlalchemy import Column, Integer, ForeignKey, Boolean, exists, case, not_, select, func, String
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from . import database
from .contours import Contours
from .rejections import AnnotationRejections


class Masks(database):
    """
        Masks table in the database. Masks are a collection of contours. Images can have multiple masks, eg. for
        different labeling schemes or different annotators.
    """
    __tablename__ = 'masks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'),
                      nullable=False, index=True)
    fully_annotated = Column(Boolean, default=False, nullable=False)  # Users can mark a mask as fully annotated indicating that all objects are there.
    file_path = Column(String, nullable=False)  # Where this mask should be saved

    image = relationship("Images")
    # passive_deletes=True: rely on the DB's ON DELETE CASCADE to remove contours
    # instead of SQLAlchemy trying to NULL out the (non-nullable) contours.mask_id.
    contours = relationship("Contours", backref="mask", passive_deletes=True)
    rejections = relationship("AnnotationRejections", back_populates="mask", passive_deletes=True)

    @hybrid_property
    def status(self) -> str:
        """Where this mask sits in the annotate -> review -> done workflow.

        ``rejected`` outranks ``reviewable``/``finished``: once a reviewer has sent
        work back, the mask belongs to the annotator again regardless of how many
        of its contours already carry approvals.
        """
        # Python-side logic (for when you already have the object)
        if not any(self.contours):
            return "not_started"
        if any(rejection.is_open for rejection in self.rejections):
            return "rejected"
        if not self.fully_annotated:
            return "in_progress"
        for contour in self.contours:
            if not any(contour.reviewed_by):
                return "reviewable"
        return "finished"

    @status.expression
    def status(cls):
        # 1. Count contours for this mask
        # We use scalar_subquery so it can be used inside the CASE statement
        contour_count = (
            select(func.count(Contours.id))
            .where(Contours.mask_id == cls.id)
            .scalar_subquery()
        )

        # 2. Check for an open (unresolved) rejection
        open_rejection_exists = exists().where(
            AnnotationRejections.mask_id == cls.id
        ).where(
            AnnotationRejections.resolved_at.is_(None)
        )

        # 3. Check for existence of unreviewed contours
        unreviewed_exists = exists().where(
            Contours.mask_id == cls.id
        ).where(
            ~Contours.reviewed_by.any()
        )

        return case(
            # Check the count explicitly
            (contour_count == 0, "not_started"),

            # Sent back by a reviewer
            (open_rejection_exists, "rejected"),

            # Check the boolean flag
            (not_(cls.fully_annotated), "in_progress"),

            # Check the unreviewed subquery
            (unreviewed_exists, "reviewable"),

            else_="finished"
        )

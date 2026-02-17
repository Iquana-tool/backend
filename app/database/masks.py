from sqlalchemy import Column, Integer, ForeignKey, Boolean, exists, case, not_, select, func, String
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship

from . import database
from .contours import Contours


class Masks(database):
    """ Represents a mask in the database. A mask holds all added contours."""
    __tablename__ = 'masks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.id', ondelete='CASCADE'),
                      nullable=False)
    fully_annotated = Column(Boolean, default=False, nullable=False)  # Users can mark a mask as fully annotated indicating that all objects are there.
    file_path = Column(String, nullable=False)  # Where this mask should be saved
    contours = relationship("Contours", backref="mask")

    @hybrid_property
    def status(self) -> str:
        # Python-side logic (for when you already have the object)
        if not any(self.contours):
            return "not_started"
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

        # 2. Check for existence of unreviewed contours
        unreviewed_exists = exists().where(
            Contours.mask_id == cls.id
        ).where(
            ~Contours.reviewed_by.any()
        )

        return case(
            # Check the count explicitly
            (contour_count == 0, "not_started"),

            # Check the boolean flag
            (not_(cls.fully_annotated), "in_progress"),

            # Check the unreviewed subquery
            (unreviewed_exists, "reviewable"),

            else_="finished"
        )
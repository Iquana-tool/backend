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

    # -- Phase statuses ------------------------------------------------------
    #
    # Annotate and review are two of the three phases an image goes through
    # (calibrate is the first, and lives on the image because it has nothing to do
    # with any mask -- see ``app.services.image_status``). Each phase answers the
    # same three-way question independently, rather than being folded into one
    # ranked lifecycle: the old single ``status`` could not say "reviewed but never
    # calibrated", and had to invent a rank order between states that are not
    # actually ordered.

    @hybrid_property
    def annotate_status(self) -> str:
        """Whether the objects on this mask have all been drawn.

        ``finished`` means the annotator marked the mask as containing every
        object. An open rejection pulls it back to ``in_progress``: a reviewer sent
        the work back, so it belongs to the annotator again. (``reject`` also
        clears ``fully_annotated``; the check is kept so the two can never disagree.)
        """
        if not any(self.contours):
            return "not_started"
        if any(rejection.is_open for rejection in self.rejections):
            return "in_progress"
        if not self.fully_annotated:
            return "in_progress"
        return "finished"

    @annotate_status.expression
    def annotate_status(cls):
        return case(
            (_contour_count(cls) == 0, "not_started"),
            (_open_rejection_exists(cls), "in_progress"),
            (not_(cls.fully_annotated), "in_progress"),
            else_="finished",
        )

    @hybrid_property
    def review_status(self) -> str:
        """How far a reviewer has got through this mask's objects.

        ``finished`` needs both halves: every contour approved *and* the mask
        submitted, because objects can still be added to a mask that was never
        marked complete. An open rejection means a reviewer has been through it and
        is waiting on a fix, which is ``in_progress`` however many approvals the
        mask already carries.
        """
        if not any(self.contours):
            return "not_started"
        if any(rejection.is_open for rejection in self.rejections):
            return "in_progress"
        reviewed = [contour for contour in self.contours if any(contour.reviewed_by)]
        if not reviewed:
            return "not_started"
        if len(reviewed) < len(self.contours) or not self.fully_annotated:
            return "in_progress"
        return "finished"

    @review_status.expression
    def review_status(cls):
        return case(
            (_contour_count(cls) == 0, "not_started"),
            (_open_rejection_exists(cls), "in_progress"),
            (not_(_reviewed_exists(cls)), "not_started"),
            (_unreviewed_exists(cls), "in_progress"),
            (not_(cls.fully_annotated), "in_progress"),
            else_="finished",
        )


# -- Building blocks shared by the two SQL expressions above --------------------
#
# Written as module-level helpers rather than inlined so the two CASEs are read as
# the same predicates in a different order, which is what they are.

def _contour_count(cls):
    """Correlated count of this mask's contours (scalar, usable inside CASE)."""
    return (
        select(func.count(Contours.id))
        .where(Contours.mask_id == cls.id)
        .scalar_subquery()
    )


def _open_rejection_exists(cls):
    """Whether a reviewer sent this mask back and the complaint is still open."""
    return exists().where(
        AnnotationRejections.mask_id == cls.id
    ).where(
        AnnotationRejections.resolved_at.is_(None)
    )


def _reviewed_exists(cls):
    """Whether at least one contour on this mask has been approved."""
    return exists().where(
        Contours.mask_id == cls.id
    ).where(
        Contours.reviewed_by.any()
    )


def _unreviewed_exists(cls):
    """Whether at least one contour on this mask still awaits approval."""
    return exists().where(
        Contours.mask_id == cls.id
    ).where(
        ~Contours.reviewed_by.any()
    )

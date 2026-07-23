from logging import getLogger

from iquana_toolbox.schemas.database.contours import Contour
from sqlalchemy import Column, Integer, ForeignKey, Float, JSON, Boolean, String, Table, case
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship, backref, Mapped

from app.database import database
# Importing the tall metrics model here registers it on ``database.metadata`` so that
# ``create_all`` creates the ``contour_metrics`` table. ``contours`` is imported by
# every DB flow (masks, users, services), so this guarantees the table always exists.
from app.database import contour_metrics as _contour_metrics  # noqa: F401

logger = getLogger(__name__)

reviewer_contour_association = Table('reviewer_contour_association',
                                     database.metadata,
                                     Column('reviewer_id', Integer,
                                            ForeignKey('users.username', ondelete='CASCADE'), primary_key=True),
                                     Column('contour_id', Integer,
                                            ForeignKey('contours.id', ondelete='CASCADE'), primary_key=True),
                                     )


class Contours(database):
    """Contours table to store contour information for masks."""
    __tablename__ = 'contours'
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    mask_id: Mapped[int] = Column(Integer, ForeignKey('masks.id', ondelete='CASCADE'),
                     nullable=False)
    parent_id: Mapped[int] = Column(Integer, ForeignKey('contours.id', ondelete='CASCADE'))
    temporary = Column(Boolean, nullable=False, default=False)  # Whether a contour is temporary or not.
    added_by: Mapped[str] = Column(String(255), nullable=False)  # Who added this contour: User, SAM2, UNET, DINO etc.
    confidence_score: Mapped[float] = Column(Float, nullable=False)  # Confidence score provided by a model, for users this is set to 1
    # Allowing labels to be null, this allows contours without labels to exist, such that users can label them later.
    label_id: Mapped[int] = Column(Integer, ForeignKey('labels.id', ondelete='CASCADE'), nullable=True)
    area: Mapped[float] = Column(Float, nullable=False)
    perimeter: Mapped[float] = Column(Float, nullable=False)
    circularity: Mapped[float] = Column(Float, nullable=False)
    diameter: Mapped[float] = Column(Float, nullable=False)
    x = Column(JSON, nullable=False)
    y = Column(JSON, nullable=False)

    # Easy access to children, this makes accessing children much faster.
    # The "parent" backref is the one-to-many (collection) side, so passive_deletes
    # goes there: deleting a contour lets the DB's ON DELETE CASCADE remove the child
    # contours instead of SQLAlchemy nulling their parent_id (which would orphan them).
    children = relationship("Contours", backref=backref("parent", passive_deletes=True),
                            remote_side=[id], single_parent=True)
    reviewed_by = relationship("Users", secondary=reviewer_contour_association, back_populates="reviewed_objects")


    @classmethod
    def from_schema(cls, model_schema: Contour, mask_id: int):
        """
        Creates a Contours DB instance from a Pydantic Contour schema.
        Assumes model_schema.quantification was already computed in pixel space
        (see save_contour_tree); missing quantifications fall back to 0.0 to
        satisfy the NOT NULL columns.
        """
        # Handle quantification mapping safely
        quant = model_schema.quantification

        return cls(
            id=model_schema.id,  # SQLAlchemy handles None as autoincrement
            mask_id=mask_id,
            parent_id=model_schema.parent_id,
            added_by=model_schema.added_by,
            confidence_score=model_schema.confidence,
            label_id=model_schema.label_id,
            # Normalized coordinates stored as JSON lists
            x=model_schema.x,
            y=model_schema.y,
            # Flat mapping of the nested Quantification object
            area=quant.area if quant else 0.0,
            perimeter=quant.perimeter if quant else 0.0,
            circularity=quant.circularity if quant else 0.0,
            diameter=quant.max_diameter if quant else 0.0,
        )


def _get_image_of_mask(session, mask_id: int):
    """Fetch the image (width/height/scales/unit) a mask belongs to."""
    from app.database.images import Images  # local imports to avoid circular deps
    from app.database.masks import Masks

    return (
        session.query(Images)
        .join(Masks, Masks.image_id == Images.id)
        .filter(Masks.id == mask_id)
        .one_or_none()
    )


def _write_geometry_metrics(session, contour_id: int, quantification, image) -> None:
    """Upsert the four geometry metric rows for a contour into ``contour_metrics``.

    Values are taken from the already-computed ``QuantificationModel`` (pixel space, scaled
    to physical units); units are resolved per metric via the registry so area rows get
    e.g. "mm²" and lengths "mm". A missing quantification is skipped silently. Existing rows
    for the same (contour_id, metric_key) are replaced so re-saving a contour is idempotent.
    """
    if quantification is None:
        return
    from app.database.contour_metrics import ContourMetrics

    unit = (image.unit if image is not None and image.unit else "px")
    # metric_key -> scalar value from the QuantificationModel.
    values_by_key = {
        "area": quantification.area,
        "perimeter": quantification.perimeter,
        "circularity": quantification.circularity,
        "max_diameter": quantification.max_diameter,
    }
    for metric_key, value in values_by_key.items():
        if value is None:
            continue
        resolved_unit = _resolve_metric_unit(metric_key, unit)
        # Replace any existing row (delete-then-insert) to keep this idempotent.
        session.query(ContourMetrics).filter(
            ContourMetrics.contour_id == contour_id,
            ContourMetrics.metric_key == metric_key,
            ContourMetrics.component == 0,
        ).delete(synchronize_session=False)
        session.add(ContourMetrics(
            contour_id=contour_id,
            metric_key=metric_key,
            component=0,
            value=float(value),
            unit=resolved_unit,
            stale=False,
        ))


def _resolve_metric_unit(metric_key: str, unit: str) -> str:
    """Resolve the per-row unit string for a geometry metric via the registry."""
    from iquana_toolbox.quantification import get_metric, resolve_unit
    return resolve_unit(get_metric(metric_key).unit_kind, unit)


def save_contour_tree(session, contour_schema: Contour, mask_id: int, parent_id=None, _image=None):
    """Recursively saves a contour and all its children to the DB.

    This is the central hook for quantification: contour coordinates are stored
    normalized to [0, 1], so any contour arriving here without quantification gets
    its metrics computed in PIXEL space (scaled to the image's physical units)
    before being written to the DB. Computing metrics from the normalized
    coordinates would anisotropically distort shapes on non-square images.
    """
    from app.database.users import Users  # local import to avoid circular deps

    # 0. Resolve the mask's image once (the recursion below reuses it) and make sure
    #    the quantification is computed from pixel-space coordinates.
    if _image is None:
        _image = _get_image_of_mask(session, mask_id)
    if contour_schema.quantification is None or contour_schema.quantification.is_empty:
        if _image is not None:
            contour_schema.compute_quantification(
                width=_image.width,
                height=_image.height,
                scale_x=_image.scale_x,
                scale_y=_image.scale_y,
                unit=_image.unit,
            )
        else:
            # Should not happen (contours belong to masks, masks to images); the
            # NOT NULL columns are then filled with 0.0 by Contours.from_schema.
            logger.warning(f"No image found for mask {mask_id}; "
                           f"cannot compute quantification for contour {contour_schema.id}.")

    # 1. Convert schema to DB model
    db_contour = Contours.from_schema(contour_schema, mask_id)
    db_contour.parent_id = parent_id

    # 2. Add to session and flush to generate db_contour.id
    session.add(db_contour)
    session.flush()

    # 3. Restore reviewed_by relationship from schema (list of usernames)
    if contour_schema.reviewed_by:
        reviewers = session.query(Users).filter(
            Users.username.in_(contour_schema.reviewed_by)
        ).all()
        db_contour.reviewed_by = reviewers

    # 3b. Dual-write the geometry metrics into the tall contour_metrics table.
    #     The values come straight from the QuantificationModel just computed above; that
    #     model and the metric registry both compute via iquana_toolbox.quantification.
    #     geometry_math, so the two stores are guaranteed identical. replace_contour
    #     deletes the old contour first (CASCADE removes its metric rows), so a plain
    #     insert here stays idempotent.
    _write_geometry_metrics(session, db_contour.id, contour_schema.quantification, _image)

    # 3c. A brand new contour joining a parent group changes the correct CONTEXTUAL
    #     (nearest-neighbour) value for every sibling already in that group (one more
    #     potential neighbor to consider), not just itself - mark the whole group stale
    #     so compute_contextual_metrics_for_dataset picks all of them up. Local import:
    #     app.services.quantification imports this module, so importing it at module
    #     level here would be circular.
    from app.services.quantification import mark_contextual_stale, mark_relational_stale
    sibling_query = session.query(Contours.id).filter(Contours.mask_id == mask_id)
    sibling_query = sibling_query.filter(Contours.parent_id.is_(None)) if parent_id is None else sibling_query.filter(
        Contours.parent_id == parent_id
    )
    mark_contextual_stale(session, {row.id for row in sibling_query.all()} | {db_contour.id})

    # 3d. RELATIONAL: this new contour joining parent P means P's n_children count grew, so
    #     P's relational row is stale (parent-targeted; unlike the sibling-group contextual
    #     invalidation above, only the single parent is affected). A root-level insert
    #     (parent_id is None) has no parent to invalidate, so this is skipped. Note this
    #     marks the row of an ALREADY-EXISTING parent; the new contour's own n_children row
    #     (0 until its own children recurse below) is written fresh by the batch compute.
    if parent_id is not None:
        mark_relational_stale(session, [parent_id])

    # 4. Recurse for children
    for child_schema in contour_schema.children:
        save_contour_tree(session, child_schema, mask_id, parent_id=db_contour.id, _image=_image)

    return db_contour

from datetime import datetime, timezone
from logging import getLogger

from iquana_toolbox.quantification.registry import get_metric, resolve_unit
from iquana_toolbox.schemas.database.contours import Contour
from sqlalchemy import Column, DateTime, Integer, ForeignKey, Float, JSON, Boolean, String, Table, case
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship, backref, Mapped

from app.database import database
# Importing the tall metrics model here registers it on ``database.metadata`` so that
# ``create_all`` creates the ``contour_metrics`` table. ``contours`` is imported by
# every DB flow (masks, users, services), so this guarantees the table always exists.
from app.database import contour_metrics as _contour_metrics  # noqa: F401

logger = getLogger(__name__)

logger = getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _resolve_metric_unit(metric_key: str, unit: str) -> str:
    """Resolve the stored unit string for one metric given the image's length unit.

    Thin adapter over the registry's :func:`resolve_unit`, keyed by metric key rather than
    unit kind so callers that only know the key (the dual-write below and
    ``scripts/backfill_contour_metrics``) do not each have to look the metric up.

    Args:
        metric_key: A registry metric key, e.g. ``"area"``.
        unit: The image's length unit, e.g. ``"px"`` or ``"mm"``.

    Returns:
        ``"mm"`` for lengths, ``"mm²"`` for areas, ``""`` for unitless metrics.
    """
    return resolve_unit(get_metric(metric_key).unit_kind, unit)


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
    added_by: Mapped[str] = Column(String(255), nullable=False)  # What produced the geometry: User, SAM2, UNET, DINO etc.
    # Who was at the keyboard. Set server-side from the authenticated session, and
    # populated even for AI-assisted contours (added_by names the model, this names
    # the human who accepted it). Separation of duties on review and the planned
    # per-user study metrics both key off this, so it must not come from the client.
    author_username: Mapped[str] = Column(String, ForeignKey("users.username", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
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
    def from_schema(cls, model_schema: Contour, mask_id: int, author_username: str | None = None):
        """
        Creates a Contours DB instance from a Pydantic Contour schema.
        Assumes model_schema.quantification is already populated by your validator.

        `author_username` is passed in by the caller from the authenticated session
        rather than read off the schema, because the schema arrives from the client.
        """
        # Handle quantification mapping safely
        quant = model_schema.quantification

        return cls(
            id=model_schema.id,  # SQLAlchemy handles None as autoincrement
            mask_id=mask_id,
            parent_id=model_schema.parent_id,
            added_by=model_schema.added_by,
            author_username=author_username,
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


def _image_for_mask(session, mask_id: int):
    """Return the ``Images`` row behind a mask, or ``None`` if it cannot be resolved."""
    from app.database.images import Images  # local imports to avoid circular deps
    from app.database.masks import Masks

    return (
        session.query(Images)
        .join(Masks, Masks.image_id == Images.id)
        .filter(Masks.id == mask_id)
        .first()
    )


def dual_write_geometry_metrics(session, mask_id: int, contours: list["Contours"]) -> None:
    """Recompute geometry server-side and write it to BOTH quantification stores.

    The geometry tier is the only one computed synchronously on the write path (it needs
    nothing but the contour points), so it is never stale. The values are written to two
    stores, in two DIFFERENT unit conventions (both derived from the same contour points,
    so they never disagree about the geometry itself):

      * the legacy float columns on ``contours`` (area / perimeter / ...), in the image's
        PHYSICAL unit - these back single-image surfaces (per-object display, COCO export)
        where the image's own scale is unambiguous, and
      * the tall ``contour_metrics`` rows, PIXEL-native (``px`` / ``px²`` / unitless) - the
        per-image physical scale is applied later, at read time, and only when a dataset's
        images share one unit, so a dataset mixing scaled and unscaled images still
        aggregates correctly (see
        ``app.services.database_access.datasets.get_quantification_summary``).

    Geometry is recomputed here rather than trusted from ``contour_schema.quantification``
    because that field arrives from the client, and because it can only be computed
    correctly with the image's dimensions and physical scale, which the schema lacks.

    A missing image (mask with no resolvable image row) is logged and skipped rather than
    raised: failing to record a metric must not abort the contour write itself.

    Args:
        session: The database session (caller controls commit).
        mask_id: The mask the contours were saved to; resolves the image geometry.
        contours: The freshly saved ``Contours`` rows (must already have ids).
    """
    # Local imports: app.services.quantification imports this module at module level.
    from app.database.contour_metrics import ContourMetrics
    from app.services.quantification import GEOMETRY_METRIC_KEYS, quantify_contour_row

    if not contours:
        return
    image = _image_for_mask(session, mask_id)
    if image is None:
        logger.warning("No image found for mask id=%s; skipping the geometry dual-write "
                       "for %d contour(s).", mask_id, len(contours))
        return

    rows: list[dict] = []
    for contour in contours:
        # 1. Legacy columns: PHYSICAL units (note: the model's column is `diameter`, the
        #    metric key is `max_diameter` - the same quantity under two names).
        quant = quantify_contour_row(contour, image)
        contour.area = quant.area
        contour.perimeter = quant.perimeter
        contour.circularity = quant.circularity
        contour.diameter = quant.max_diameter

        # 2. Tall table: PIXEL-native, one row per metric (all geometry metrics are
        #    single-component). Computed with the image's scale ignored (scale 1, px).
        quant_px = quantify_contour_row(contour, image, pixel=True)
        values_by_key = {
            "area": quant_px.area,
            "perimeter": quant_px.perimeter,
            "circularity": quant_px.circularity,
            "max_diameter": quant_px.max_diameter,
        }
        for metric_key in GEOMETRY_METRIC_KEYS:
            rows.append({
                "contour_id": contour.id,
                "metric_key": metric_key,
                "component": 0,
                "value": float(values_by_key[metric_key]),
                "unit": _resolve_metric_unit(metric_key, "px"),
                "stale": False,
            })

    # Delete-then-insert keeps this idempotent for callers that reuse a contour id
    # (``replace_contour``), where the DB-level CASCADE may not have fired.
    session.query(ContourMetrics).filter(
        ContourMetrics.contour_id.in_([c.id for c in contours]),
        ContourMetrics.metric_key.in_(GEOMETRY_METRIC_KEYS),
    ).delete(synchronize_session=False)
    session.bulk_insert_mappings(ContourMetrics, rows)
    session.flush()


def _save_contour_subtree(session, contour_schema: Contour, mask_id: int, parent_id,
                          author_username: str | None, created: list["Contours"],
                          _image=None):
    """Recursive half of :func:`save_contour_tree`; appends every saved row to ``created``."""
    from app.database.users import Users  # local import to avoid circular deps

    # 0. Resolve the mask's image once (the recursion below reuses it) and make sure
    #    the quantification is computed from pixel-space coordinates.
    if _image is None:
        _image = _image_for_mask(session, mask_id)
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
    db_contour = Contours.from_schema(contour_schema, mask_id, author_username=author_username)
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

    created.append(db_contour)

    # 4. Recurse for children
    for child_schema in contour_schema.children:
        _save_contour_subtree(session, child_schema, mask_id, db_contour.id,
                              author_username, created, _image=_image)

    return db_contour


def save_contour_tree(session, contour_schema: Contour, mask_id: int, parent_id=None,
                      author_username: str | None = None, invalidate_metrics: bool = True):
    """Recursively saves a contour and all its children to the DB.

    Beyond persisting the rows, this is the synchronous half of the quantification
    system: it dual-writes the geometry tier (see :func:`dual_write_geometry_metrics`)
    and invalidates the lazily-computed tiers that the new contours affect (see
    ``app.services.database_access.contours.invalidate_metrics_for_new_contours``). The
    whole tree is saved first so the invalidation sees the final parent links.

    Does not commit - the caller controls the transaction.

    Args:
        session: The database session.
        contour_schema: The contour tree to persist.
        mask_id: The mask to attach the contours to.
        parent_id: Parent contour id for the root of this tree, or ``None`` for root level.
        author_username: The human whose session created these contours.
        invalidate_metrics: Whether to flag the surrounding contours' lazily-computed
            metrics stale. Only set this False when the caller KNOWS there is nothing to
            invalidate - notably when repopulating a mask that was just cleared, where the
            per-contour group invalidation would be quadratic and would find no rows
            anyway (see ``masks.add_contours_from_hierarchy``).
    """
    from app.services.database_access.contours import invalidate_metrics_for_new_contours

    created: list[Contours] = []
    root = _save_contour_subtree(session, contour_schema, mask_id, parent_id,
                                 author_username, created)
    dual_write_geometry_metrics(session, mask_id, created)
    if invalidate_metrics:
        invalidate_metrics_for_new_contours(session, created)
    return root

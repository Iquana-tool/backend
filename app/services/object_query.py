"""Find objects across a dataset by label, attributes, image metadata and relationships.

This is the query side of the scene graph: *all coral fragments of species X from
images where treatment = heat that are attached to substrate*. The question is a small,
checked spec (:class:`ObjectQuery`), compiled to SQL by code here. Nothing writes SQL
from free text. A later research-question assistant (#86) can produce a spec, and this
module validates it like any other.

The result is the flat export table the quantification page already produces (one row
per object, ``meta_*``, ``attr_*``, ``rel_*`` and metric columns), restricted to the
objects that match. Measurements reuse that path, so units follow the same
all-or-nothing scale rule and the numbers match the quantification page.

Measured attributes (metrics) can be filtered on too: *area > 2 cm²*, *L\* < 40*. Stored
metric values are pixel-native, so a threshold is read in the dataset's display unit
(physical when every contributing image shares one unit, pixels otherwise) and each
object's value is converted by its own image's scale before the comparison.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import and_, exists, or_, select
from sqlalchemy.orm import Session, aliased

from app.database.contour_metrics import ContourMetrics
from app.database.contours import Contours
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.scene_graph import (
    STATUS_CONFIRMED,
    ObjectAttributeDefinitions,
    ObjectAttributes,
    ObjectRelationships,
    RelationshipTypes,
)
from app.exceptions import InvalidMetadataError, InvalidSceneGraphError
from app.services.database_access import image_metadata as metadata_db
from app.services.database_access import scene_graph as graph_db
from app.services.metadata_types import ORDERED_TYPES, MetadataValueType, coerce

Op = Literal["eq", "ne", "in", "lt", "lte", "gt", "gte", "set", "not_set", "cannot_tell"]

#: Operators that compare against the sortable ``value_num`` and therefore need an
#: ordered type (number, date, boolean).
_RANGE_OPS = {"lt", "lte", "gt", "gte"}

DEFAULT_METRICS = ["area", "perimeter", "circularity", "max_diameter"]


#: Operators a metric condition accepts. Metrics are numbers: no vocabulary to match
#: with 'in', and no "cannot tell".
_METRIC_OPS = {"eq", "ne", "lt", "lte", "gt", "gte", "set", "not_set"}


class Condition(BaseModel):
    """One filter on a given attribute, an image-metadata key or a measured metric."""
    attribute: str | None = None
    metadata: str | None = None
    metric: str | None = Field(default=None, description="A metric key, e.g. 'area' or 'mean_color_lab'.")
    component: int = Field(default=0, ge=0,
                           description="For multi-value metrics: 0/1/2 = L*/a*/b* of mean_color_lab.")
    op: Op = "eq"
    value: str | float | bool | None = None
    values: list[str | float | bool] = Field(default_factory=list, description="For op 'in'.")

    @model_validator(mode="after")
    def _one_subject(self):
        if sum(x is not None for x in (self.attribute, self.metadata, self.metric)) != 1:
            raise ValueError("A condition names exactly one of 'attribute', 'metadata' or 'metric'.")
        if self.op == "cannot_tell" and self.attribute is None:
            raise ValueError("'cannot_tell' applies to attributes only.")
        if self.metric is not None and self.op not in _METRIC_OPS:
            raise ValueError(f"A metric condition takes one of {', '.join(sorted(_METRIC_OPS))}.")
        return self


class RelatedCondition(BaseModel):
    """The object must have (or, with ``negate``, must not have) such a relationship."""
    type: str
    direction: Literal["out", "in", "any"] = "out"
    label_id: int | None = Field(default=None, description="Label of the object at the other end.")
    negate: bool = False


class ObjectQuery(BaseModel):
    label_ids: list[int] = Field(default_factory=list, description="Empty = every label.")
    image_ids: list[int] | None = None
    where: list[Condition] = Field(default_factory=list)
    related: list[RelatedCondition] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_METRICS))
    exclude_not_fully_annotated: bool = False
    exclude_unreviewed: bool = False


def _value_text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def _typed(value, value_type: str, label: str) -> tuple[str, float | None]:
    """Coerce a comparison value by the attribute's / key's type, as stored values are."""
    try:
        return coerce(_value_text(value), value_type)
    except InvalidMetadataError as exc:
        raise InvalidSceneGraphError(f"{label}: {exc}") from exc


def _compare(text_col, num_col, condition: Condition, value_type: str, label: str):
    """The SQL comparison for a value-carrying operator."""
    op = condition.op
    if op in _RANGE_OPS:
        if MetadataValueType(value_type) not in ORDERED_TYPES:
            raise InvalidSceneGraphError(f"{label} is {value_type}; '{op}' needs a number, date or yes/no.")
        _, number = _typed(condition.value, value_type, label)
        return {"lt": num_col < number, "lte": num_col <= number,
                "gt": num_col > number, "gte": num_col >= number}[op]
    if op == "in":
        if not condition.values:
            raise InvalidSceneGraphError(f"{label}: 'in' needs a list of values.")
        return text_col.in_([_typed(v, value_type, label)[0] for v in condition.values])
    if condition.value is None:
        raise InvalidSceneGraphError(f"{label}: '{op}' needs a value.")
    text, _ = _typed(condition.value, value_type, label)
    return text_col == text if op == "eq" else text_col != text


def _attribute_filter(db: Session, dataset_id: int, condition: Condition):
    definition = graph_db.find_attribute(db, dataset_id, condition.attribute)
    if definition is None:
        raise InvalidSceneGraphError(f"This dataset has no attribute '{condition.attribute}'.")
    label = f"Attribute '{definition.name}'"
    row = select(ObjectAttributes.id).where(
        ObjectAttributes.contour_id == Contours.id,
        ObjectAttributes.attribute_id == definition.id,
        ObjectAttributes.status == STATUS_CONFIRMED,
    )
    if condition.op == "not_set":
        return ~exists(row)
    if condition.op == "set":
        return exists(row.where(ObjectAttributes.value.is_not(None)))
    if condition.op == "cannot_tell":
        return exists(row.where(ObjectAttributes.value.is_(None)))
    return exists(row.where(_compare(ObjectAttributes.value, ObjectAttributes.value_num,
                                     condition, definition.value_type, label)))


def _metadata_filter(db: Session, dataset_id: int, condition: Condition):
    key = metadata_db.normalize_key(condition.metadata)
    descriptor = metadata_db.get_key(db, dataset_id, key)
    if descriptor is None:
        raise InvalidSceneGraphError(f"This dataset has no metadata key '{key}'.")
    eff = metadata_db.effective_metadata()
    row = select(eff.c.image_id).where(eff.c.image_id == Images.id, eff.c.key == key)
    if condition.op == "not_set":
        return ~exists(row)
    if condition.op == "set":
        return exists(row)
    return exists(row.where(_compare(eff.c.value, eff.c.value_num, condition,
                                     descriptor.value_type, f"Metadata '{key}'")))


def _metric_filter(condition: Condition, display_physical: bool):
    """An object matches when its fresh value of the metric component passes the test.

    With ``display_physical`` the stored pixel value is converted by the object's own
    image scale (``Images`` is the outer query's row), so one threshold in mm² means the
    same on every image whatever its resolution.
    """
    # Local imports: the quantification modules import the database-access layer.
    from iquana_toolbox.quantification import METRIC_REGISTRY
    from app.services.database_access.datasets import _AREA_METRIC_KEYS, _LENGTH_METRIC_KEYS

    metric = METRIC_REGISTRY.get(condition.metric)
    if metric is None:
        raise InvalidSceneGraphError(f"Unknown metric '{condition.metric}'.")
    if condition.component >= metric.value_dim:
        raise InvalidSceneGraphError(
            f"Metric '{condition.metric}' has {metric.value_dim} component(s); "
            f"component {condition.component} does not exist.")
    row = select(ContourMetrics.contour_id).where(
        ContourMetrics.contour_id == Contours.id,
        ContourMetrics.metric_key == condition.metric,
        ContourMetrics.component == condition.component,
        ContourMetrics.stale == False,  # noqa: E712
    )
    if condition.op == "not_set":
        return ~exists(row)
    if condition.op == "set":
        return exists(row)

    if isinstance(condition.value, bool) or condition.value is None:
        raise InvalidSceneGraphError(f"Metric '{condition.metric}': '{condition.op}' needs a number.")
    try:
        threshold = float(condition.value)
    except (TypeError, ValueError):
        raise InvalidSceneGraphError(f"Metric '{condition.metric}': '{condition.value}' is not a number.")

    value = ContourMetrics.value
    if display_physical and condition.metric in _AREA_METRIC_KEYS:
        value = value * Images.scale_x * Images.scale_y
    elif display_physical and condition.metric in _LENGTH_METRIC_KEYS:
        value = value * Images.scale_x
    test = {"eq": value == threshold, "ne": value != threshold, "lt": value < threshold,
            "lte": value <= threshold, "gt": value > threshold, "gte": value >= threshold}[condition.op]
    return exists(row.where(test))


def scale_status(db: Session, dataset_id: int, query: ObjectQuery) -> dict:
    """The unit metric thresholds are read in and metric columns are reported in.

    The same all-or-nothing decision the quantification page makes, so a threshold
    typed against a number on that page means the same thing here.
    """
    from app.services.database_access.datasets import _resolve_scale_display
    return _resolve_scale_display(dataset_id, query.exclude_not_fully_annotated,
                                  query.exclude_unreviewed, db)


def _related_filter(db: Session, dataset_id: int, condition: RelatedCondition):
    rel_type = graph_db.find_relationship_type(db, dataset_id, condition.type)
    if rel_type is None:
        raise InvalidSceneGraphError(f"This dataset has no relationship '{condition.type}'.")
    other = aliased(Contours)
    edge = ObjectRelationships
    outgoing = and_(edge.subject_id == Contours.id, other.id == edge.object_id)
    incoming = and_(edge.object_id == Contours.id, other.id == edge.subject_id)
    direction = ("any" if rel_type.symmetric else condition.direction)
    ends = {"out": outgoing, "in": incoming, "any": or_(outgoing, incoming)}[direction]
    row = select(edge.id).where(edge.type_id == rel_type.id, edge.status == STATUS_CONFIRMED, ends)
    if condition.label_id is not None:
        row = row.where(other.label_id == condition.label_id)
    clause = exists(row)
    return ~clause if condition.negate else clause


def matching_rows(db: Session, dataset_id: int, query: ObjectQuery,
                  display_physical: bool | None = None):
    """The ``(Contours, file_name, Labels, image_id)`` rows matching ``query``.

    Same row shape as the quantification export, which is what makes the rest of that
    path reusable. Metric conditions compare against fresh values only; :func:`run_query`
    computes the stale ones first. ``display_physical`` is the unit decision for metric
    thresholds; it is resolved here when not passed.

    :raises InvalidSceneGraphError: for an unknown attribute, key or relationship, or a
        value that its type cannot represent.
    """
    sql = (
        db.query(Contours, Images.file_name, Labels, Images.id)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .join(Labels, Labels.id == Contours.label_id)
        .filter(Images.dataset_id == dataset_id, Contours.temporary == False)  # noqa: E712
    )
    if query.label_ids:
        sql = sql.filter(Contours.label_id.in_(query.label_ids))
    if query.image_ids is not None:
        sql = sql.filter(Images.id.in_(query.image_ids))
    if query.exclude_not_fully_annotated:
        sql = sql.filter(Masks.fully_annotated == True)  # noqa: E712
    if query.exclude_unreviewed:
        sql = sql.filter(Contours.reviewed_by.any())
    if display_physical is None and any(c.metric for c in query.where):
        display_physical = scale_status(db, dataset_id, query)["display_physical"]
    for condition in query.where:
        if condition.metric is not None:
            sql = sql.filter(_metric_filter(condition, bool(display_physical)))
        elif condition.attribute is not None:
            sql = sql.filter(_attribute_filter(db, dataset_id, condition))
        else:
            sql = sql.filter(_metadata_filter(db, dataset_id, condition))
    for condition in query.related:
        sql = sql.filter(_related_filter(db, dataset_id, condition))
    return sql.order_by(Images.id, Contours.id).all()


def run_query(db: Session, dataset_id: int, query: ObjectQuery) -> pd.DataFrame:
    """The flat table of the matching objects, with the requested metrics.

    Metric tiers that are computed lazily (appearance, contextual, relational) are
    brought up to date first, for the metrics shown *and* those filtered on, so that a
    filter never runs against values that do not exist yet. The unit decision is in
    ``df.attrs["scale_status"]``.
    """
    # Local imports: the quantification modules import the database-access layer.
    from iquana_toolbox.quantification import METRIC_REGISTRY
    from app.services.database_access.datasets import _dataset_df_from_profile
    from app.services import quantification

    filtered = [c.metric for c in query.where if c.metric is not None]
    unknown = [key for key in [*query.metrics, *filtered] if key not in METRIC_REGISTRY]
    if unknown:
        raise InvalidSceneGraphError(f"Unknown metric(s): {', '.join(unknown)}.")

    status = scale_status(db, dataset_id, query)
    tiers = {METRIC_REGISTRY[key].tier.value for key in [*query.metrics, *filtered]}
    compute = {
        "geometry": quantification.compute_geometry_metrics_for_dataset,
        "appearance": quantification.compute_appearance_metrics_for_dataset,
        "contextual": quantification.compute_contextual_metrics_for_dataset,
        "relational": quantification.compute_relational_metrics_for_dataset,
    }
    for tier in tiers:
        compute[tier](db, dataset_id, only_stale=True)

    rows = matching_rows(db, dataset_id, query, display_physical=status["display_physical"])
    df = _dataset_df_from_profile(
        rows, dataset_id, {key: None for key in query.metrics}, db,
        query.exclude_not_fully_annotated, query.exclude_unreviewed,
    )
    df.attrs["scale_status"] = status
    return df

"""Reading and writing the scene graph: attribute definitions and values, relationship
types and relationships.

See ``app.database.scene_graph`` for the model. Everything here is dataset-scoped:
callers pass the dataset they checked permissions on, and every id they hand in is
verified to belong to it, so a route cannot be used to reach another dataset's
objects by guessing ids.

Write functions do not commit unless they say so; the caller owns the transaction
(the routes commit once, after recording the undo history).
"""
from __future__ import annotations

from collections import defaultdict
from logging import getLogger

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.contours import Contours
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.scene_graph import (
    SOURCE_USER,
    SOURCES,
    STATUS_CONFIRMED,
    STATUSES,
    ObjectAttributeDefinitions,
    ObjectAttributes,
    ObjectRelationships,
    RelationshipTypes,
)
from app.exceptions import (
    InvalidMetadataError,
    InvalidSceneGraphError,
    SceneGraphNotFoundError,
)
from app.services.database_access.image_metadata import normalize_key, normalize_value
from app.services.metadata_types import MetadataValueType, coerce

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _name(raw: str) -> str:
    """Canonical attribute / relationship name, by the metadata-key rules."""
    try:
        return normalize_key(raw)
    except InvalidMetadataError as exc:
        raise InvalidSceneGraphError(str(exc).replace("metadata key", "name")) from exc


def contours_in_dataset(db: Session, dataset_id: int, contour_ids) -> dict[int, Contours]:
    """``{id: contour}`` for the given ids, all of which must belong to ``dataset_id``.

    :raises SceneGraphNotFoundError: naming the ids that are missing or elsewhere.
    """
    wanted = {int(cid) for cid in contour_ids}
    if not wanted:
        return {}
    rows = (
        db.query(Contours)
        .join(Masks, Masks.id == Contours.mask_id)
        .join(Images, Images.id == Masks.image_id)
        .filter(Contours.id.in_(wanted), Images.dataset_id == dataset_id)
        .all()
    )
    found = {row.id: row for row in rows}
    missing = sorted(wanted - found.keys())
    if missing:
        raise SceneGraphNotFoundError(
            f"Object(s) {', '.join(map(str, missing[:10]))} do not exist in dataset {dataset_id}."
        )
    return found


def _check_provenance(source: str, status: str) -> None:
    if source not in SOURCES:
        raise InvalidSceneGraphError(f"source must be one of {', '.join(SOURCES)}.")
    if status not in STATUSES:
        raise InvalidSceneGraphError(f"status must be one of {', '.join(STATUSES)}.")


# ---------------------------------------------------------------------------
# Attribute definitions
# ---------------------------------------------------------------------------

def _normalize_options(options) -> list[dict]:
    """Accept ``["Acropora", ...]`` or ``[{"value": ..., "code": ...}, ...]``.

    Values are trimmed and must be unique; ``code`` and ``description`` are optional.
    """
    normalized: list[dict] = []
    seen: set[str] = set()
    for option in options or []:
        if isinstance(option, str):
            option = {"value": option}
        if not isinstance(option, dict) or "value" not in option:
            raise InvalidSceneGraphError("Each option needs a 'value'.")
        value = normalize_value(option["value"])
        if not value:
            raise InvalidSceneGraphError("An option cannot be empty.")
        if value in seen:
            raise InvalidSceneGraphError(f"Option '{value}' is listed twice.")
        seen.add(value)
        normalized.append({
            "value": value,
            "code": (str(option["code"]).strip() or None) if option.get("code") else None,
            "description": (str(option["description"]).strip() or None)
            if option.get("description") else None,
        })
    return normalized


def _value_type(raw) -> MetadataValueType:
    try:
        return MetadataValueType(raw)
    except ValueError:
        raise InvalidSceneGraphError(
            f"value_type must be one of {', '.join(t.value for t in MetadataValueType)}."
        )


def _labels_in_dataset(db: Session, dataset_id: int, label_ids) -> list[Labels]:
    wanted = {int(lid) for lid in label_ids or []}
    if not wanted:
        return []
    labels = db.query(Labels).filter(Labels.id.in_(wanted), Labels.dataset_id == dataset_id).all()
    missing = wanted - {label.id for label in labels}
    if missing:
        raise SceneGraphNotFoundError(
            f"Label(s) {', '.join(map(str, sorted(missing)))} do not exist in dataset {dataset_id}."
        )
    return labels


def list_attributes(db: Session, dataset_id: int) -> list[ObjectAttributeDefinitions]:
    return (db.query(ObjectAttributeDefinitions)
            .filter_by(dataset_id=dataset_id)
            .order_by(ObjectAttributeDefinitions.name)
            .all())


def get_attribute(db: Session, dataset_id: int, attribute_id: int) -> ObjectAttributeDefinitions:
    definition = (db.query(ObjectAttributeDefinitions)
                  .filter_by(id=attribute_id, dataset_id=dataset_id)
                  .first())
    if definition is None:
        raise SceneGraphNotFoundError(f"Dataset {dataset_id} has no attribute {attribute_id}.")
    return definition


def find_attribute(db: Session, dataset_id: int, name: str) -> ObjectAttributeDefinitions | None:
    return (db.query(ObjectAttributeDefinitions)
            .filter_by(dataset_id=dataset_id, name=_name(name))
            .first())


def create_attribute(
        db: Session,
        dataset_id: int,
        name: str,
        value_type: str = MetadataValueType.CATEGORICAL.value,
        unit: str | None = None,
        options=None,
        description: str | None = None,
        label_ids=None,
        username: str | None = None,
) -> ObjectAttributeDefinitions:
    """Define a new attribute. Names are unique per dataset."""
    name = _name(name)
    if find_attribute(db, dataset_id, name) is not None:
        raise InvalidSceneGraphError(f"Attribute '{name}' already exists in this dataset.")
    definition = ObjectAttributeDefinitions(
        dataset_id=dataset_id,
        name=name,
        value_type=_value_type(value_type).value,
        unit=(unit or None),
        options=_normalize_options(options),
        description=(description or None),
        created_by=username,
    )
    definition.labels = _labels_in_dataset(db, dataset_id, label_ids)
    db.add(definition)
    db.flush()
    return definition


def update_attribute(
        db: Session,
        dataset_id: int,
        attribute_id: int,
        name: str | None = None,
        value_type: str | None = None,
        unit: str | None = None,
        options=None,
        description: str | None = None,
        label_ids=None,
        rename_options: dict[str, str] | None = None,
) -> ObjectAttributeDefinitions:
    """Change an attribute's declaration, re-validating every stored value.

    As for image-metadata keys, retyping or narrowing the vocabulary is refused,
    naming the offending values, when stored values would not fit. ``rename_options``
    (``{old: new}``) renames values in the vocabulary *and* on every object in one go,
    which is how a split like ``Acropora`` / ``acropora`` is repaired.

    ``None`` leaves a field as it is; an empty string clears ``unit``/``description``,
    an empty list clears ``options``/``label_ids``.
    """
    definition = get_attribute(db, dataset_id, attribute_id)

    if name is not None:
        new_name = _name(name)
        clash = find_attribute(db, dataset_id, new_name)
        if clash is not None and clash.id != definition.id:
            raise InvalidSceneGraphError(f"Attribute '{new_name}' already exists in this dataset.")
        definition.name = new_name

    new_type = _value_type(value_type) if value_type else MetadataValueType(definition.value_type)
    new_options = (_normalize_options(options) if options is not None
                   else [dict(option) for option in definition.options or []])

    renames = {normalize_value(old): normalize_value(new)
               for old, new in (rename_options or {}).items()}
    if any(not new for new in renames.values()):
        raise InvalidSceneGraphError("An option cannot be renamed to an empty value.")
    if renames and options is None:
        # Rename in the vocabulary too, unless the caller sent a new one.
        for option in new_options:
            if option["value"] in renames:
                option["value"] = renames[option["value"]]
        if len({o["value"] for o in new_options}) != len(new_options):
            raise InvalidSceneGraphError("Renaming would list an option twice.")

    allowed = [option["value"] for option in new_options]
    rows = (db.query(ObjectAttributes)
            .filter(ObjectAttributes.attribute_id == definition.id,
                    ObjectAttributes.value.is_not(None))
            .all())
    coerced, rejected = [], []
    for row in rows:
        value = renames.get(row.value, row.value)
        try:
            canonical, numeric = coerce(value, new_type, allowed)
        except InvalidMetadataError:
            rejected.append(value)
            continue
        coerced.append((row, canonical, numeric))
    if rejected:
        sample = sorted(set(rejected))[:5]
        raise InvalidSceneGraphError(
            f"{len(rejected)} object(s) have a '{definition.name}' value that is not a valid "
            f"{new_type.value}: {', '.join(repr(v) for v in sample)}"
            + (" ..." if len(set(rejected)) > len(sample) else "")
            + ". Fix or clear those values first."
        )
    for row, canonical, numeric in coerced:
        row.value = canonical
        row.value_num = numeric

    definition.value_type = new_type.value
    definition.options = new_options
    if unit is not None:
        definition.unit = unit or None
    if description is not None:
        definition.description = description or None
    if label_ids is not None:
        definition.labels = _labels_in_dataset(db, dataset_id, label_ids)
    db.flush()
    return definition


def delete_attribute(db: Session, dataset_id: int, attribute_id: int) -> int:
    """Delete an attribute and every value of it. Returns how many values went."""
    definition = get_attribute(db, dataset_id, attribute_id)
    count = db.query(ObjectAttributes).filter_by(attribute_id=definition.id).count()
    db.delete(definition)
    db.flush()
    return count


# ---------------------------------------------------------------------------
# Attribute values
# ---------------------------------------------------------------------------

def value_row_state(row: ObjectAttributes | None) -> dict | None:
    """A value row as plain data, for the undo history. ``None`` = no row."""
    if row is None:
        return None
    return {
        "value": row.value,
        "value_num": row.value_num,
        "source": row.source,
        "status": row.status,
        "model_key": row.model_key,
        "confidence": row.confidence,
        "author_username": row.author_username,
    }


def apply_value_state(db: Session, contour_id: int, attribute_id: int, state: dict | None) -> None:
    """Make the (object, attribute) row match ``state`` exactly; ``None`` deletes it."""
    row = db.query(ObjectAttributes).filter_by(contour_id=contour_id,
                                               attribute_id=attribute_id).first()
    if state is None:
        if row is not None:
            db.delete(row)
        return
    if row is None:
        row = ObjectAttributes(contour_id=contour_id, attribute_id=attribute_id)
        db.add(row)
    for key, value in state.items():
        setattr(row, key, value)


def coerce_value(definition: ObjectAttributeDefinitions, value) -> tuple[str | None, float | None]:
    """Validate one value against its attribute. ``None`` (cannot tell) passes as is."""
    if value is None:
        return None, None
    if isinstance(value, bool):
        value = "true" if value else "false"
    text = normalize_value(value)
    if not text:
        raise InvalidSceneGraphError(
            "An empty value is not a value. Clear the attribute to unset it, or send null "
            "for 'cannot tell'."
        )
    try:
        return coerce(text, definition.value_type, definition.option_values)
    except InvalidMetadataError as exc:
        message = str(exc)
        if definition.option_values and definition.value_type == MetadataValueType.CATEGORICAL:
            message = (f"'{text}' is not an allowed value of '{definition.name}'. Allowed: "
                       f"{', '.join(definition.option_values[:10])}"
                       + (" ..." if len(definition.option_values) > 10 else "") + ".")
        raise InvalidSceneGraphError(message) from exc


def set_attribute_values(
        db: Session,
        dataset_id: int,
        attribute_id: int,
        contour_ids,
        value,
        username: str | None = None,
        source: str = SOURCE_USER,
        status: str = STATUS_CONFIRMED,
        model_key: str | None = None,
        confidence: float | None = None,
) -> list[dict]:
    """Set one attribute to one value on many objects.

    ``value`` is a string (or number/bool, stringified), or ``None`` for *cannot tell*.
    Objects whose label is outside the attribute's scope are refused, naming them.

    Returns the changes as ``[{contour_id, mask_id, attribute_id, before, after}]``
    for the undo history; objects whose value did not change are left out.
    """
    _check_provenance(source, status)
    definition = get_attribute(db, dataset_id, attribute_id)
    contours = contours_in_dataset(db, dataset_id, contour_ids)
    out_of_scope = sorted(cid for cid, c in contours.items() if not definition.applies_to(c.label_id))
    if out_of_scope:
        raise InvalidSceneGraphError(
            f"'{definition.name}' does not apply to the label of object(s) "
            f"{', '.join(map(str, out_of_scope[:10]))}."
        )
    canonical, numeric = coerce_value(definition, value)

    existing = {
        row.contour_id: row
        for row in db.query(ObjectAttributes).filter(
            ObjectAttributes.attribute_id == definition.id,
            ObjectAttributes.contour_id.in_(contours.keys()),
        )
    }
    after = {
        "value": canonical,
        "value_num": numeric,
        "source": source,
        "status": status,
        "model_key": model_key,
        "confidence": confidence,
        "author_username": username,
    }
    changes = []
    for contour_id in sorted(contours):
        before = value_row_state(existing.get(contour_id))
        if before is not None and all(before[k] == after[k]
                                      for k in ("value", "source", "status", "model_key")):
            continue
        apply_value_state(db, contour_id, definition.id, after)
        changes.append({"contour_id": contour_id, "mask_id": contours[contour_id].mask_id,
                        "attribute_id": definition.id, "before": before, "after": dict(after)})
    db.flush()
    return changes


def clear_attribute_values(db: Session, dataset_id: int, attribute_id: int, contour_ids) -> list[dict]:
    """Unset one attribute on many objects (back to *not annotated*)."""
    definition = get_attribute(db, dataset_id, attribute_id)
    contours = contours_in_dataset(db, dataset_id, contour_ids)
    rows = db.query(ObjectAttributes).filter(
        ObjectAttributes.attribute_id == definition.id,
        ObjectAttributes.contour_id.in_(contours.keys()),
    ).all()
    changes = []
    for row in rows:
        changes.append({"contour_id": row.contour_id, "mask_id": contours[row.contour_id].mask_id,
                        "attribute_id": definition.id, "before": value_row_state(row),
                        "after": None})
        db.delete(row)
    db.flush()
    return changes


def confirm_attribute_values(db: Session, dataset_id: int, attribute_id: int, contour_ids,
                             username: str | None) -> list[dict]:
    """Accept suggested values: they become confirmed, credited to ``username``."""
    definition = get_attribute(db, dataset_id, attribute_id)
    contours = contours_in_dataset(db, dataset_id, contour_ids)
    changes = []
    for row in db.query(ObjectAttributes).filter(
            ObjectAttributes.attribute_id == definition.id,
            ObjectAttributes.contour_id.in_(contours.keys()),
            ObjectAttributes.status != STATUS_CONFIRMED):
        before = value_row_state(row)
        row.status = STATUS_CONFIRMED
        row.author_username = username
        changes.append({"contour_id": row.contour_id, "mask_id": contours[row.contour_id].mask_id,
                        "attribute_id": definition.id, "before": before,
                        "after": value_row_state(row)})
    db.flush()
    return changes


def attributes_of_contours(db: Session, contour_ids) -> dict[int, list[dict]]:
    """``{contour_id: [value dict, ...]}``, each with its definition's name and scope."""
    ids = list(contour_ids)
    out: dict[int, list[dict]] = defaultdict(list)
    if not ids:
        return out
    rows = (db.query(ObjectAttributes, Contours.label_id)
            .join(Contours, Contours.id == ObjectAttributes.contour_id)
            .filter(ObjectAttributes.contour_id.in_(ids))
            .all())
    for row, label_id in rows:
        definition = row.definition
        out[row.contour_id].append({
            "attribute_id": definition.id,
            "name": definition.name,
            "value": row.value,
            "unit": definition.unit,
            "cannot_tell": row.value is None,
            "source": row.source,
            "status": row.status,
            "model_key": row.model_key,
            "confidence": row.confidence,
            "author_username": row.author_username,
            # Kept when the object was relabelled out of the attribute's scope, so that
            # relabelling back restores it; exports skip it.
            "out_of_scope": not definition.applies_to(label_id),
        })
    return out


# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------

def list_relationship_types(db: Session, dataset_id: int) -> list[RelationshipTypes]:
    return (db.query(RelationshipTypes)
            .filter_by(dataset_id=dataset_id)
            .order_by(RelationshipTypes.name)
            .all())


def relationship_type_usage(db: Session, dataset_id: int) -> dict[int, int]:
    """``{type_id: number of relationships}`` across the dataset."""
    rows = (db.query(ObjectRelationships.type_id, func.count(ObjectRelationships.id))
            .filter(ObjectRelationships.dataset_id == dataset_id)
            .group_by(ObjectRelationships.type_id)
            .all())
    return {type_id: count for type_id, count in rows}


def get_relationship_type(db: Session, dataset_id: int, type_id: int) -> RelationshipTypes:
    rel_type = db.query(RelationshipTypes).filter_by(id=type_id, dataset_id=dataset_id).first()
    if rel_type is None:
        raise SceneGraphNotFoundError(f"Dataset {dataset_id} has no relationship type {type_id}.")
    return rel_type


def find_relationship_type(db: Session, dataset_id: int, name: str) -> RelationshipTypes | None:
    return db.query(RelationshipTypes).filter_by(dataset_id=dataset_id, name=_name(name)).first()


def create_relationship_type(db: Session, dataset_id: int, name: str, symmetric: bool = False,
                             description: str | None = None,
                             username: str | None = None) -> RelationshipTypes:
    name = _name(name)
    if find_relationship_type(db, dataset_id, name) is not None:
        raise InvalidSceneGraphError(f"Relationship '{name}' already exists in this dataset.")
    rel_type = RelationshipTypes(dataset_id=dataset_id, name=name, symmetric=bool(symmetric),
                                 description=description or None, created_by=username)
    db.add(rel_type)
    db.flush()
    return rel_type


def ensure_relationship_type(db: Session, dataset_id: int, name: str,
                             username: str | None = None) -> RelationshipTypes:
    """The type called ``name``, created (as directed) if the dataset does not use it yet.

    This is how names get reused: typing an existing name picks it up.
    """
    return (find_relationship_type(db, dataset_id, name)
            or create_relationship_type(db, dataset_id, name, username=username))


def update_relationship_type(db: Session, dataset_id: int, type_id: int, name: str | None = None,
                             symmetric: bool | None = None,
                             description: str | None = None) -> RelationshipTypes:
    """Rename or redeclare a type.

    Making a type symmetric rewrites its edges into canonical order (lower id first)
    and drops the duplicates that creates (``A touches B`` and ``B touches A``).
    """
    rel_type = get_relationship_type(db, dataset_id, type_id)
    if name is not None:
        new_name = _name(name)
        clash = find_relationship_type(db, dataset_id, new_name)
        if clash is not None and clash.id != rel_type.id:
            raise InvalidSceneGraphError(f"Relationship '{new_name}' already exists in this dataset.")
        rel_type.name = new_name
    if description is not None:
        rel_type.description = description or None
    if symmetric is not None and bool(symmetric) != bool(rel_type.symmetric):
        rel_type.symmetric = bool(symmetric)
        if rel_type.symmetric:
            seen: set[tuple[int, int]] = set()
            for edge in (db.query(ObjectRelationships)
                         .filter_by(type_id=rel_type.id)
                         .order_by(ObjectRelationships.id)):
                pair = (min(edge.subject_id, edge.object_id), max(edge.subject_id, edge.object_id))
                if pair in seen:
                    db.delete(edge)
                    continue
                seen.add(pair)
            db.flush()
            for edge in db.query(ObjectRelationships).filter_by(type_id=rel_type.id):
                if edge.subject_id > edge.object_id:
                    edge.subject_id, edge.object_id = edge.object_id, edge.subject_id
    db.flush()
    return rel_type


def merge_relationship_types(db: Session, dataset_id: int, source_id: int,
                             target_id: int) -> RelationshipTypes:
    """Fold one relationship name into another ("attached to" into "attached_to")."""
    source = get_relationship_type(db, dataset_id, source_id)
    target = get_relationship_type(db, dataset_id, target_id)
    if source.id == target.id:
        return target
    existing = {(e.subject_id, e.object_id)
                for e in db.query(ObjectRelationships).filter_by(type_id=target.id)}
    for edge in db.query(ObjectRelationships).filter_by(type_id=source.id).all():
        pair = (edge.subject_id, edge.object_id)
        if target.symmetric:
            pair = (min(pair), max(pair))
        if pair in existing:
            db.delete(edge)
            continue
        existing.add(pair)
        edge.type_id = target.id
        edge.subject_id, edge.object_id = pair
    db.flush()
    db.delete(source)
    db.flush()
    return target


def delete_relationship_type(db: Session, dataset_id: int, type_id: int) -> int:
    """Delete a type and all its edges. Returns how many edges went."""
    rel_type = get_relationship_type(db, dataset_id, type_id)
    count = db.query(ObjectRelationships).filter_by(type_id=rel_type.id).count()
    db.delete(rel_type)
    db.flush()
    return count


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

def edge_state(edge: ObjectRelationships) -> dict:
    """An edge as plain data, for the undo history and snapshots."""
    return {
        "id": edge.id,
        "dataset_id": edge.dataset_id,
        "type_id": edge.type_id,
        "subject_id": edge.subject_id,
        "object_id": edge.object_id,
        "source": edge.source,
        "status": edge.status,
        "model_key": edge.model_key,
        "confidence": edge.confidence,
        "author_username": edge.author_username,
        "created_at": edge.created_at.isoformat() if edge.created_at else None,
    }


def edge_to_dict(edge: ObjectRelationships) -> dict:
    """An edge for API responses: its state plus the type's name."""
    return {**edge_state(edge), "type": edge.type.name, "symmetric": bool(edge.type.symmetric)}


def add_relationships(
        db: Session,
        dataset_id: int,
        subject_id: int,
        object_ids,
        type_id: int | None = None,
        type_name: str | None = None,
        username: str | None = None,
        source: str = SOURCE_USER,
        status: str = STATUS_CONFIRMED,
        model_key: str | None = None,
        confidence: float | None = None,
) -> tuple[list[ObjectRelationships], list[dict]]:
    """Connect ``subject`` to each of ``object_ids`` with one relationship.

    Selecting several objects fans out from the subject: A→B, A→C. Edges that already
    exist are left alone. Name the type by id, or by name (created if new).

    Returns ``(edges, created_states)``: every requested edge (new or already there)
    and the states of the ones actually created, for the undo history.
    """
    _check_provenance(source, status)
    if (type_id is None) == (type_name is None):
        raise InvalidSceneGraphError("Name the relationship by exactly one of type_id or type_name.")
    rel_type = (get_relationship_type(db, dataset_id, type_id) if type_id is not None
                else ensure_relationship_type(db, dataset_id, type_name, username=username))
    object_ids = [int(oid) for oid in object_ids]
    if not object_ids:
        raise InvalidSceneGraphError("A relationship needs at least one other object.")
    if int(subject_id) in object_ids:
        raise InvalidSceneGraphError("An object cannot be related to itself.")
    contours_in_dataset(db, dataset_id, [subject_id, *object_ids])

    edges, created = [], []
    for object_id in dict.fromkeys(object_ids):
        a, b = int(subject_id), object_id
        if rel_type.symmetric and a > b:
            a, b = b, a
        edge = db.query(ObjectRelationships).filter_by(type_id=rel_type.id, subject_id=a,
                                                       object_id=b).first()
        if edge is None:
            edge = ObjectRelationships(dataset_id=dataset_id, type_id=rel_type.id, subject_id=a,
                                       object_id=b, source=source, status=status,
                                       model_key=model_key, confidence=confidence,
                                       author_username=username)
            db.add(edge)
            db.flush()
            created.append(edge_state(edge))
        edges.append(edge)
    return edges, created


def get_relationship(db: Session, dataset_id: int, relationship_id: int) -> ObjectRelationships:
    edge = db.query(ObjectRelationships).filter_by(id=relationship_id, dataset_id=dataset_id).first()
    if edge is None:
        raise SceneGraphNotFoundError(f"Dataset {dataset_id} has no relationship {relationship_id}.")
    return edge


def delete_relationships(db: Session, dataset_id: int, relationship_ids) -> list[dict]:
    """Delete edges by id. Returns their states for the undo history."""
    deleted = []
    for relationship_id in relationship_ids:
        edge = get_relationship(db, dataset_id, int(relationship_id))
        deleted.append(edge_state(edge))
        db.delete(edge)
    db.flush()
    return deleted


def confirm_relationships(db: Session, dataset_id: int, relationship_ids,
                          username: str | None) -> list[ObjectRelationships]:
    """Accept suggested edges."""
    edges = [get_relationship(db, dataset_id, int(rid)) for rid in relationship_ids]
    for edge in edges:
        if edge.status != STATUS_CONFIRMED:
            edge.status = STATUS_CONFIRMED
            edge.author_username = username
    db.flush()
    return edges


def restore_edge(db: Session, state: dict) -> bool:
    """Re-insert an edge under its original id. False when it cannot come back
    (already there, or its type or one of its ends is gone)."""
    if db.query(ObjectRelationships.id).filter_by(id=state["id"]).first() is not None:
        return False
    if db.query(RelationshipTypes.id).filter_by(id=state["type_id"]).first() is None:
        return False
    ends = {state["subject_id"], state["object_id"]}
    if db.query(Contours.id).filter(Contours.id.in_(ends)).count() != len(ends):
        return False
    if db.query(ObjectRelationships.id).filter_by(
            type_id=state["type_id"], subject_id=state["subject_id"],
            object_id=state["object_id"]).first() is not None:
        return False
    from datetime import datetime
    fields = dict(state)
    fields["created_at"] = (datetime.fromisoformat(state["created_at"])
                            if state.get("created_at") else None)
    db.add(ObjectRelationships(**fields))
    db.flush()
    return True


def relationships_of_contours(db: Session, contour_ids) -> list[ObjectRelationships]:
    """Every edge with at least one end among ``contour_ids``."""
    ids = list(contour_ids)
    if not ids:
        return []
    return (db.query(ObjectRelationships)
            .filter((ObjectRelationships.subject_id.in_(ids))
                    | (ObjectRelationships.object_id.in_(ids)))
            .order_by(ObjectRelationships.id)
            .all())


# ---------------------------------------------------------------------------
# Moving graph data between objects (split and merge)
# ---------------------------------------------------------------------------

def transfer(db: Session, dataset_id: int, from_ids, to_id: int,
             move_relationships: bool) -> dict:
    """Carry attributes (and optionally relationships) from some objects to another.

    For the two outline operations that create or remove objects:

    * **split** keeps the original and adds a new half: transfer from the original to
      the new half, without moving relationships, so the half gets the same species
      and the relationships stay where they were drawn;
    * **merge** keeps the largest object and deletes the rest: transfer from the
      absorbed objects to the survivor *with* relationships, before deleting them.

    Attributes the target already has are kept; missing ones are filled from the
    sources in the order given. Moved edges that would become self-loops or
    duplicates are dropped.
    """
    from_ids = [int(fid) for fid in from_ids if int(fid) != int(to_id)]
    contours_in_dataset(db, dataset_id, [*from_ids, to_id])
    target = db.get(Contours, int(to_id))

    have = {row.attribute_id for row in db.query(ObjectAttributes).filter_by(contour_id=to_id)}
    copied = 0
    for from_id in from_ids:
        for row in db.query(ObjectAttributes).filter_by(contour_id=from_id).order_by(ObjectAttributes.id):
            if row.attribute_id in have or not row.definition.applies_to(target.label_id):
                continue
            state = value_row_state(row)
            apply_value_state(db, to_id, row.attribute_id, state)
            have.add(row.attribute_id)
            copied += 1

    moved = dropped = 0
    if move_relationships and from_ids:
        for edge in relationships_of_contours(db, from_ids):
            subject = to_id if edge.subject_id in from_ids else edge.subject_id
            obj = to_id if edge.object_id in from_ids else edge.object_id
            if edge.type.symmetric and subject > obj:
                subject, obj = obj, subject
            duplicate = db.query(ObjectRelationships.id).filter(
                ObjectRelationships.type_id == edge.type_id,
                ObjectRelationships.subject_id == subject,
                ObjectRelationships.object_id == obj,
                ObjectRelationships.id != edge.id,
            ).first()
            if subject == obj or duplicate is not None:
                db.delete(edge)
                dropped += 1
                continue
            edge.subject_id, edge.object_id = subject, obj
            moved += 1
    db.flush()
    return {"attributes_copied": copied, "relationships_moved": moved,
            "relationships_dropped": dropped}


# ---------------------------------------------------------------------------
# Tables for export and grouping
# ---------------------------------------------------------------------------

#: How a confirmed "looked at, cannot tell" value is written into exports and group
#: names. Distinct from an empty cell, which means nobody set the attribute.
CANNOT_TELL = "(cannot tell)"

#: Prefix that marks a quantification ``group_by`` as an object attribute rather than
#: an image-metadata key: ``group_by=attr:species``.
ATTRIBUTE_GROUP_PREFIX = "attr:"

#: Attribute types that can be a grouping axis, as for metadata keys.
_GROUPABLE_ATTRIBUTE_TYPES = {MetadataValueType.CATEGORICAL.value, MetadataValueType.BOOLEAN.value}


def assert_groupable_attribute(db: Session, dataset_id: int, name: str) -> str:
    """Check that an attribute can split a quantification, returning its name.

    :raises InvalidSceneGraphError: if it does not exist or has too many distinct values.
    """
    definition = find_attribute(db, dataset_id, name)
    if definition is None:
        raise InvalidSceneGraphError(f"This dataset has no attribute '{name}'.")
    if definition.value_type not in _GROUPABLE_ATTRIBUTE_TYPES:
        raise InvalidSceneGraphError(
            f"'{definition.name}' is a {definition.value_type} attribute, which has too many "
            f"distinct values to group by. Group by a category or yes/no attribute instead."
        )
    return definition.name


def attribute_columns(db: Session, dataset_id: int,
                      contour_ids) -> tuple[dict[int, dict[str, str | None]], list[str]]:
    """``({contour_id: {name: value}}, names)`` for exporting given attributes.

    Every attribute of the dataset is a column, used or not, so the columns of an
    export do not depend on which objects it happens to contain. Only confirmed values
    are exported, and only for objects in the attribute's scope. "Cannot tell" is
    written as :data:`CANNOT_TELL`; not set stays empty.
    """
    names = [definition.name for definition in list_attributes(db, dataset_id)]
    values: dict[int, dict[str, str | None]] = defaultdict(dict)
    ids = list(contour_ids)
    if ids:
        for contour_id, entries in attributes_of_contours(db, ids).items():
            for entry in entries:
                if entry["status"] != STATUS_CONFIRMED or entry["out_of_scope"]:
                    continue
                values[contour_id][entry["name"]] = (CANNOT_TELL if entry["cannot_tell"]
                                                     else entry["value"])
    return values, names


def relationship_columns(db: Session, dataset_id: int,
                         contour_ids) -> tuple[dict[int, dict[str, str]], list[str]]:
    """``({contour_id: {column: "id;id"}}, columns)`` for exporting named relationships.

    Two columns per relationship type: ``<name>`` lists the objects this one points to,
    ``<name>_of`` the objects pointing to it (a symmetric type has just ``<name>``, with
    both directions). Ids are ``;``-separated so the cell stays one CSV field.
    """
    types = list_relationship_types(db, dataset_id)
    columns = []
    for rel_type in types:
        columns.append(rel_type.name)
        if not rel_type.symmetric:
            columns.append(f"{rel_type.name}_of")
    out: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    ids = set(contour_ids)
    for edge in relationships_of_contours(db, ids):
        if edge.status != STATUS_CONFIRMED:
            continue
        name, symmetric = edge.type.name, edge.type.symmetric
        if edge.subject_id in ids:
            out[edge.subject_id][name].append(edge.object_id)
        if edge.object_id in ids:
            out[edge.object_id][name if symmetric else f"{name}_of"].append(edge.subject_id)
    return ({cid: {col: ";".join(map(str, sorted(v))) for col, v in cols.items()}
             for cid, cols in out.items()}, columns)


# ---------------------------------------------------------------------------
# Reading one image's graph
# ---------------------------------------------------------------------------

def image_scene_graph(db: Session, image_id: int) -> dict:
    """One image as a scene graph: the scene, its objects and every edge between them.

    The scene's attributes are the image's effective metadata. Objects carry their
    label, given attributes and the geometry metrics. Edges are ``part_of`` (from
    nesting) plus every named relationship touching an object of the image; an edge
    whose other end is in another image is included, with that end marked foreign.
    """
    # Local import: the metadata access module imports this one's neighbours.
    from app.database.contour_metrics import ContourMetrics
    from app.services.database_access.image_metadata import get_metadata

    image = db.get(Images, image_id)
    if image is None:
        raise SceneGraphNotFoundError(f"Image {image_id} does not exist.")

    contours = (db.query(Contours, Labels.name)
                .join(Masks, Masks.id == Contours.mask_id)
                .outerjoin(Labels, Labels.id == Contours.label_id)
                .filter(Masks.image_id == image_id, Contours.temporary == False)  # noqa: E712
                .order_by(Contours.id)
                .all())
    ids = [contour.id for contour, _ in contours]
    attributes = attributes_of_contours(db, ids)

    metrics: dict[int, dict[str, list]] = defaultdict(dict)
    if ids:
        for row in (db.query(ContourMetrics)
                    .filter(ContourMetrics.contour_id.in_(ids), ContourMetrics.stale == False)  # noqa: E712
                    .order_by(ContourMetrics.metric_key, ContourMetrics.component)):
            metrics[row.contour_id].setdefault(row.metric_key, []).append(row.value)

    objects = [{
        "id": contour.id,
        "label_id": contour.label_id,
        "label": label_name,
        "parent_id": contour.parent_id,
        "author_username": contour.author_username,
        "origin": contour.origin,
        "reviewed": bool(contour.reviewed_by),
        "attributes": attributes.get(contour.id, []),
        "metrics": {key: (values[0] if len(values) == 1 else values)
                    for key, values in metrics.get(contour.id, {}).items()},
    } for contour, label_name in contours]

    on_image = set(ids)
    edges = [{"type": "part_of", "subject_id": contour.id, "object_id": contour.parent_id,
              "builtin": True}
             for contour, _ in contours if contour.parent_id is not None]
    for edge in relationships_of_contours(db, ids):
        entry = edge_to_dict(edge)
        entry["builtin"] = False
        entry["foreign_ends"] = [end for end in (edge.subject_id, edge.object_id)
                                 if end not in on_image]
        edges.append(entry)

    return {
        "scene": {"image_id": image.id, "dataset_id": image.dataset_id,
                  "file_name": image.file_name, "metadata": get_metadata(db, image.id)},
        "objects": objects,
        "relationships": edges,
    }

"""Read a model's predictions in a dataset's own labels.

A model with a fixed label space declares its classes (``ModelInfo.classes``) and
names the class of every prediction (``Contour.model_class``). This module decides,
per dataset, what each class means there:

1. a saved choice (``model_label_mappings``): one of the dataset's labels, or discard;
2. otherwise the dataset label with the class's name, if there is exactly that label
   (case-insensitive) -- so a model whose classes match a dataset needs no setup;
3. otherwise nothing: the prediction is kept, unlabelled, for a person to sort out.

:func:`apply_mapping` rewrites a model's output with that resolution; the planner
uses :func:`mapped_label_ids` as the labels a classed model can be bound to.
Predictions without a ``model_class`` (prompted, exemplar-driven or legacy fine-tuned
models) pass through untouched.
"""
from __future__ import annotations

import time
from logging import getLogger
from typing import Iterable, Optional

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.labels import Labels
from app.database.model_label_mappings import DISCARD, MAP, ModelLabelMappings
from app.schemas.model_label_mapping import (
    ClassMappingRead,
    ClassMappingWrite,
    ModelLabelMappingRead,
)
from iquana_toolbox.schemas.database.contours import Contour
from iquana_toolbox.schemas.model_info import ModelClass

logger = getLogger(__name__)

#: Model metadata changes only when a model is re-registered; a minute is plenty fresh
#: and keeps an interactive run from paying an MLflow round trip every time.
_INFO_TTL_SECONDS = 60.0
_info_cache: dict[str, tuple[float, dict]] = {}


def _model_info(registry_key: str) -> dict:
    from app.services.model_registry import get_model_info

    cached = _info_cache.get(registry_key)
    if cached is not None and time.monotonic() - cached[0] < _INFO_TTL_SECONDS:
        return cached[1]
    info = get_model_info(registry_key)
    _info_cache[registry_key] = (time.monotonic(), info)
    return info


def classes_of(info: dict) -> list[ModelClass]:
    """The label space declared in a ``model_info`` dict; empty for class-agnostic models."""
    return [ModelClass.model_validate(entry) for entry in info.get("classes") or []]


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def resolve(
    db: Session, dataset_id: int, registry_key: str, classes: Iterable[ModelClass]
) -> list[ClassMappingRead]:
    """What each class means in the dataset: saved choice, else same-name label, else nothing."""
    saved = {
        row.model_class: row
        for row in db.query(ModelLabelMappings).filter_by(
            dataset_id=dataset_id, model_registry_key=registry_key
        )
    }
    label_by_name = {
        name.strip().lower(): label_id
        for label_id, name in db.query(Labels.id, Labels.name).filter_by(dataset_id=dataset_id)
    }
    resolved: list[ClassMappingRead] = []
    for model_class in classes:
        row = saved.get(model_class.key)
        if row is not None:
            action = DISCARD if row.action == DISCARD else MAP
            resolved.append(ClassMappingRead(
                model_class=model_class, action=action,
                label_id=row.label_id if action == MAP else None, source="saved",
                attribute_id=row.attribute_id if action == MAP else None,
                attribute_value=row.attribute_value if action == MAP and row.attribute_id else None,
            ))
            continue
        suggested = label_by_name.get(model_class.name.strip().lower())
        resolved.append(ClassMappingRead(
            model_class=model_class,
            action=MAP if suggested is not None else "unmapped",
            label_id=suggested,
            source="suggested" if suggested is not None else "none",
        ))
    return resolved


def mapped_label_ids(resolved: Iterable[ClassMappingRead]) -> list[int]:
    """The dataset labels a classed model's output lands in."""
    return sorted({entry.label_id for entry in resolved if entry.action == MAP})


def get_mapping(db: Session, dataset_id: int, registry_key: str) -> ModelLabelMappingRead:
    info = _model_info(registry_key)
    return ModelLabelMappingRead(
        dataset_id=dataset_id,
        model_registry_key=registry_key,
        model_name=info.get("name") or registry_key,
        classes=resolve(db, dataset_id, registry_key, classes_of(info)),
    )


# --------------------------------------------------------------------------- #
# Applying it to predictions
# --------------------------------------------------------------------------- #
def apply_mapping(
    db: Session, dataset_id: int, registry_key: str, contours: list[Contour]
) -> list[Contour]:
    """Label a model's output in the dataset's terms.

    Mapped classes get their dataset label, unmapped ones none. A discarded prediction is
    removed, but its children are kept and take its place: discarding "cell" while
    keeping "nucleus" leaves the nuclei, at the level the cells were.
    """
    if not any(_has_model_class(contour) for contour in contours):
        return contours
    classes = classes_of(_model_info(registry_key))
    if not classes:
        logger.warning("Model '%s' names classes but declares no label space.", registry_key)
        return contours
    by_class = {entry.model_class.key: entry for entry in resolve(db, dataset_id, registry_key, classes)}
    return _apply(contours, by_class)


def _has_model_class(contour: Contour) -> bool:
    return contour.model_class is not None or any(_has_model_class(c) for c in contour.children)


def _apply(contours: list[Contour], by_class: dict[str, ClassMappingRead]) -> list[Contour]:
    kept: list[Contour] = []
    for contour in contours:
        contour.children = _apply(contour.children, by_class)
        if contour.model_class is None:
            kept.append(contour)
            continue
        entry = by_class.get(contour.model_class)
        if entry is not None and entry.action == DISCARD:
            kept.extend(contour.children)
            continue
        contour.label_id = entry.label_id if entry is not None and entry.action == MAP else None
        kept.append(contour)
    return kept


def flatten(contours: list[Contour]) -> list[Contour]:
    """Every contour of a tree, depth first, with the children links cut."""
    flat: list[Contour] = []
    for contour in contours:
        children, contour.children = contour.children, []
        flat.append(contour)
        flat.extend(flatten(children))
    return flat


# --------------------------------------------------------------------------- #
# Saving choices
# --------------------------------------------------------------------------- #
def save_mapping(
    db: Session,
    dataset_id: int,
    registry_key: str,
    changes: list[ClassMappingWrite],
    username: Optional[str],
) -> ModelLabelMappingRead:
    """Apply the changes in one transaction and return the resulting mapping.

    ``create`` makes a label named after the class (or ``label_name``) and maps the class
    to it. When the class is ``part_of`` another whose label is known by then -- mapped
    before, or created in the same call -- the new label is nested under that label, so
    the dataset's hierarchy follows the model's.
    """
    info = _model_info(registry_key)
    classes = {model_class.key: model_class for model_class in classes_of(info)}
    unknown = [change.model_class for change in changes if change.model_class not in classes]
    if unknown:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Model '{registry_key}' has no class(es) {unknown}; it predicts {sorted(classes)}.",
        )
    dataset_labels = {
        label.id: label for label in db.query(Labels).filter_by(dataset_id=dataset_id)
    }

    current = {
        entry.model_class.key: entry.label_id
        for entry in resolve(db, dataset_id, registry_key, classes.values())
        if entry.action == MAP
    }
    # Parents first, so a created child can nest under its parent's new label.
    for change in sorted(changes, key=lambda c: _depth(classes, c.model_class)):
        row = db.query(ModelLabelMappings).filter_by(
            dataset_id=dataset_id, model_registry_key=registry_key, model_class=change.model_class,
        ).first()
        if change.action == "unmapped":
            if row is not None:
                db.delete(row)
            current.pop(change.model_class, None)
            continue

        label_id = None
        if change.action == "map":
            if change.label_id not in dataset_labels:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"Label {change.label_id} is not a label of dataset {dataset_id}.",
                )
            label_id = change.label_id
        elif change.action == "create":
            label = _create_label(
                db, dataset_id, dataset_labels,
                name=(change.label_name or classes[change.model_class].name).strip(),
                parent_id=current.get(classes[change.model_class].part_of),
            )
            label_id = label.id

        if row is None:
            row = ModelLabelMappings(
                dataset_id=dataset_id, model_registry_key=registry_key, model_class=change.model_class,
            )
            db.add(row)
        row.action = DISCARD if change.action == DISCARD else MAP
        row.label_id = label_id
        row.attribute_id, row.attribute_value = _attribute_for(
            db, dataset_id, change, classes[change.model_class], label_id)
        row.updated_by = username
        if label_id is None:
            current.pop(change.model_class, None)
        else:
            current[change.model_class] = label_id
    db.commit()
    return get_mapping(db, dataset_id, registry_key)


def _attribute_for(db: Session, dataset_id: int, change: ClassMappingWrite,
                   model_class: ModelClass, label_id: Optional[int]) -> tuple[Optional[int], Optional[str]]:
    """Validate the attribute value a class mapping sets, by the attribute's own rules."""
    if change.attribute_id is None:
        return None, None
    from app.exceptions import InvalidSceneGraphError, SceneGraphNotFoundError
    from app.services.database_access import scene_graph as graph_db
    try:
        definition = graph_db.get_attribute(db, dataset_id, change.attribute_id)
        if not definition.applies_to(label_id):
            raise InvalidSceneGraphError(
                f"Attribute '{definition.name}' does not apply to the label class "
                f"'{model_class.key}' is mapped to.")
        value, _ = graph_db.coerce_value(definition, change.attribute_value or model_class.name)
    except (InvalidSceneGraphError, SceneGraphNotFoundError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    return definition.id, value


def apply_class_attributes(db: Session, dataset_id: int, registry_key: str, contour_ids) -> int:
    """Give freshly saved model output the attribute values its classes are mapped to.

    Runs after the objects are saved (it needs their ids), reading the class each one
    was predicted as from ``contours.model_class``. Values arrive as *suggestions*
    carrying the model and its confidence, and never overwrite a value that is already
    there. Does not commit. Returns how many values were written.
    """
    from app.database.contours import Contours
    from app.database.scene_graph import (
        SOURCE_MODEL,
        STATUS_SUGGESTED,
        ObjectAttributeDefinitions,
        ObjectAttributes,
    )
    from app.services.metadata_types import coerce

    ids = list(contour_ids)
    if not ids:
        return 0
    mapped = {
        row.model_class: row
        for row in db.query(ModelLabelMappings).filter(
            ModelLabelMappings.dataset_id == dataset_id,
            ModelLabelMappings.model_registry_key == registry_key,
            ModelLabelMappings.attribute_id.is_not(None),
        )
    }
    if not mapped:
        return 0
    written = 0
    for contour in db.query(Contours).filter(Contours.id.in_(ids), Contours.model_class.in_(mapped)):
        row = mapped[contour.model_class]
        if contour.label_id != row.label_id:
            continue  # moved by hierarchy fitting or a person; the class no longer applies
        if db.query(ObjectAttributes.id).filter_by(contour_id=contour.id,
                                                   attribute_id=row.attribute_id).first():
            continue
        definition = db.get(ObjectAttributeDefinitions, row.attribute_id)
        _, number = coerce(row.attribute_value, definition.value_type)
        db.add(ObjectAttributes(
            contour_id=contour.id, attribute_id=row.attribute_id, value=row.attribute_value,
            value_num=number, source=SOURCE_MODEL, status=STATUS_SUGGESTED,
            model_key=registry_key, confidence=contour.confidence_score,
            author_username=contour.author_username,
        ))
        written += 1
    db.flush()
    return written


def _depth(classes: dict[str, ModelClass], key: str) -> int:
    depth, seen = 0, {key}
    parent = classes[key].part_of if key in classes else None
    while parent is not None and parent not in seen:
        depth += 1
        seen.add(parent)
        parent = classes[parent].part_of if parent in classes else None
    return depth


def _create_label(
    db: Session, dataset_id: int, dataset_labels: dict[int, Labels], *, name: str, parent_id: Optional[int]
) -> Labels:
    if not name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "A new label needs a name.")
    clash = next((label for label in dataset_labels.values() if label.name.lower() == name.lower()), None)
    if clash is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"The dataset already has a label '{clash.name}'; map the class to it instead.",
        )
    next_value = (db.query(func.max(Labels.value)).filter_by(dataset_id=dataset_id).scalar() or 0) + 1
    label = Labels(dataset_id=dataset_id, name=name, parent_id=parent_id, value=next_value)
    db.add(label)
    db.flush()
    dataset_labels[label.id] = label
    return label

"""Turn is-a labels into one label plus an attribute.

Datasets annotated before attributes existed encoded classification as labels:
``Acropora`` and ``Pocillopora`` side by side, where the outline only ever said *coral
fragment*. Under the scene-graph rule (a label is what the outline is; anything that
would not change the outline is an attribute) those become one label, ``Coral
fragment``, and a categorical attribute ``species`` with one value per old label.

The conversion keeps every object (same ids, children, review history, outlines). It
relabels them, writes the value each old label implied, and points everything that
named an old label at the new one before the old labels are deleted: child labels,
model label mappings (which also gain the attribute value), quantification profiles and
attribute scopes. A dry run reports all of that without writing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.contours import Contours
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.model_label_mappings import ModelLabelMappings
from app.database.quantification_profiles import QuantificationProfiles
from app.database.scene_graph import (
    SOURCE_IMPORT,
    STATUS_CONFIRMED,
    ObjectAttributeDefinitions,
    ObjectAttributes,
)
from app.exceptions import InvalidSceneGraphError, SceneGraphNotFoundError
from app.services import hierarchy_cache
from app.services.database_access import scene_graph as graph_db
from app.services.metadata_types import MetadataValueType


@dataclass
class ConversionReport:
    dry_run: bool
    target_label: dict
    attribute: dict
    labels: list[dict] = field(default_factory=list)
    objects_relabelled: int = 0
    values_written: int = 0
    #: Objects that already carried a different value of the attribute. They are
    #: relabelled, but their value is kept, since someone set it on purpose.
    conflicts: list[dict] = field(default_factory=list)
    child_labels_moved: list[str] = field(default_factory=list)
    model_mappings_repointed: int = 0
    profiles_updated: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def convert_labels_to_attribute(
        db: Session,
        dataset_id: int,
        label_ids: list[int],
        attribute_name: str,
        target_label_id: int | None = None,
        target_label_name: str | None = None,
        dry_run: bool = True,
        username: str | None = None,
) -> ConversionReport:
    """Fold sibling labels into one label and an attribute value each.

    Args:
        label_ids: The is-a labels to fold, e.g. the species labels. They must share a
            parent, since their objects are nested where that parent's objects are.
        attribute_name: The categorical attribute that takes their names as values.
            Created if the dataset has none of that name; reused (and its vocabulary
            extended, if it is locked) otherwise.
        target_label_id / target_label_name: The label the objects get: an existing
            sibling of the folded labels, or a new one created next to them.
        dry_run: Report what would happen, and write nothing.

    :raises InvalidSceneGraphError: when the labels or the target do not fit together.
    :raises SceneGraphNotFoundError: for labels outside the dataset.
    """
    sources = db.query(Labels).filter(Labels.id.in_(label_ids), Labels.dataset_id == dataset_id).all()
    missing = set(label_ids) - {label.id for label in sources}
    if missing:
        raise SceneGraphNotFoundError(f"Label(s) {sorted(missing)} do not exist in dataset {dataset_id}.")
    if not sources:
        raise InvalidSceneGraphError("Name at least one label to fold.")
    parents = {label.parent_id for label in sources}
    if len(parents) != 1:
        raise InvalidSceneGraphError(
            "The labels must share a parent: their objects sit inside that parent's objects, "
            "and the label they become has to sit there too.")
    parent_id = parents.pop()

    if (target_label_id is None) == (not target_label_name):
        raise InvalidSceneGraphError("Name the target by exactly one of target_label_id or target_label_name.")
    target = None
    if target_label_id is not None:
        target = db.query(Labels).filter_by(id=target_label_id, dataset_id=dataset_id).first()
        if target is None:
            raise SceneGraphNotFoundError(f"Label {target_label_id} does not exist in dataset {dataset_id}.")
        if target.id in label_ids:
            raise InvalidSceneGraphError("The target cannot be one of the labels being folded.")
        if target.parent_id != parent_id:
            raise InvalidSceneGraphError("The target label must sit next to the folded labels.")
    else:
        name = target_label_name.strip()
        target = db.query(Labels).filter_by(dataset_id=dataset_id, name=name).first()
        if target is not None:
            raise InvalidSceneGraphError(
                f"Label '{name}' already exists; pass its id as target_label_id instead.")

    definition = graph_db.find_attribute(db, dataset_id, attribute_name)
    if definition is not None and definition.value_type != MetadataValueType.CATEGORICAL.value:
        raise InvalidSceneGraphError(
            f"Attribute '{definition.name}' is {definition.value_type}; label names need a category.")

    source_ids = [label.id for label in sources]
    contours = (db.query(Contours)
                .join(Masks, Masks.id == Contours.mask_id)
                .join(Images, Images.id == Masks.image_id)
                .filter(Images.dataset_id == dataset_id, Contours.label_id.in_(source_ids))
                .all())
    existing_values = {}
    if definition is not None and contours:
        existing_values = {
            row.contour_id: row
            for row in db.query(ObjectAttributes).filter(
                ObjectAttributes.attribute_id == definition.id,
                ObjectAttributes.contour_id.in_([c.id for c in contours]))
        }
    name_of = {label.id: label.name for label in sources}
    child_labels = db.query(Labels).filter(Labels.parent_id.in_(source_ids)).all()
    mappings = db.query(ModelLabelMappings).filter(
        ModelLabelMappings.dataset_id == dataset_id,
        ModelLabelMappings.label_id.in_(source_ids)).all()
    profiles = [profile for profile in db.query(QuantificationProfiles).filter_by(dataset_id=dataset_id)
                if any(set(entry.get("label_ids") or []) & set(source_ids)
                       for entry in profile.entries or [])]

    report = ConversionReport(
        dry_run=dry_run,
        target_label={"id": target.id if target else None,
                      "name": target.name if target else target_label_name.strip(),
                      "created": target is None},
        attribute={"id": definition.id if definition else None,
                   "name": definition.name if definition else attribute_name.strip(),
                   "created": definition is None},
        labels=[{"id": label.id, "name": label.name,
                 "objects": sum(1 for c in contours if c.label_id == label.id)} for label in sources],
        objects_relabelled=len(contours),
        child_labels_moved=sorted(label.name for label in child_labels),
        model_mappings_repointed=len(mappings),
        profiles_updated=sorted(profile.name for profile in profiles),
    )
    for contour in contours:
        row = existing_values.get(contour.id)
        if row is not None and row.value != name_of[contour.label_id]:
            report.conflicts.append({"contour_id": contour.id, "label": name_of[contour.label_id],
                                     "existing_value": row.value})
    report.values_written = len(contours) - len(existing_values)
    if dry_run:
        return report

    # --- Write -----------------------------------------------------------------
    if target is None:
        next_value = (db.query(func.max(Labels.value)).filter_by(dataset_id=dataset_id).scalar() or 0) + 1
        target = Labels(dataset_id=dataset_id, name=target_label_name.strip(), parent_id=parent_id,
                        value=next_value)
        db.add(target)
        db.flush()
        report.target_label["id"] = target.id

    if definition is None:
        definition = graph_db.create_attribute(
            db, dataset_id, attribute_name, value_type=MetadataValueType.CATEGORICAL.value,
            options=[label.name for label in sources], label_ids=[target.id], username=username)
        report.attribute["id"] = definition.id
    else:
        if definition.options:
            known = set(definition.option_values)
            definition.options = list(definition.options) + [
                {"value": label.name, "code": None, "description": None}
                for label in sources if label.name not in known]
        if definition.labels and target not in definition.labels:
            definition.labels = list(definition.labels) + [target]

    for contour in contours:
        if contour.id not in existing_values:
            graph_db.apply_value_state(db, contour.id, definition.id, {
                "value": name_of[contour.label_id], "value_num": None, "source": SOURCE_IMPORT,
                "status": STATUS_CONFIRMED, "model_key": None, "confidence": None,
                "author_username": contour.author_username,
            })
        contour.label_id = target.id

    for label in child_labels:
        label.parent_id = target.id
    for mapping in mappings:
        if mapping.attribute_id is None:
            mapping.attribute_id, mapping.attribute_value = definition.id, name_of[mapping.label_id]
        mapping.label_id = target.id
    for profile in profiles:
        entries = []
        for entry in profile.entries or []:
            entry = dict(entry)
            ids = entry.get("label_ids")
            if ids:
                folded = [target.id if lid in source_ids else lid for lid in ids]
                entry["label_ids"] = list(dict.fromkeys(folded))
            entries.append(entry)
        profile.entries = entries
    # Other attributes scoped to a folded label now apply to the target instead.
    for other in db.query(ObjectAttributeDefinitions).filter_by(dataset_id=dataset_id):
        if any(label.id in source_ids for label in other.labels):
            other.labels = list({label.id: label for label in
                                 [l for l in other.labels if l.id not in source_ids] + [target]}.values())
    db.flush()

    for label in sources:
        db.delete(label)
    db.flush()
    for mask_id in {contour.mask_id for contour in contours}:
        hierarchy_cache.invalidate(mask_id)
    return report

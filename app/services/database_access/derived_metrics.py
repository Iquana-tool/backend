"""Database access for a dataset's derived measurements.

Definitions are checked as a set (see :func:`app.services.derived_metrics.resolve_definitions`)
on every write, so a dataset never stores a formula that clashes with its neighbours: one
that would close a cycle, refer to a measurement that does not exist, or add an area to a
length. Things that break *later* -- a label deleted out from under a formula -- are not
prevented; the definition is kept and reports the problem when listed.
"""
from __future__ import annotations

import copy
from typing import Any

from sqlalchemy.orm import Session

from app.database.derived_metrics import DerivedMetrics
from app.database.labels import Labels
from app.services.derived_metrics import (
    DerivedMetricDefinition,
    DerivedMetricError,
    ResolvedDerivedMetric,
    check_key_available,
    direct_metric_inputs,
    referenced_keys,
    referenced_label_ids,
    resolve_definitions,
)


def list_rows(db: Session, dataset_id: int) -> list[DerivedMetrics]:
    """A dataset's definitions, oldest first (the order they appear in as columns)."""
    return (
        db.query(DerivedMetrics)
        .filter(DerivedMetrics.dataset_id == dataset_id)
        .order_by(DerivedMetrics.id)
        .all()
    )


def get_row(db: Session, dataset_id: int, metric_id: int) -> DerivedMetrics | None:
    return (
        db.query(DerivedMetrics)
        .filter(DerivedMetrics.id == metric_id, DerivedMetrics.dataset_id == dataset_id)
        .first()
    )


def to_definition(row: DerivedMetrics) -> DerivedMetricDefinition:
    return DerivedMetricDefinition(
        key=row.key,
        name=row.name,
        description=row.description or "",
        label_ids=row.label_ids,
        expression=row.expression,
    )


def label_parents(db: Session, dataset_id: int) -> dict[int, int | None]:
    """``{label_id: parent label id}`` for every label of the dataset."""
    return {
        label_id: parent_id
        for label_id, parent_id in db.query(Labels.id, Labels.parent_id)
        .filter(Labels.dataset_id == dataset_id)
        .all()
    }


def resolve_for_dataset(
        db: Session,
        dataset_id: int,
        candidate: DerivedMetricDefinition | None = None,
        candidate_id: int | None = None,
) -> tuple[list[ResolvedDerivedMetric], list[ResolvedDerivedMetric], dict[int, int | None]]:
    """Check the dataset's definitions, optionally with one added or replaced.

    ``candidate`` is a definition not yet saved: a new one (``candidate_id`` None) is
    appended, an edit (``candidate_id`` set) takes its stored row's place. That is how a
    write is checked before it happens and how the editor previews unsaved work.

    :returns: ``(all, evaluation_order, label_parents)``, see ``resolve_definitions``.
    """
    entries: list[tuple[int | None, DerivedMetricDefinition]] = []
    for row in list_rows(db, dataset_id):
        if candidate is not None and candidate_id is not None and row.id == candidate_id:
            entries.append((row.id, candidate))
        else:
            entries.append((row.id, to_definition(row)))
    if candidate is not None and candidate_id is None:
        entries.append((None, candidate))
    parents = label_parents(db, dataset_id)
    everything, order = resolve_definitions(entries, set(parents))
    return everything, order, parents


def required_metric_keys(db: Session, dataset_id: int) -> set[str]:
    """Registry keys the dataset's usable formulas read, so their tiers get computed."""
    _, order, _ = resolve_for_dataset(db, dataset_id)
    return {key for key, _ in direct_metric_inputs(item.definition for item in order)}


def serialize(item: ResolvedDerivedMetric, row: DerivedMetrics | None = None) -> dict[str, Any]:
    """The API shape of one definition, with what checking it found."""
    data = item.definition.model_dump()
    data.update({
        "id": item.id,
        "unit_exponent": item.exponent,
        "problems": item.problems,
    })
    if row is not None:
        data["created_by"] = row.created_by
        data["updated_at"] = row.updated_at.isoformat() if row.updated_at else None
    return data


def list_serialized(db: Session, dataset_id: int) -> list[dict[str, Any]]:
    rows = {row.id: row for row in list_rows(db, dataset_id)}
    everything, _, _ = resolve_for_dataset(db, dataset_id)
    return [serialize(item, rows.get(item.id)) for item in everything]


def _checked(db: Session, dataset_id: int, definition: DerivedMetricDefinition,
             definition_id: int | None) -> ResolvedDerivedMetric:
    """Resolve the dataset with ``definition`` in it; raise if it is not usable."""
    everything, _, _ = resolve_for_dataset(db, dataset_id, definition, definition_id)
    item = next(i for i in everything if i.definition.key == definition.key and i.id == definition_id)
    if not item.ok:
        raise DerivedMetricError(item.problems)
    return item


def create(db: Session, dataset_id: int, definition: DerivedMetricDefinition,
           username: str | None) -> dict[str, Any]:
    """Store a new definition after checking it against the dataset's others.

    :raises DerivedMetricError: When the key is taken or the formula is not usable.
    """
    problems = check_key_available(definition.key, (row.key for row in list_rows(db, dataset_id)))
    if problems:
        raise DerivedMetricError(problems)
    item = _checked(db, dataset_id, definition, None)
    row = DerivedMetrics(
        dataset_id=dataset_id,
        key=definition.key,
        name=definition.name,
        description=definition.description,
        label_ids=definition.label_ids,
        expression=definition.expression.model_dump(),
        created_by=username,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    item.id = row.id
    return serialize(item, row)


def update(db: Session, row: DerivedMetrics, definition: DerivedMetricDefinition) -> dict[str, Any]:
    """Replace a definition's name, description, labels and formula.

    The key is fixed: other formulas and saved analyses refer to it by name.

    :raises DerivedMetricError: When the key changed or the new formula is not usable.
    """
    if definition.key != row.key:
        raise DerivedMetricError([f"The key of a derived measurement cannot change (it is '{row.key}'); "
                                  "other formulas and exports refer to it. Create a new one instead."])
    item = _checked(db, row.dataset_id, definition, row.id)
    row.name = definition.name
    row.description = definition.description
    row.label_ids = definition.label_ids
    row.expression = definition.expression.model_dump()
    db.commit()
    db.refresh(row)
    return serialize(item, row)


def dependents(db: Session, dataset_id: int, key: str) -> list[str]:
    """Keys of the definitions whose formula reads ``key``."""
    return [row.key for row in list_rows(db, dataset_id)
            if key in referenced_keys(to_definition(row).expression)]


def delete(db: Session, row: DerivedMetrics) -> None:
    """Delete a definition nothing else reads.

    :raises DerivedMetricError: When another formula still uses it.
    """
    users = [key for key in dependents(db, row.dataset_id, row.key) if key != row.key]
    if users:
        raise DerivedMetricError([f"'{row.key}' is used by {', '.join(repr(u) for u in users)}; "
                                  "change or delete those first."])
    db.delete(row)
    db.commit()


def _remap_labels(node: Any, mapping: dict[int, int]) -> Any:
    """A copy of a JSON formula with every rollup's label ids translated."""
    node = copy.deepcopy(node)
    stack = [node]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        if current.get("type") == "rollup" and current.get("label_ids"):
            current["label_ids"] = [mapping[i] for i in current["label_ids"]]
        for child in ("left", "right", "arg"):
            if child in current:
                stack.append(current[child])
    return node


def copy_from_dataset(db: Session, source_dataset_id: int, target_dataset_id: int,
                      keys: list[str] | None, username: str | None) -> list[dict[str, Any]]:
    """Copy definitions from one dataset into another, matching labels by name.

    Label names are unique within a dataset, so "Vessel" in the source becomes whatever id
    "Vessel" has in the target. A definition that names a label the target lacks is
    refused, as is one whose key the target already uses -- both say which. Definitions the
    copied ones read are copied along, so a formula never arrives without its inputs.

    :param keys: The definitions to copy; ``None`` copies every one.
    :raises DerivedMetricError: When anything cannot be copied; nothing is written then.
    """
    source_rows = {row.key: row for row in list_rows(db, source_dataset_id)}
    wanted = list(source_rows) if keys is None else list(keys)
    unknown = [key for key in wanted if key not in source_rows]
    if unknown:
        raise DerivedMetricError([f"The source dataset has no derived measurement {', '.join(map(repr, unknown))}."])

    # Pull in what the selection reads, dependencies first.
    selected: list[str] = []

    def include(key: str) -> None:
        if key in selected:
            return
        for dep in sorted(referenced_keys(to_definition(source_rows[key]).expression) & set(source_rows)):
            if dep != key:
                include(dep)
        selected.append(key)

    for key in wanted:
        include(key)

    source_names = {lid: name for lid, name in db.query(Labels.id, Labels.name)
                    .filter(Labels.dataset_id == source_dataset_id).all()}
    target_ids = {name: lid for lid, name in db.query(Labels.id, Labels.name)
                  .filter(Labels.dataset_id == target_dataset_id).all()}
    existing = {row.key for row in list_rows(db, target_dataset_id)}

    problems: list[str] = []
    translated: list[DerivedMetricDefinition] = []
    for key in selected:
        row = source_rows[key]
        definition = to_definition(row)
        if key in existing:
            problems.append(f"This dataset already has a derived measurement called '{key}'.")
            continue
        named = [source_names.get(i) for i in referenced_label_ids(definition)]
        missing = sorted({name or "(deleted label)" for name in named if name not in target_ids})
        if missing:
            problems.append(f"'{key}' uses labels this dataset does not have: {', '.join(missing)}.")
            continue
        mapping = {i: target_ids[source_names[i]] for i in referenced_label_ids(definition)}
        translated.append(DerivedMetricDefinition(
            key=definition.key,
            name=definition.name,
            description=definition.description,
            label_ids=[mapping[i] for i in definition.label_ids] if definition.label_ids is not None else None,
            expression=_remap_labels(row.expression, mapping),
        ))
    if problems:
        raise DerivedMetricError(problems)

    # Check the copies together with what the target already has before writing any.
    entries = [(row.id, to_definition(row)) for row in list_rows(db, target_dataset_id)]
    entries += [(None, definition) for definition in translated]
    everything, _ = resolve_definitions(entries, set(label_parents(db, target_dataset_id)))
    for item in everything:
        if item.id is None and not item.ok:
            problems.extend(f"'{item.definition.key}': {p}" for p in item.problems)
    if problems:
        raise DerivedMetricError(problems)

    for definition in translated:
        db.add(DerivedMetrics(
            dataset_id=target_dataset_id,
            key=definition.key,
            name=definition.name,
            description=definition.description,
            label_ids=definition.label_ids,
            expression=definition.expression.model_dump(),
            created_by=username,
        ))
    db.commit()
    return list_serialized(db, target_dataset_id)

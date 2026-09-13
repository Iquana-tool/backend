"""Read/write access to per-image metadata (see ``app.database.image_metadata``).

Everything that mutates goes through :func:`set_metadata_for_images`, so key/value
normalisation, type coercion and the upsert-or-delete rule exist in exactly one
place: the single-image editor, the bulk "tag these 40 images" action and the CSV
importer all behave identically.

Keys carry a type, declared per dataset in ``dataset_metadata_keys``. A key that
has never been declared is created on first write as ``categorical``, so a
dataset never has to be set up before it can be tagged and behaviour from before
types existed is unchanged.
"""
from __future__ import annotations

import re
from logging import getLogger

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.dataset_metadata_keys import DatasetMetadataKeys
from app.database.image_metadata import (
    MAX_KEY_LENGTH,
    MAX_VALUE_LENGTH,
    ImageMetadata,
)
from app.database.images import Images
from app.exceptions import DatasetNotFoundError, ImageNotFoundError, InvalidMetadataError
from app.services.metadata_types import (
    DEFAULT_TYPE,
    GROUPABLE_TYPES,
    ORDERED_TYPES,
    MetadataValueType,
    coerce,
)

logger = getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def normalize_key(key: str) -> str:
    """Canonical form of a metadata key: trimmed, inner whitespace collapsed.

    Case is preserved. Folding it would make ``Site`` and ``site`` the same key,
    which is usually what the user meant but occasionally is not, and the guess
    is unrecoverable once the row is written. The frontend suggests the dataset's
    existing keys instead, which prevents the duplicate before it is created.

    :raises InvalidMetadataError: if the key is empty or longer than
        :data:`~app.database.image_metadata.MAX_KEY_LENGTH`.
    """
    cleaned = _WHITESPACE.sub(" ", (key or "").strip())
    if not cleaned:
        raise InvalidMetadataError("A metadata key cannot be empty.")
    if len(cleaned) > MAX_KEY_LENGTH:
        raise InvalidMetadataError(
            f"Metadata key '{cleaned[:20]}...' is longer than {MAX_KEY_LENGTH} characters."
        )
    return cleaned


def normalize_value(value: str) -> str:
    """Trim a metadata value, rejecting one that is too long.

    Unlike keys, inner whitespace is left alone — a value can legitimately be a
    short free-text note. An empty value is not an error here: callers treat it
    as "remove this key" (see :func:`set_metadata_for_images`).

    :raises InvalidMetadataError: if the value is longer than
        :data:`~app.database.image_metadata.MAX_VALUE_LENGTH`.
    """
    cleaned = "" if value is None else str(value).strip()
    if len(cleaned) > MAX_VALUE_LENGTH:
        raise InvalidMetadataError(
            f"Metadata value '{cleaned[:20]}...' is longer than "
            f"{MAX_VALUE_LENGTH} characters."
        )
    return cleaned


def _normalize_entries(entries: dict[str, str]) -> dict[str, str]:
    """Normalise a whole ``{key: value}`` payload, rejecting duplicate keys.

    Two keys that normalise to the same thing (``"site"`` and ``" site "``) would
    otherwise silently drop one of the two values depending on dict ordering.
    """
    normalized: dict[str, str] = {}
    for raw_key, raw_value in (entries or {}).items():
        key = normalize_key(raw_key)
        if key in normalized:
            raise InvalidMetadataError(f"Duplicate metadata key '{key}' in the request.")
        normalized[key] = normalize_value(raw_value)
    return normalized


# ---------------------------------------------------------------------------
# Key descriptors
# ---------------------------------------------------------------------------

def list_keys(db: Session, dataset_id: int) -> list[DatasetMetadataKeys]:
    """Every declared key of a dataset, alphabetically."""
    return (
        db.query(DatasetMetadataKeys)
        .filter_by(dataset_id=dataset_id)
        .order_by(func.lower(DatasetMetadataKeys.key))
        .all()
    )


def get_key(db: Session, dataset_id: int, key: str) -> DatasetMetadataKeys | None:
    """One key's descriptor, or None if it has never been used or declared."""
    return (
        db.query(DatasetMetadataKeys)
        .filter_by(dataset_id=dataset_id, key=normalize_key(key))
        .first()
    )


def assert_groupable(db: Session, dataset_id: int, key: str) -> str:
    """Check that a key can be a grouping axis, returning its normalised name.

    Only some types can: a number or a date is near-unique per image, so grouping
    by one would draw a bar per image rather than a comparison, and free text is
    a note rather than a vocabulary. Refusing here — with a message naming the
    type — is much kinder than rendering that chart.

    :raises InvalidMetadataError: if the key does not exist or is not groupable.
    """
    normalized = normalize_key(key)
    descriptor = get_key(db, dataset_id, normalized)
    if descriptor is None:
        raise InvalidMetadataError(f"This dataset has no metadata key '{normalized}'.")
    value_type = MetadataValueType(descriptor.value_type)
    if value_type not in GROUPABLE_TYPES:
        raise InvalidMetadataError(
            f"'{normalized}' is a {value_type.value} key, which has too many distinct "
            f"values to group by. Group by a category or yes/no key instead."
        )
    return normalized


def _dataset_id_for_images(db: Session, image_ids: list[int]) -> int:
    """The single dataset a batch of images belongs to.

    Key descriptors are per dataset, so a write that spans two datasets has no
    single type to validate against. The routes already refuse such a batch on
    permission grounds; this is the invariant stated where it is relied on.
    """
    dataset_ids = {
        row.dataset_id for row in
        db.query(Images.dataset_id).filter(Images.id.in_(image_ids)).distinct()
    }
    if not dataset_ids:
        raise ImageNotFoundError("None of the given images exist.")
    if len(dataset_ids) > 1:
        raise InvalidMetadataError(
            "Cannot write metadata across datasets in one request: "
            "keys are declared per dataset."
        )
    return dataset_ids.pop()


def ensure_key(
        db: Session,
        dataset_id: int,
        key: str,
        value_type: MetadataValueType | str | None = None,
        username: str | None = None,
) -> DatasetMetadataKeys:
    """Fetch a key's descriptor, creating a default one if it has none.

    Creating on demand is what keeps the type system out of the way: a curator
    types ``site`` into the editor and gets a working categorical key, rather
    than a form asking them to declare a schema first. ``value_type`` is only
    honoured when the descriptor is created — changing an existing key's type
    goes through :func:`update_key`, which has to re-validate the stored values.
    """
    descriptor = get_key(db, dataset_id, key)
    if descriptor is not None:
        return descriptor
    descriptor = DatasetMetadataKeys(
        dataset_id=dataset_id,
        key=normalize_key(key),
        value_type=MetadataValueType(value_type or DEFAULT_TYPE).value,
        options=[],
        created_by=username,
    )
    db.add(descriptor)
    db.flush()
    return descriptor


def update_key(
        db: Session,
        dataset_id: int,
        key: str,
        value_type: MetadataValueType | str | None = None,
        unit: str | None = None,
        options: list[str] | None = None,
        description: str | None = None,
) -> DatasetMetadataKeys:
    """Change a key's declaration, re-validating every value already stored.

    Retyping is the dangerous operation: declaring ``depth`` numeric when one
    image has ``depth = "shallow"`` would either lose that row or leave a value
    the type says cannot exist. So every stored value is coerced first and the
    whole change is refused, naming the offenders, if any of them fail. The
    caller can then fix those images and retry.

    The same applies to narrowing a categorical key's option list: values outside
    the new vocabulary are reported rather than silently orphaned.

    :raises InvalidMetadataError: if a stored value does not fit the new type.
    """
    descriptor = get_key(db, dataset_id, key)
    if descriptor is None:
        raise InvalidMetadataError(f"Dataset {dataset_id} has no key '{key}'.")

    new_type = MetadataValueType(value_type) if value_type else MetadataValueType(descriptor.value_type)
    new_options = options if options is not None else list(descriptor.options or [])

    rows = (
        db.query(ImageMetadata)
        .join(Images, Images.id == ImageMetadata.image_id)
        .filter(Images.dataset_id == dataset_id, ImageMetadata.key == descriptor.key)
        .all()
    )

    coerced: list[tuple[ImageMetadata, str, float | None]] = []
    rejected: list[str] = []
    for row in rows:
        try:
            canonical, numeric = coerce(row.value, new_type, new_options)
        except InvalidMetadataError:
            rejected.append(row.value)
            continue
        coerced.append((row, canonical, numeric))

    if rejected:
        sample = sorted(set(rejected))[:5]
        raise InvalidMetadataError(
            f"{len(rejected)} image(s) have a '{descriptor.key}' value that is not "
            f"a valid {new_type.value}: {', '.join(repr(v) for v in sample)}"
            + (" ..." if len(set(rejected)) > len(sample) else "")
            + ". Fix or clear those values first."
        )

    for row, canonical, numeric in coerced:
        row.value = canonical
        row.value_num = numeric

    descriptor.value_type = new_type.value
    descriptor.options = new_options
    if unit is not None:
        descriptor.unit = unit or None
    if description is not None:
        descriptor.description = description or None
    db.commit()
    return descriptor


def rename_key(
        db: Session,
        dataset_id: int,
        key: str,
        new_key: str,
        merge: bool = False,
) -> dict:
    """Rename a key across the dataset, optionally merging it into an existing one.

    This is the repair for the split every free-form key system eventually grows:
    someone typed ``Site`` where everyone else typed ``site``, and the dataset now
    reports two subgroups that are one. Renaming without ``merge`` refuses to
    collide with an existing key rather than quietly folding two vocabularies
    together; with ``merge`` the source wins on images that carry both, since the
    caller named it as the one to keep.

    :raises InvalidMetadataError: if the target exists and ``merge`` is not set.
    """
    descriptor = get_key(db, dataset_id, key)
    if descriptor is None:
        raise InvalidMetadataError(f"Dataset {dataset_id} has no key '{key}'.")
    target_name = normalize_key(new_key)
    if target_name == descriptor.key:
        return {"renamed": 0, "merged": 0}

    target = get_key(db, dataset_id, target_name)
    if target is not None and not merge:
        raise InvalidMetadataError(
            f"'{target_name}' already exists. Rename with merge to combine them."
        )

    rows = (
        db.query(ImageMetadata)
        .join(Images, Images.id == ImageMetadata.image_id)
        .filter(Images.dataset_id == dataset_id, ImageMetadata.key == descriptor.key)
        .all()
    )
    # Rows already under the target key on the same image: the source replaces
    # them, so they are deleted rather than left to violate the unique constraint.
    clashing = {
        row.image_id: row for row in
        db.query(ImageMetadata)
        .join(Images, Images.id == ImageMetadata.image_id)
        .filter(Images.dataset_id == dataset_id, ImageMetadata.key == target_name)
        .all()
    }

    # Delete the clashing rows and flush BEFORE renaming: SQLAlchemy's unit of
    # work emits UPDATEs before DELETEs, so renaming first would momentarily put
    # two rows on the same (image_id, key) and trip the unique constraint.
    merged = 0
    for row in rows:
        existing = clashing.get(row.image_id)
        if existing is not None:
            db.delete(existing)
            merged += 1
    db.flush()
    for row in rows:
        row.key = target_name

    if target is not None:
        # The surviving descriptor is the target's, so the merged key keeps the
        # type the caller was merging into. Values were already validated against
        # the source's type, so re-coerce them.
        db.flush()
        db.delete(descriptor)
        db.commit()
        update_key(db, dataset_id, target_name, value_type=target.value_type,
                   options=list(target.options or []))
    else:
        descriptor.key = target_name
        db.commit()
    return {"renamed": len(rows), "merged": merged}


def delete_key_from_dataset(db: Session, dataset_id: int, key: str) -> int:
    """Drop a key and every value of it in the dataset. Returns rows removed."""
    descriptor = get_key(db, dataset_id, key)
    normalized = normalize_key(key)
    rows = (
        db.query(ImageMetadata)
        .join(Images, Images.id == ImageMetadata.image_id)
        .filter(Images.dataset_id == dataset_id, ImageMetadata.key == normalized)
        .all()
    )
    for row in rows:
        db.delete(row)
    if descriptor is not None:
        db.delete(descriptor)
    db.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

def get_metadata(db: Session, image_id: int) -> dict[str, str]:
    """Every metadata pair of one image, as a plain ``{key: value}`` dict."""
    rows = db.query(ImageMetadata).filter_by(image_id=image_id).all()
    return {row.key: row.value for row in sorted(rows, key=lambda r: r.key.lower())}


def get_metadata_for_images(db: Session, image_ids: list[int]) -> dict[int, dict[str, str]]:
    """``{image_id: {key: value}}`` for a batch of images, in one query.

    Images with no metadata are present with an empty dict, so a caller can read
    the result without worrying whether an id is missing or merely untagged.
    """
    result: dict[int, dict[str, str]] = {image_id: {} for image_id in image_ids}
    if not image_ids:
        return result
    rows = (
        db.query(ImageMetadata)
        .filter(ImageMetadata.image_id.in_(image_ids))
        .order_by(ImageMetadata.key)
        .all()
    )
    for row in rows:
        result[row.image_id][row.key] = row.value
    return result


def get_metadata_for_dataset(db: Session, dataset_id: int) -> dict[int, dict[str, str]]:
    """``{image_id: {key: value}}`` for every image of a dataset."""
    rows = (
        db.query(ImageMetadata, Images.id)
        .join(Images, Images.id == ImageMetadata.image_id)
        .filter(Images.dataset_id == dataset_id)
        .order_by(ImageMetadata.key)
        .all()
    )
    result: dict[int, dict[str, str]] = {}
    for row, image_id in rows:
        result.setdefault(image_id, {})[row.key] = row.value
    return result


def set_metadata_for_images(
        db: Session,
        image_ids: list[int],
        entries: dict[str, str],
        username: str | None = None,
        replace: bool = False,
        remove_keys: list[str] | None = None,
        key_types: dict[str, str] | None = None,
) -> dict:
    """Apply one set of metadata edits to one or many images.

    Args:
        db: SQLAlchemy session.
        image_ids: Images to edit. Every id must exist.
        entries: ``{key: value}`` to write. **An empty value removes the key** —
            the editor's "clear this field" and "delete this row" are the same
            gesture, and an empty-string value is not a subgroup anyone can filter on.
        username: Account making the change, recorded on the row.
        replace: Treat ``entries`` as the image's complete metadata, deleting any
            key not mentioned. Off by default so the bulk action ("give all of
            these ``site=reef_a``") does not wipe the per-image keys it says
            nothing about.
        remove_keys: Keys to delete outright, whatever ``entries`` says. Only
            useful with ``replace=False``, where it is the bulk "untag" action.
        key_types: Optional ``{key: value_type}`` for keys that do not exist yet,
            used by the CSV importer to declare a column's type as it creates it.
            Ignored for a key that already has a descriptor — retyping goes
            through :func:`update_key`, which re-validates what is stored.

    Returns:
        ``{"updated_images": [...], "written": n, "removed": n}``.

    :raises ImageNotFoundError: if any id does not exist.
    :raises InvalidMetadataError: if a key or value fails its type's validation.
    """
    normalized = _normalize_entries(entries)
    to_remove = {normalize_key(k) for k in (remove_keys or [])}
    # An empty value is a removal, not a stored blank.
    to_remove |= {key for key, value in normalized.items() if not value}
    to_write = {key: value for key, value in normalized.items() if value}

    if not image_ids:
        return {"updated_images": [], "written": 0, "removed": 0}

    found = {row.id for row in db.query(Images.id).filter(Images.id.in_(image_ids))}
    missing = [image_id for image_id in image_ids if image_id not in found]
    if missing:
        raise ImageNotFoundError(
            f"Image(s) {', '.join(str(i) for i in missing)} were not found."
        )

    # Coerce once, before touching anything: a value that fails its key's type
    # must not leave half the batch tagged. Also creates descriptors for keys
    # seen for the first time.
    coerced: dict[str, tuple[str, float | None]] = {}
    if to_write:
        dataset_id = _dataset_id_for_images(db, image_ids)
        for key, value in to_write.items():
            descriptor = ensure_key(db, dataset_id, key,
                                    value_type=(key_types or {}).get(key),
                                    username=username)
            try:
                coerced[key] = coerce(value, descriptor.value_type,
                                      list(descriptor.options or []))
            except InvalidMetadataError as exc:
                raise InvalidMetadataError(f"{key}: {exc}") from exc

    existing_rows = (
        db.query(ImageMetadata).filter(ImageMetadata.image_id.in_(image_ids)).all()
    )
    by_image: dict[int, dict[str, ImageMetadata]] = {}
    for row in existing_rows:
        by_image.setdefault(row.image_id, {})[row.key] = row

    written = removed = 0
    for image_id in image_ids:
        current = by_image.get(image_id, {})

        drop = set(to_remove)
        if replace:
            drop |= set(current) - set(to_write)
        for key in drop:
            row = current.get(key)
            if row is not None:
                db.delete(row)
                removed += 1

        for key in to_write:
            value, numeric = coerced[key]
            row = current.get(key)
            if row is None:
                db.add(ImageMetadata(image_id=image_id, key=key, value=value,
                                     value_num=numeric, created_by=username))
                written += 1
            elif row.value != value:
                row.value = value
                row.value_num = numeric
                row.created_by = username
                written += 1

    db.commit()
    logger.info("Metadata on %s image(s): %s written, %s removed.",
                len(image_ids), written, removed)
    return {"updated_images": list(image_ids), "written": written, "removed": removed}


def delete_key(db: Session, image_id: int, key: str) -> bool:
    """Remove one key from one image. Returns whether a row was actually deleted."""
    row = (
        db.query(ImageMetadata)
        .filter_by(image_id=image_id, key=normalize_key(key))
        .first()
    )
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True


def get_dataset_facets(db: Session, dataset_id: int) -> list[dict]:
    """The dataset's metadata vocabulary: every key, its values, and their counts.

    This is what the grouping UI is built from — the filter chips, the key and
    value suggestions in the editor, and the "12 images have no ``site``"
    warning that tells a curator the grouping is incomplete.

    Each entry also carries its key's declaration — type, unit, whether it can be
    grouped on — so a client can pick the right control (a range for a depth, a
    date picker for a collection date, chips for a site) without a second request.
    Ordered keys additionally get ``range``, the min/max needed to draw a slider.

    A key that has a descriptor but no values yet still appears, with an empty
    value list: a curator who declares ``treatment`` up front should see it in the
    editor before the first image is tagged.

    Returns:
        One entry per key, most-used first::

            [{"key": "site", "value_type": "categorical", "unit": None,
              "groupable": True, "options": [], "image_count": 40, "range": None,
              "values": [{"value": "reef_a", "count": 22}, ...]}, ...]
    """
    rows = (
        db.query(
            ImageMetadata.key,
            ImageMetadata.value,
            func.count(ImageMetadata.id),
            func.min(ImageMetadata.value_num),
            func.max(ImageMetadata.value_num),
        )
        .join(Images, Images.id == ImageMetadata.image_id)
        .filter(Images.dataset_id == dataset_id)
        .group_by(ImageMetadata.key, ImageMetadata.value)
        .all()
    )

    facets: dict[str, list[dict]] = {}
    numeric_bounds: dict[str, tuple[float, float]] = {}
    for key, value, count, low, high in rows:
        facets.setdefault(key, []).append({"value": value, "count": count})
        if low is None or high is None:
            continue
        current = numeric_bounds.get(key)
        numeric_bounds[key] = (
            min(low, current[0]) if current else low,
            max(high, current[1]) if current else high,
        )

    descriptors = {row.key: row for row in list_keys(db, dataset_id)}

    result = []
    for key in set(facets) | set(descriptors):
        values = facets.get(key, [])
        descriptor = descriptors.get(key)
        value_type = MetadataValueType(descriptor.value_type) if descriptor else DEFAULT_TYPE
        bounds = numeric_bounds.get(key)
        result.append({
            "key": key,
            "value_type": value_type.value,
            "unit": descriptor.unit if descriptor else None,
            "description": descriptor.description if descriptor else None,
            "options": list(descriptor.options or []) if descriptor else [],
            "groupable": value_type in GROUPABLE_TYPES,
            "ordered": value_type in ORDERED_TYPES,
            "image_count": sum(entry["count"] for entry in values),
            "range": ({"min": bounds[0], "max": bounds[1]}
                      if bounds and value_type in ORDERED_TYPES else None),
            # Most-used value first, then alphabetically, so the chip row is
            # stable between requests and leads with the dominant subgroup. An
            # ordered key is sorted by its value instead: a depth list that reads
            # 2, 5, 12, 30 is a scale, one sorted by popularity is noise.
            "values": (
                sorted(values, key=lambda e: _sort_value(e["value"]))
                if value_type in ORDERED_TYPES
                else sorted(values, key=lambda e: (-e["count"], e["value"].lower()))
            ),
        })
    result.sort(key=lambda facet: (-facet["image_count"], facet["key"].lower()))
    return result


def _sort_value(value: str) -> tuple[float, str]:
    """Sort an ordered key's values numerically, falling back to text."""
    try:
        return float(value), ""
    except ValueError:
        return float("inf"), value.lower()


def filter_image_ids(
        db: Session,
        dataset_id: int,
        filters: dict[str, list[str] | dict],
) -> list[int]:
    """Ids of the dataset's images matching a metadata filter.

    Values within one key are OR-ed (``site in (reef_a, reef_b)``); different
    keys are AND-ed (``site=reef_a AND treatment=control``) — the reading that
    makes a row of filter chips behave the way people expect.

    A key's condition is either:

    * a **list of values** — match any of them. An empty list means "has this key
      at all, whatever the value", which is how you find everything that has been
      given a ``treatment`` regardless of which.
    * a **dict**, for the typed controls: ``{"min": x, "max": y}`` compares
      against ``value_num`` (so it works for numbers, dates as epoch seconds and
      booleans), and ``{"contains": "..."}`` is the substring match a free-text
      key gets instead of chips. Either bound may be omitted for an open range.
    """
    query = db.query(Images.id).filter(Images.dataset_id == dataset_id)
    for raw_key, condition in (filters or {}).items():
        key = normalize_key(raw_key)
        subquery = db.query(ImageMetadata.image_id).filter(ImageMetadata.key == key)

        if isinstance(condition, dict):
            low, high = condition.get("min"), condition.get("max")
            if low is not None:
                subquery = subquery.filter(ImageMetadata.value_num >= float(low))
            if high is not None:
                subquery = subquery.filter(ImageMetadata.value_num <= float(high))
            contains = normalize_value(condition.get("contains", ""))
            if contains:
                subquery = subquery.filter(ImageMetadata.value.ilike(f"%{contains}%"))
        else:
            cleaned = [normalize_value(v) for v in (condition or []) if normalize_value(v)]
            if cleaned:
                subquery = subquery.filter(ImageMetadata.value.in_(cleaned))

        query = query.filter(Images.id.in_(subquery))
    return [row.id for row in query.all()]

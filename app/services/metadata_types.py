"""The type system for image-metadata keys.

A key's *type* is what lets the UI offer the right filter (a range for a depth, a
date picker for a collection date, chips for a site) and what lets the tool say
which keys are meaningful to group a quantification by. Without it every key is a
bag of strings and every filter is a chip list.

Design points:

* **The type lives on the key, not the value.** One descriptor row per
  ``(dataset, key)`` — see ``app.database.dataset_metadata_keys``. A per-value
  type would let one image's ``depth`` be a number and another's a sentence,
  which is exactly the state the type is meant to rule out.
* **Values are still stored as text**, with an indexed ``value_num`` sidecar
  filled in for the ordered types (numbers, dates as epoch seconds, booleans as
  0/1). One storage path, one range-query path, and the human-readable form is
  never lost to a lossy round trip.
* **Coercion is validation, not repair.** :func:`coerce` rejects a value the type
  cannot represent rather than guessing; the caller turns that into a 422 naming
  the offending value. The one exception is canonical *spelling* of booleans and
  dates, where there is a single unambiguous answer.

Numbers and dates deliberately are not groupable: every value tends to be
distinct, so grouping a quantification by them would produce one bar per image.
They need binning first, which is not built yet — see :data:`GROUPABLE_TYPES`.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum

from app.exceptions import InvalidMetadataError


class MetadataValueType(StrEnum):
    """What kind of thing a metadata key holds."""

    #: Free text. Filterable by substring, never offered as a grouping — a note
    #: is not a subgroup.
    TEXT = "text"
    #: A small, repeating vocabulary (site, treatment, transect). The default for
    #: a key created by typing one in, because that is overwhelmingly what ad-hoc
    #: metadata is, and it is the only type the grouping UI can use as-is.
    CATEGORICAL = "categorical"
    #: A measured quantity, optionally carrying a unit.
    NUMBER = "number"
    #: A calendar date or timestamp, stored ISO-8601.
    DATE = "date"
    #: A flag. Stored canonically as "true"/"false" however it was written.
    BOOLEAN = "boolean"


#: Types whose values can be a grouping axis. Numbers and dates are excluded
#: because their values are near-unique; they need binning, which does not exist
#: yet. Text is excluded because a free-text note is not a subgroup.
GROUPABLE_TYPES: frozenset[MetadataValueType] = frozenset({
    MetadataValueType.CATEGORICAL,
    MetadataValueType.BOOLEAN,
})

#: Types that fill ``image_metadata.value_num`` and support range filtering.
ORDERED_TYPES: frozenset[MetadataValueType] = frozenset({
    MetadataValueType.NUMBER,
    MetadataValueType.DATE,
    MetadataValueType.BOOLEAN,
})

#: The type a key gets when it is created by simply typing it into the editor.
#: Categorical rather than text so the chips and grouping keep working exactly as
#: they did before types existed; a curator narrows or widens it afterwards.
DEFAULT_TYPE = MetadataValueType.CATEGORICAL

_TRUE_SPELLINGS = {"true", "yes", "y", "1", "t"}
_FALSE_SPELLINGS = {"false", "no", "n", "0", "f"}

#: Rows needed before "every value is distinct" is taken as evidence of free text
#: rather than of a small dataset. Three images with three sites is a vocabulary;
#: twenty rows with twenty different values is not.
_MIN_ROWS_TO_TRUST_UNIQUENESS = 4

#: Distinct values a column may have and still be offered as a vocabulary. Sites
#: and treatments live well under this; anything past it is a poor grouping axis
#: even when values do repeat.
_MAX_CATEGORICAL_DISTINCT = 30

#: Accepted date spellings, tried in order. ISO first because that is what the
#: canonical form is and what a round trip through the CSV export produces.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d/%m/%Y",
)


def _parse_datetime(value: str) -> datetime | None:
    """Parse a date or timestamp, or None if it is neither."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        parsed = None
    if parsed is None:
        for fmt in _DATE_FORMATS:
            try:
                parsed = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    # A bare date is midnight UTC; a timestamp keeps its offset. Without this the
    # epoch sidecar would shift with the server's local zone.
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)


def _canonical_datetime(parsed: datetime, original: str) -> str:
    """Render a parsed datetime back to ISO-8601, keeping date-only as a date.

    A ``collection_date`` typed as ``2024-05-01`` should read back as
    ``2024-05-01``, not ``2024-05-01T00:00:00+00:00`` — the extra precision is
    invented and it makes the CSV export unpleasant to read.
    """
    looks_date_only = "T" not in original and ":" not in original
    if looks_date_only:
        return parsed.date().isoformat()
    return parsed.isoformat()


def coerce(
        value: str,
        value_type: MetadataValueType | str,
        options: list[str] | None = None,
) -> tuple[str, float | None]:
    """Validate a value against its key's type.

    Args:
        value: The trimmed value as the user wrote it.
        value_type: The key's declared type.
        options: For a categorical key, the allowed vocabulary. An empty or absent
            list means the key is open and collects whatever is used — which is
            how every key behaves before anyone locks it down.

    Returns:
        ``(canonical_text, value_num)``. ``value_num`` is None for the unordered
        types and is what range filters and sorting run on for the rest.

    :raises InvalidMetadataError: if the value cannot be represented as this type.
    """
    value_type = MetadataValueType(value_type)

    if value_type is MetadataValueType.NUMBER:
        try:
            # Kept as written rather than reformatted: "12.0" and "12" are the
            # same number to every filter (they share a value_num) and the user's
            # spelling is the one that belongs in an export.
            return value, float(value)
        except ValueError:
            raise InvalidMetadataError(f"'{value}' is not a number.")

    if value_type is MetadataValueType.DATE:
        parsed = _parse_datetime(value)
        if parsed is None:
            raise InvalidMetadataError(
                f"'{value}' is not a date. Use YYYY-MM-DD."
            )
        return _canonical_datetime(parsed, value), parsed.timestamp()

    if value_type is MetadataValueType.BOOLEAN:
        lowered = value.lower()
        if lowered in _TRUE_SPELLINGS:
            return "true", 1.0
        if lowered in _FALSE_SPELLINGS:
            return "false", 0.0
        raise InvalidMetadataError(f"'{value}' is not true or false.")

    if value_type is MetadataValueType.CATEGORICAL and options:
        if value not in options:
            raise InvalidMetadataError(
                f"'{value}' is not one of the allowed values for this key "
                f"({', '.join(options)})."
            )

    return value, None


def infer_type(values: list[str]) -> MetadataValueType:
    """Guess the type of a column from the values it contains.

    Used by the CSV importer to *propose* a type for a new key — never to change
    an existing one. The order matters: booleans before numbers, because "1"/"0"
    parse as both and a column of only ones and zeros is far more often a flag.

    Every non-empty value has to fit, so a single unparseable row keeps the whole
    column categorical. A column that is mostly numbers with one "n/a" in it is
    not a numeric column; pretending otherwise loses that row on import.
    """
    present = [value for value in values if value]
    if not present:
        return DEFAULT_TYPE

    lowered = [value.lower() for value in present]
    if all(value in _TRUE_SPELLINGS or value in _FALSE_SPELLINGS for value in lowered):
        # All-numeric "1"/"0" columns land here too, which is the intent.
        return MetadataValueType.BOOLEAN

    def _parses_as_number(value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False

    if all(_parses_as_number(value) for value in present):
        return MetadataValueType.NUMBER

    if all(_parse_datetime(value) is not None for value in present):
        return MetadataValueType.DATE

    # The categorical/text divide is repetition. A vocabulary repeats — that is
    # what makes it a set of groups to compare. A column where every single row
    # differs is an identifier or a free-text note, and calling it categorical
    # would offer a grouping with one image per group.
    distinct = len(set(present))
    if distinct == len(present) and len(present) >= _MIN_ROWS_TO_TRUST_UNIQUENESS:
        return MetadataValueType.TEXT

    # Below that, fall back to an absolute cap: a handful of distinct values is a
    # vocabulary however few rows there are, which keeps a three-image dataset
    # from being told its sites are free text.
    if distinct <= max(_MAX_CATEGORICAL_DISTINCT, len(present) // 2):
        return MetadataValueType.CATEGORICAL
    return MetadataValueType.TEXT


def describe(value_type: MetadataValueType | str) -> dict:
    """Serializable facts about a type, for a client rendering its controls."""
    value_type = MetadataValueType(value_type)
    return {
        "value_type": value_type.value,
        "groupable": value_type in GROUPABLE_TYPES,
        "ordered": value_type in ORDERED_TYPES,
    }

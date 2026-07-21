"""Per-dataset quantification profiles (Step 5).

A profile is a named selection of which metrics to compute / report for a dataset, with
per-metric params and optional label scoping. The whole entry list is stored in a single
JSON column (``entries``) rather than a child table: it mirrors how ``contours`` already
store their coordinate lists as JSON, keeps CRUD to a single row, and the entry shape is
validated by the pydantic ``QuantificationProfile`` / ``ProfileEntry`` schemas in the
toolbox rather than by the relational schema.

The model is imported by ``app.database`` consumers (and the test suite) so
``create_all`` builds the table.
"""
from iquana_toolbox.schemas.database.quantification_profile import QuantificationProfile
from sqlalchemy import JSON, Boolean, Column, ForeignKey, Integer, String

from app.database import database


class QuantificationProfiles(database):
    """A named quantification profile scoped to one dataset.

    ``entries`` holds the ordered list of ``{metric_key, params, label_ids}`` dicts (the
    JSON serialization of a list of ``ProfileEntry``). ``is_default`` marks the single
    profile used when the frontend asks for a dataset without specifying one; the CRUD
    layer keeps at most one default per dataset.
    """
    __tablename__ = "quantification_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(128), nullable=False)
    is_default = Column(Boolean, nullable=False, default=False)
    # Ordered list of ProfileEntry dicts: {metric_key, params, label_ids}.
    entries = Column(JSON, nullable=False, default=list)

    @classmethod
    def from_schema(cls, schema: QuantificationProfile) -> "QuantificationProfiles":
        return cls(
            id=schema.id,
            dataset_id=schema.dataset_id,
            name=schema.name,
            is_default=schema.is_default,
            entries=schema.entries_as_json(),
        )

    def to_schema(self) -> QuantificationProfile:
        return QuantificationProfile.from_db(self)

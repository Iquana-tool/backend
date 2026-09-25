"""Object provenance, the AI suggestion log and per-dataset AI tool switches.

* ``ai_suggestions``: every AI output that became an object, with the outline as the
  model returned it and what a person later did to it (edited, deleted, approved).
* ``contours.origin`` / ``suggestion_id`` / ``geometry_edited_at``: which tool created
  an object, which suggestion it carries, and when its outline was last edited by hand.
* ``datasets.disabled_ai_tools``: AI tools switched off for a dataset.

Like 0002, everything is created only where it is missing, so a database that booted
this code before being put under Alembic leaves with the same schema as one that did
not. The definitions are a snapshot, not imports from the models.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-25 16:00:00.000000

"""
from logging import getLogger
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = getLogger("migrations.provenance_and_ai_tools")

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

schema = sa.MetaData()

sa.Table(
    "ai_suggestions",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("run_id", sa.String(length=64), nullable=False),
    sa.Column("source", sa.String(length=32), nullable=False),
    sa.Column("model_key", sa.String(length=255), nullable=True),
    sa.Column("username", sa.String(), nullable=True),
    sa.Column("dataset_id", sa.Integer(), nullable=True),
    sa.Column("image_id", sa.Integer(), nullable=True),
    sa.Column("mask_id", sa.Integer(), nullable=True),
    sa.Column("contour_id", sa.Integer(), nullable=True),
    sa.Column("label_id", sa.Integer(), nullable=True),
    sa.Column("confidence", sa.Float(), nullable=True),
    sa.Column("x", sa.JSON(), nullable=False),
    sa.Column("y", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(), nullable=False),
    sa.Column("edited_at", sa.DateTime(), nullable=True),
    sa.Column("deleted_at", sa.DateTime(), nullable=True),
    sa.Column("reviewed_at", sa.DateTime(), nullable=True),
    sa.Column("superseded_at", sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint("id", name="ai_suggestions_pkey"),
    sa.Index("ix_ai_suggestions_contour_id", "contour_id"),
    sa.Index("ix_ai_suggestions_dataset_id", "dataset_id"),
    sa.Index("ix_ai_suggestions_image_id", "image_id"),
    sa.Index("ix_ai_suggestions_mask_created", "mask_id", "created_at"),
    sa.Index("ix_ai_suggestions_mask_id", "mask_id"),
    sa.Index("ix_ai_suggestions_run_id", "run_id"),
    sa.Index("ix_ai_suggestions_username", "username"),
)


def _added_columns() -> tuple[tuple[str, sa.Column], ...]:
    """Columns added to existing tables, as (table, column). All nullable.

    Built fresh on every call: ``add_column`` binds a Column to its table, and one
    Column cannot be bound twice.
    """
    return (
        ("contours", sa.Column("origin", sa.String(length=32), nullable=True)),
        ("contours", sa.Column("suggestion_id", sa.Integer(), nullable=True)),
        ("contours", sa.Column("geometry_edited_at", sa.DateTime(), nullable=True)),
        ("datasets", sa.Column("disabled_ai_tools", sa.String(length=255), nullable=True)),
    )


#: Indexes on added columns, as (name, table, columns).
_ADDED_INDEXES = (
    ("ix_contours_suggestion_id", "contours", ["suggestion_id"]),
)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    missing = [table for table in schema.sorted_tables if table.name not in existing]
    for table in missing:
        logger.info("Creating table %s.", table.name)
    schema.create_all(bind, tables=missing)

    for table, column in _added_columns():
        if column.name in {c["name"] for c in inspector.get_columns(table)}:
            continue
        logger.info("Adding column %s.%s.", table, column.name)
        op.add_column(table, column)

    for name, table, columns in _ADDED_INDEXES:
        if name in {index["name"] for index in sa.inspect(bind).get_indexes(table)}:
            continue
        logger.info("Creating index %s.", name)
        op.create_index(name, table, columns)


def downgrade() -> None:
    for name, table, _columns in reversed(_ADDED_INDEXES):
        op.drop_index(name, table_name=table)
    for table, column in reversed(_added_columns()):
        op.drop_column(table, column.name)
    schema.drop_all(op.get_bind())

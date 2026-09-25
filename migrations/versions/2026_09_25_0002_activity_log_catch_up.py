"""Activity log catch-up: the schema of #130 and #131, which the baseline missed.

The activity log (#130: ``user_events``, ``activity_log_settings``) and the dataset
activity summary (#131: who finished a mask and when, when an approval was given)
were merged while Alembic was being introduced, and still used the old ways --
``create_all`` for the tables, ``_ADDED_COLUMNS`` for the columns. The baseline was
snapshotted from the tree just before them, so they arrive here.

Everything is created only where it is missing. A database that booted #130/#131
before it was put under Alembic already has these tables and columns; one that did
not gets them now. Either way it leaves this revision with the same schema.

Like the baseline, the definitions are a snapshot rather than imports from the
models, so this revision keeps describing the schema as of 2026-09-25.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-25 14:42:52.188095

"""
from logging import getLogger
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = getLogger("migrations.activity_log_catch_up")

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

schema = sa.MetaData()

sa.Table(
    "activity_log_settings",
    schema,
    sa.Column("id", sa.Integer(), nullable=False),
    sa.Column("capture_enabled", sa.Boolean(), nullable=False),
    sa.Column("components", sa.String(length=255), nullable=False),
    sa.Column("idle_threshold_ms", sa.Integer(), nullable=True),
    sa.Column("updated_at", sa.DateTime(), nullable=False),
    sa.Column("updated_by", sa.String(), nullable=True),
    sa.PrimaryKeyConstraint("id", name="activity_log_settings_pkey"),
)
sa.Table(
    "user_events",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("event_id", sa.String(length=64), nullable=False),
    sa.Column("ts", sa.DateTime(), nullable=False),
    sa.Column("received_at", sa.DateTime(), nullable=False),
    sa.Column("username", sa.String(), nullable=True),
    sa.Column("session_id", sa.String(length=64), nullable=True),
    sa.Column("component", sa.String(length=32), nullable=False),
    sa.Column("event_type", sa.String(length=128), nullable=False),
    sa.Column("source", sa.String(length=16), nullable=False),
    sa.Column("dataset_id", sa.Integer(), nullable=True),
    sa.Column("image_id", sa.Integer(), nullable=True),
    sa.Column("duration_ms", sa.Integer(), nullable=True),
    sa.Column("payload", sa.Text(), nullable=True),
    sa.Column("client", sa.String(length=255), nullable=True),
    sa.PrimaryKeyConstraint("id", name="user_events_pkey"),
    sa.Index("ix_user_events_component_ts", "component", "ts"),
    sa.Index("ix_user_events_dataset_id", "dataset_id"),
    sa.Index("ix_user_events_event_id", "event_id", unique=True),
    sa.Index("ix_user_events_event_type", "event_type"),
    sa.Index("ix_user_events_image_id", "image_id"),
    sa.Index("ix_user_events_session_ts", "session_id", "ts"),
    sa.Index("ix_user_events_ts", "ts"),
    sa.Index("ix_user_events_username", "username"),
)

def _added_columns() -> tuple[tuple[str, sa.Column], ...]:
    """Columns #131 added to existing tables, as (table, column).

    All nullable: NULL on rows recorded before the columns existed. Built fresh on
    every call because ``add_column`` binds a Column to its table, and one Column
    cannot be bound twice -- which a second upgrade in the same process would try.
    """
    return (
        ("masks", sa.Column("fully_annotated_by", sa.String(), nullable=True)),
        ("masks", sa.Column("fully_annotated_at", sa.DateTime(), nullable=True)),
        ("reviewer_contour_association", sa.Column("reviewed_at", sa.DateTime(), nullable=True)),
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


def downgrade() -> None:
    for table, column in reversed(_added_columns()):
        op.drop_column(table, column.name)
    schema.drop_all(op.get_bind())

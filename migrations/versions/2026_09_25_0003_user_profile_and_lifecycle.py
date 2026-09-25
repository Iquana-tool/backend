"""User profile and lifecycle columns; usernames that can be changed.

Two changes to accounts:

* **New columns on ``users``.** A profile (``display_name``, ``email``,
  ``preferences``) and a lifecycle (``created_at``, ``last_login_at``,
  ``tokens_valid_after``, ``must_change_password``). All nullable or defaulted, so
  existing rows need no backfill. ``created_at`` stays NULL on existing accounts:
  when they were made is not known, and a guessed date would look like a real one.

* **ON UPDATE CASCADE on every foreign key into ``users.username``.** Changing a
  username (a rename, or anonymising a departed account) becomes a single UPDATE
  on ``users`` instead of being impossible. Each key keeps its ON DELETE rule,
  except ``datasets.created_by``, which goes from CASCADE to RESTRICT: deleting an
  account must never delete the datasets it created. Deactivation is the way to
  remove someone.

Columns that hold a username without a foreign key (``masks.fully_annotated_by``,
``user_events.username``, the ``updated_by``/``created_by`` provenance columns) do
not follow a rename. That is deliberate for study data and provenance, and has to
be handled by whatever performs the rename.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-25 14:47:29.525429

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

#: Every foreign key into users.username, as (table, column, ON DELETE before,
#: ON DELETE after). Names follow ``<table>_<column>_fkey``.
USER_FOREIGN_KEYS = (
    ("annotation_actions", "username", "CASCADE", "CASCADE"),
    ("annotation_queues", "username", "CASCADE", "CASCADE"),
    ("annotation_rejections", "created_by", "SET NULL", "SET NULL"),
    ("annotation_rejections", "resolved_by", "SET NULL", "SET NULL"),
    ("contours", "author_username", "SET NULL", "SET NULL"),
    ("dataset_invites", "created_by", "CASCADE", "CASCADE"),
    ("dataset_members", "granted_by", "SET NULL", "SET NULL"),
    ("dataset_members", "username", "CASCADE", "CASCADE"),
    ("dataset_model_routing_configs", "updated_by", "SET NULL", "SET NULL"),
    ("datasets", "created_by", "CASCADE", "RESTRICT"),
    ("inference_jobs", "created_by", "SET NULL", "SET NULL"),
    ("reviewer_contour_association", "reviewer_id", "CASCADE", "CASCADE"),
    ("user_model_favorites", "username", "CASCADE", "CASCADE"),
)


def _user_columns() -> tuple[sa.Column, ...]:
    """The new ``users`` columns, built fresh per call (a Column binds to one table)."""
    return (
        sa.Column("display_name", sa.String(length=100), nullable=True),
        sa.Column("email", sa.String(length=254), nullable=True),
        sa.Column("preferences", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
        sa.Column("tokens_valid_after", sa.DateTime(), nullable=True),
        sa.Column("must_change_password", sa.Boolean(), server_default=sa.false(), nullable=False),
    )


def _replace_user_foreign_keys(upgrading: bool) -> None:
    for table, column, delete_before, delete_after in USER_FOREIGN_KEYS:
        name = f"{table}_{column}_fkey"
        op.drop_constraint(name, table, type_="foreignkey")
        op.create_foreign_key(
            name, table, "users", [column], ["username"],
            ondelete=delete_after if upgrading else delete_before,
            onupdate="CASCADE" if upgrading else None,
        )


def upgrade() -> None:
    for column in _user_columns():
        op.add_column("users", column)
    op.create_unique_constraint("users_email_key", "users", ["email"])
    _replace_user_foreign_keys(upgrading=True)


def downgrade() -> None:
    # Loses whatever the new columns held: profiles, preferences, login times.
    _replace_user_foreign_keys(upgrading=False)
    op.drop_constraint("users_email_key", "users", type_="unique")
    for column in reversed(_user_columns()):
        op.drop_column("users", column.name)

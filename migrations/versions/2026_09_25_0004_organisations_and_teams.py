"""Organisations and teams, and datasets shared with a team.

New tables: ``organizations``, ``organization_members``, ``teams`` (nesting through
``parent_team_id``, which is how departments are modelled), ``team_members`` and
``dataset_team_grants``; and ``datasets.organization_id``.

**Backfill.** Every database gets one organisation, marked as the default -- the one
new accounts join. Every existing account becomes a member of it (platform admins
as organisation admins) and every existing dataset belongs to it. A single-lab
instance therefore behaves exactly as before, and nothing is left outside an
organisation. The organisation is named after the instance when the instance has a
name, and can be renamed.

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-25

"""
import os
from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _default_organization_name(bind) -> str:
    """The instance's own name if it has one (admin page first, then environment)."""
    name = bind.execute(sa.text(
        "SELECT value FROM instance_settings WHERE key = 'instance_name'")).scalar()
    name = (name or os.getenv("INSTANCE_NAME") or "").strip()
    return name[:100] or "Default organisation"


def _backfill_default_organization(bind) -> None:
    # Naive UTC, like the users timestamps: see app.database.users.utc_now.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    organization_id = bind.execute(sa.text(
        "INSERT INTO organizations (name, is_default, created_at) "
        "VALUES (:name, true, :now) RETURNING id"),
        {"name": _default_organization_name(bind), "now": now}).scalar()
    bind.execute(sa.text(
        "INSERT INTO organization_members (organization_id, username, role, joined_at) "
        "SELECT :organization_id, username, "
        "CASE WHEN global_role = 'admin' THEN 'admin' ELSE 'member' END, :now FROM users"),
        {"organization_id": organization_id, "now": now})
    bind.execute(sa.text("UPDATE datasets SET organization_id = :organization_id"),
                 {"organization_id": organization_id})


def upgrade() -> None:
    op.create_table('organizations',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(length=100), nullable=False),
    sa.Column('is_default', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id', name='organizations_pkey'),
    sa.UniqueConstraint('name', name='organizations_name_key')
    )
    op.create_index('uq_organizations_default', 'organizations', ['is_default'], unique=True, postgresql_where=sa.text('is_default'))
    op.create_table('organization_members',
    sa.Column('organization_id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(), nullable=False),
    sa.Column('role', sa.String(length=20), nullable=False),
    sa.Column('joined_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name='organization_members_organization_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['username'], ['users.username'], name='organization_members_username_fkey', onupdate='CASCADE', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('organization_id', 'username', name='organization_members_pkey')
    )
    op.create_index('ix_organization_members_username', 'organization_members', ['username'], unique=False)
    op.create_table('teams',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('organization_id', sa.Integer(), nullable=False),
    sa.Column('parent_team_id', sa.Integer(), nullable=True),
    sa.Column('name', sa.String(length=100), nullable=False),
    sa.Column('description', sa.String(length=255), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name='teams_organization_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['parent_team_id'], ['teams.id'], name='teams_parent_team_id_fkey', ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name='teams_pkey'),
    sa.UniqueConstraint('organization_id', 'name', name='uq_teams_organization_name')
    )
    op.create_index('ix_teams_organization_id', 'teams', ['organization_id'], unique=False)
    op.create_index('ix_teams_parent_team_id', 'teams', ['parent_team_id'], unique=False)
    op.create_table('dataset_team_grants',
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('team_id', sa.Integer(), nullable=False),
    sa.Column('role', sa.String(length=20), nullable=False),
    sa.Column('extra_permissions', sa.JSON(), nullable=False),
    sa.Column('denied_permissions', sa.JSON(), nullable=False),
    sa.Column('granted_by', sa.String(), nullable=True),
    sa.Column('granted_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], name='dataset_team_grants_dataset_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['granted_by'], ['users.username'], name='dataset_team_grants_granted_by_fkey', onupdate='CASCADE', ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['team_id'], ['teams.id'], name='dataset_team_grants_team_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('dataset_id', 'team_id', name='dataset_team_grants_pkey')
    )
    op.create_index('ix_dataset_team_grants_team_id', 'dataset_team_grants', ['team_id'], unique=False)
    op.create_table('team_members',
    sa.Column('team_id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(), nullable=False),
    sa.Column('role', sa.String(length=20), nullable=False),
    sa.Column('joined_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['team_id'], ['teams.id'], name='team_members_team_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['username'], ['users.username'], name='team_members_username_fkey', onupdate='CASCADE', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('team_id', 'username', name='team_members_pkey')
    )
    op.create_index('ix_team_members_username', 'team_members', ['username'], unique=False)
    op.add_column('datasets', sa.Column('organization_id', sa.Integer(), nullable=True))
    op.create_index('ix_datasets_organization_id', 'datasets', ['organization_id'], unique=False)
    op.create_foreign_key('datasets_organization_id_fkey', 'datasets', 'organizations', ['organization_id'], ['id'], ondelete='RESTRICT')

    _backfill_default_organization(op.get_bind())


def downgrade() -> None:
    op.drop_constraint('datasets_organization_id_fkey', 'datasets', type_='foreignkey')
    op.drop_index('ix_datasets_organization_id', table_name='datasets')
    op.drop_column('datasets', 'organization_id')
    op.drop_index('ix_team_members_username', table_name='team_members')
    op.drop_table('team_members')
    op.drop_index('ix_dataset_team_grants_team_id', table_name='dataset_team_grants')
    op.drop_table('dataset_team_grants')
    op.drop_index('ix_teams_parent_team_id', table_name='teams')
    op.drop_index('ix_teams_organization_id', table_name='teams')
    op.drop_table('teams')
    op.drop_index('ix_organization_members_username', table_name='organization_members')
    op.drop_table('organization_members')
    op.drop_index('uq_organizations_default', table_name='organizations', postgresql_where=sa.text('is_default'))
    op.drop_table('organizations')

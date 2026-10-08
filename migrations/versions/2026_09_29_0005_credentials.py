"""Credentials: API keys held by one account or one organisation.

New table ``credentials`` -- per kind (so far ``llm``), one key per account and one
per organisation, the key encrypted -- and ``organizations.allow_personal_keys``,
which lets an organisation keep its work on the provider it chose.

The instance's own secrets in ``instance_settings`` are encrypted too, but not
here: encryption needs the key, which belongs to the running backend rather than to
the schema, so ``settings.encrypt_stored_secrets`` rewrites them when the backend
starts. This revision stays a pure schema change.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-29 15:14:05.162018

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0005'
down_revision: Union[str, Sequence[str], None] = '0004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('credentials',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('username', sa.String(), nullable=True),
    sa.Column('organization_id', sa.Integer(), nullable=True),
    sa.Column('model', sa.String(length=200), nullable=True),
    sa.Column('api_base', sa.String(length=255), nullable=True),
    sa.Column('secret', sa.Text(), nullable=False),
    sa.Column('hint', sa.String(length=16), nullable=False),
    sa.Column('created_by', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.Column('last_used_at', sa.DateTime(), nullable=True),
    sa.CheckConstraint('(username IS NULL) <> (organization_id IS NULL)', name='ck_credentials_one_owner'),
    sa.ForeignKeyConstraint(['created_by'], ['users.username'], name='credentials_created_by_fkey', onupdate='CASCADE', ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], name='credentials_organization_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['username'], ['users.username'], name='credentials_username_fkey', onupdate='CASCADE', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='credentials_pkey'),
    sa.UniqueConstraint('organization_id', 'kind', name='uq_credentials_organization_kind'),
    sa.UniqueConstraint('username', 'kind', name='uq_credentials_username_kind')
    )
    op.add_column('organizations', sa.Column('allow_personal_keys', sa.Boolean(), server_default=sa.text('true'), nullable=False))


def downgrade() -> None:
    op.drop_column('organizations', 'allow_personal_keys')
    op.drop_table('credentials')

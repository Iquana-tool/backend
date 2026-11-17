"""Registration links: ``account_invites``.

There is no open self-registration; an admin sends a link instead, and following it
lets someone create their own account. Only the token's SHA-256 is stored.

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-30 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0012'
down_revision: Union[str, Sequence[str], None] = '0011'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'account_invites',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('token_hash', sa.String(length=64), nullable=False),
        sa.Column('created_by', sa.String(), nullable=False),
        sa.Column('note', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('max_uses', sa.Integer(), nullable=True),
        sa.Column('uses', sa.Integer(), nullable=False),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.username'], ondelete='CASCADE',
                                onupdate='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_account_invites_token_hash', 'account_invites', ['token_hash'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_account_invites_token_hash', table_name='account_invites')
    op.drop_table('account_invites')

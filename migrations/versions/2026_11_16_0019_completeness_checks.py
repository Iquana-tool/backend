"""Completeness checks: ``completeness_checks``.

A reviewer's per-label answer to "is everything on this image annotated?", asked at
the end of each image in a review session. Rows are append-only; an undo marks a row
withdrawn.

Revision ID: 0019
Revises: 0018
Create Date: 2026-11-16 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0019'
down_revision: Union[str, Sequence[str], None] = '0018'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'completeness_checks',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('mask_id', sa.Integer(), nullable=False),
        sa.Column('item', sa.String(length=16), nullable=False),
        sa.Column('label_id', sa.Integer(), nullable=True),
        sa.Column('answer', sa.String(length=16), nullable=False),
        sa.Column('note', sa.String(length=1000), nullable=True),
        sa.Column('created_by', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('rejection_id', sa.Integer(), nullable=True),
        sa.Column('withdrawn_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['mask_id'], ['masks.id'], ondelete='CASCADE',
                                name='completeness_checks_mask_id_fkey'),
        sa.ForeignKeyConstraint(['label_id'], ['labels.id'], ondelete='CASCADE',
                                name='completeness_checks_label_id_fkey'),
        sa.ForeignKeyConstraint(['created_by'], ['users.username'], ondelete='SET NULL',
                                onupdate='CASCADE', name='completeness_checks_created_by_fkey'),
        sa.ForeignKeyConstraint(['rejection_id'], ['annotation_rejections.id'], ondelete='SET NULL',
                                name='completeness_checks_rejection_id_fkey'),
        sa.PrimaryKeyConstraint('id', name='completeness_checks_pkey'),
    )
    op.create_index('ix_completeness_checks_mask_id', 'completeness_checks', ['mask_id'])
    op.create_index('ix_completeness_checks_current', 'completeness_checks',
                    ['mask_id', 'withdrawn_at'])


def downgrade() -> None:
    op.drop_index('ix_completeness_checks_current', table_name='completeness_checks')
    op.drop_index('ix_completeness_checks_mask_id', table_name='completeness_checks')
    op.drop_table('completeness_checks')

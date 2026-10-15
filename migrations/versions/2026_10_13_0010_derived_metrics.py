"""Derived metrics: measurements a dataset defines as formulas over the direct ones.

New table ``derived_metrics`` -- per dataset, a key, a name, the labels it is reported
on and a JSON formula (the choroidal vascularity index, polyp density, ...). Only the
definition is stored; values are computed when a quantification is read. Rows go away
with their dataset.

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-13 18:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0010'
down_revision: Union[str, Sequence[str], None] = '0009'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('derived_metrics',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('key', sa.String(length=48), nullable=False),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.Column('description', sa.String(length=1024), nullable=False),
    sa.Column('label_ids', sa.JSON(), nullable=True),
    sa.Column('expression', sa.JSON(), nullable=False),
    sa.Column('created_by', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], name='derived_metrics_dataset_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['created_by'], ['users.username'], name='derived_metrics_created_by_fkey', onupdate='CASCADE', ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name='derived_metrics_pkey'),
    sa.UniqueConstraint('dataset_id', 'key', name='uq_derived_metrics_dataset_key')
    )
    op.create_index('ix_derived_metrics_dataset_id', 'derived_metrics', ['dataset_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_derived_metrics_dataset_id', table_name='derived_metrics')
    op.drop_table('derived_metrics')

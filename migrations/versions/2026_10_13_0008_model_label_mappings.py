"""Model label mappings: what a model's classes mean in a dataset.

New table ``model_label_mappings`` -- per dataset, model and model class, either a
dataset label (``action = 'map'``) or ``'discard'``. Models with a fixed label space
(``ModelInfo.classes``) name the class of every prediction; the gateway reads it
through this table. Rows go away with their dataset or their label.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-13 10:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0008'
down_revision: Union[str, Sequence[str], None] = '0007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('model_label_mappings',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('model_registry_key', sa.String(), nullable=False),
    sa.Column('model_class', sa.String(), nullable=False),
    sa.Column('action', sa.String(length=16), nullable=False),
    sa.Column('label_id', sa.Integer(), nullable=True),
    sa.Column('updated_by', sa.String(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], name='model_label_mappings_dataset_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['label_id'], ['labels.id'], name='model_label_mappings_label_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['updated_by'], ['users.username'], name='model_label_mappings_updated_by_fkey', onupdate='CASCADE', ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name='model_label_mappings_pkey'),
    sa.UniqueConstraint('dataset_id', 'model_registry_key', 'model_class', name='uq_model_label_mappings_class')
    )
    op.create_index('ix_model_label_mappings_dataset_id', 'model_label_mappings', ['dataset_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_model_label_mappings_dataset_id', table_name='model_label_mappings')
    op.drop_table('model_label_mappings')

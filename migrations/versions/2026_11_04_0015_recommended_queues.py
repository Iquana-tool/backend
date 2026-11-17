"""Recommended queues: ``datasets.recommended_queues``.

The queue settings a curator recommends for the dataset, one set per queue (annotation,
review), each with a one-line reason. Annotators and reviewers can start from them in the
builder. Kept as JSON so the settings follow the queue builders without further migrations.

Revision ID: 0015
Revises: 0014
Create Date: 2026-11-04 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0015'
down_revision: Union[str, Sequence[str], None] = '0014'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('datasets', sa.Column('recommended_queues', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('datasets') as batch:
        batch.drop_column('recommended_queues')

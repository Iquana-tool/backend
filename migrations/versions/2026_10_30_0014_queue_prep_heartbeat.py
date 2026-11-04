"""Queue preparation heartbeat: ``queue_prep_jobs.heartbeat_at``.

A preparation job is a chain of Celery tasks. When the worker dies the chain is gone, but
the job stays "running", and every later prepare request for the same settings reused it,
so preparation for that ordering was blocked for good. Each step now records when it
finished; ``start_prep`` gives up a job that has not beaten for a while and starts anew.

Revision ID: 0014
Revises: 0013
Create Date: 2026-10-30 10:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0014'
down_revision: Union[str, Sequence[str], None] = '0013'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('queue_prep_jobs', sa.Column('heartbeat_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('queue_prep_jobs') as batch:
        batch.drop_column('heartbeat_at')

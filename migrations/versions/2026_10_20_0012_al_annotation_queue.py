"""Active-learning annotation queue: image scores, prep jobs, queue model and reasons.

* ``image_scores`` -- how unsure a model is about each image.
* ``queue_prep_jobs`` -- background jobs filling image vectors and scores.
* ``annotation_queues.model_key`` / ``details`` -- the model a queue was ranked with, and
  why each image sits where it does. NULL on existing queues.

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-20 10:00:00.000000

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
        'image_scores',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('image_id', sa.Integer(), nullable=False),
        sa.Column('model_key', sa.String(length=255), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('num_detections', sa.Integer(), nullable=False),
        sa.Column('confidences', sa.JSON(), nullable=True),
        sa.Column('scored_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['image_id'], ['images.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('image_id', 'model_key', name='uq_image_scores_image_model'),
    )
    op.create_index('ix_image_scores_image_id', 'image_scores', ['image_id'])

    op.create_table(
        'queue_prep_jobs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=False),
        sa.Column('created_by', sa.String(), nullable=True),
        sa.Column('strategy', sa.String(length=32), nullable=False),
        sa.Column('model_key', sa.String(length=255), nullable=True),
        sa.Column('image_ids', sa.JSON(), nullable=False),
        sa.Column('next_index', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('failed_image_ids', sa.JSON(), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('celery_task_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['created_by'], ['users.username'], ondelete='SET NULL',
                                onupdate='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_queue_prep_jobs_dataset_id', 'queue_prep_jobs', ['dataset_id'])
    op.create_index('ix_queue_prep_jobs_status', 'queue_prep_jobs', ['status'])

    op.add_column('annotation_queues', sa.Column('model_key', sa.String(length=255), nullable=True))
    op.add_column('annotation_queues', sa.Column('details', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('annotation_queues', 'details')
    op.drop_column('annotation_queues', 'model_key')
    op.drop_index('ix_queue_prep_jobs_status', table_name='queue_prep_jobs')
    op.drop_index('ix_queue_prep_jobs_dataset_id', table_name='queue_prep_jobs')
    op.drop_table('queue_prep_jobs')
    op.drop_index('ix_image_scores_image_id', table_name='image_scores')
    op.drop_table('image_scores')

"""Annotation queue settings: any embedding model, and the builder's full settings.

* ``image_vectors`` -- whole-image vectors of any length, from any embedding model.
* ``queue_prep_jobs.needs`` replaces ``strategy`` / ``model_key``: the models a job
  prepares data for, so one job can serve several sorting methods.
* ``annotation_queues.options`` replaces ``model_key``, and ``show_builder`` is new: the
  settings a queue was built with, and whether the builder opens next time.

Existing rows are converted: a queue keeps its method and trained model, and Diversity or
Density queues get DINOv3, the only embedding model before this revision. Unfinished prep
jobs are cancelled because they filled the old store; the builder starts a new one when
needed.

Each step checks the current schema first, so databases that already have the new layout
(from a pre-release build of revision 0012) upgrade without errors.

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-23 10:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0013'
down_revision: Union[str, Sequence[str], None] = '0012'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Before this revision, vector-based methods always used DINOv3.
_VECTOR_METHODS = ('diversity', 'density')
_OLD_EMBEDDER = 'dinov3'


def _columns(table: str) -> set[str]:
    return {c['name'] for c in sa.inspect(op.get_bind()).get_columns(table)}


def _needs(strategy: str | None, model_key: str | None) -> dict:
    needs = {'embedding_model': _OLD_EMBEDDER} if strategy in _VECTOR_METHODS else {}
    if model_key:
        needs['trained_model'] = model_key
    return needs


def upgrade() -> None:
    bind = op.get_bind()

    if not sa.inspect(bind).has_table('image_vectors'):
        op.create_table(
            'image_vectors',
            sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
            sa.Column('image_id', sa.Integer(), nullable=False),
            sa.Column('model_key', sa.String(length=255), nullable=False),
            sa.Column('vector', sa.JSON(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(['image_id'], ['images.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('image_id', 'model_key', name='uq_image_vectors_image_model'),
        )
        op.create_index('ix_image_vectors_image_id', 'image_vectors', ['image_id'])

    if 'needs' not in _columns('queue_prep_jobs'):
        op.add_column('queue_prep_jobs', sa.Column('needs', sa.JSON(), nullable=True))
        jobs = sa.table('queue_prep_jobs', sa.column('id'), sa.column('strategy'),
                        sa.column('model_key'), sa.column('status'),
                        sa.column('needs', sa.JSON()))
        for job_id, strategy, model_key in bind.execute(
                sa.select(jobs.c.id, jobs.c.strategy, jobs.c.model_key)).all():
            bind.execute(jobs.update().where(jobs.c.id == job_id)
                         .values(needs=_needs(strategy, model_key)))
        bind.execute(jobs.update().where(jobs.c.status.in_(('pending', 'running', 'cancelling')))
                     .values(status='cancelled'))
        with op.batch_alter_table('queue_prep_jobs') as batch:
            batch.alter_column('needs', existing_type=sa.JSON(), nullable=False)
            batch.drop_column('strategy')
            batch.drop_column('model_key')

    queue_columns = _columns('annotation_queues')
    if 'options' not in queue_columns:
        op.add_column('annotation_queues', sa.Column('options', sa.JSON(), nullable=True))
        queues = sa.table('annotation_queues', sa.column('id'), sa.column('strategy'),
                          sa.column('model_key'), sa.column('options', sa.JSON()))
        for queue_id, strategy, model_key in bind.execute(
                sa.select(queues.c.id, queues.c.strategy, queues.c.model_key)).all():
            bind.execute(queues.update().where(queues.c.id == queue_id)
                         .values(options={'method': strategy, **_needs(strategy, model_key)}))
    if 'show_builder' not in queue_columns:
        op.add_column('annotation_queues', sa.Column('show_builder', sa.Boolean(), nullable=False,
                                                     server_default=sa.true()))
    if 'model_key' in queue_columns:
        with op.batch_alter_table('annotation_queues') as batch:
            batch.drop_column('model_key')


def downgrade() -> None:
    bind = op.get_bind()

    op.add_column('annotation_queues', sa.Column('model_key', sa.String(length=255), nullable=True))
    queues = sa.table('annotation_queues', sa.column('id'), sa.column('model_key'),
                      sa.column('options', sa.JSON()))
    for queue_id, options in bind.execute(sa.select(queues.c.id, queues.c.options)).all():
        if options and options.get('trained_model'):
            bind.execute(queues.update().where(queues.c.id == queue_id)
                         .values(model_key=options['trained_model']))
    with op.batch_alter_table('annotation_queues') as batch:
        batch.drop_column('show_builder')
        batch.drop_column('options')

    # Jobs only cache preparation progress, so they are dropped rather than converted back.
    bind.execute(sa.text('DELETE FROM queue_prep_jobs'))
    op.add_column('queue_prep_jobs', sa.Column('strategy', sa.String(length=32), nullable=False,
                                               server_default='uncertainty'))
    op.add_column('queue_prep_jobs', sa.Column('model_key', sa.String(length=255), nullable=True))
    with op.batch_alter_table('queue_prep_jobs') as batch:
        batch.alter_column('strategy', existing_type=sa.String(length=32), server_default=None)
        batch.drop_column('needs')

    op.drop_index('ix_image_vectors_image_id', table_name='image_vectors')
    op.drop_table('image_vectors')

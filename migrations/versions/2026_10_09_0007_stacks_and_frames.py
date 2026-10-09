"""Stacks and frames: datasets can hold OCT volumes, z-stacks and videos.

New table ``stacks`` -- one dataset item made of ordered 2D frames, with an optional
overview image (the IR-SLO of an OCT volume). A frame is an ``images`` row with
``kind = 'frame'``, ``stack_id`` and ``frame_index`` set, so masks, contours and
metadata attach to frames exactly as to images. Existing rows become
``kind = 'image'`` through the server default; nothing is migrated.

``image_metadata`` rows can now belong to a stack instead of an image
(``stack_id``); frames inherit those.

The ``scans`` table is dropped. Nothing ever wrote to it: masks hang off
``image_id``, so a scan could not carry annotations, which is what stacks fix.

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-09 10:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0007'
down_revision: Union[str, Sequence[str], None] = '0006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('stacks',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('kind', sa.String(length=32), nullable=False),
    sa.Column('axis', sa.String(length=1), nullable=False),
    sa.Column('frame_count', sa.Integer(), nullable=False),
    sa.Column('frame_spacing', sa.Float(), nullable=True),
    sa.Column('frame_spacing_unit', sa.String(length=10), nullable=True),
    sa.Column('overview_file_path', sa.String(), nullable=True),
    sa.Column('overview_width', sa.Integer(), nullable=True),
    sa.Column('overview_height', sa.Integer(), nullable=True),
    sa.Column('overview_scale_x', sa.Float(), nullable=True),
    sa.Column('overview_scale_y', sa.Float(), nullable=True),
    sa.Column('overview_unit', sa.String(length=10), nullable=True),
    sa.Column('thumbnail_file_path', sa.String(), nullable=False),
    sa.Column('source_format', sa.String(length=32), nullable=False),
    sa.Column('source_file_name', sa.String(), nullable=False),
    sa.Column('created_by', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['created_by'], ['users.username'], name='stacks_created_by_fkey', onupdate='CASCADE', ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], name='stacks_dataset_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='stacks_pkey')
    )
    op.create_index('ix_stacks_dataset_id', 'stacks', ['dataset_id'], unique=False)

    op.add_column('images', sa.Column('kind', sa.String(length=16), server_default='image', nullable=False))
    op.add_column('images', sa.Column('stack_id', sa.Integer(), nullable=True))
    op.add_column('images', sa.Column('frame_index', sa.Integer(), nullable=True))
    op.add_column('images', sa.Column('frame_position', sa.Float(), nullable=True))
    op.add_column('images', sa.Column('overview_geometry', sa.JSON(), nullable=True))
    op.create_index('ix_images_kind', 'images', ['kind'], unique=False)
    op.create_index('ix_images_stack_id', 'images', ['stack_id'], unique=False)
    op.create_foreign_key('images_stack_id_fkey', 'images', 'stacks', ['stack_id'], ['id'], ondelete='CASCADE')
    op.create_unique_constraint('uq_images_stack_frame_index', 'images', ['stack_id', 'frame_index'])
    op.create_check_constraint(
        'ck_images_frame_columns', 'images',
        "(kind = 'frame') = (stack_id IS NOT NULL AND frame_index IS NOT NULL)")

    op.alter_column('image_metadata', 'image_id', existing_type=sa.Integer(), nullable=True)
    op.add_column('image_metadata', sa.Column('stack_id', sa.Integer(), nullable=True))
    op.create_index('ix_image_metadata_stack_id', 'image_metadata', ['stack_id'], unique=False)
    op.create_foreign_key('image_metadata_stack_id_fkey', 'image_metadata', 'stacks', ['stack_id'], ['id'], ondelete='CASCADE')
    op.create_unique_constraint('uq_image_metadata_stack_key', 'image_metadata', ['stack_id', 'key'])
    op.create_check_constraint(
        'ck_image_metadata_one_owner', 'image_metadata',
        '(image_id IS NULL) <> (stack_id IS NULL)')

    op.drop_table('scans')


def downgrade() -> None:
    op.create_table('scans',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('folder_path', sa.String(), nullable=False),
    sa.Column('type', sa.String(), nullable=True),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('number_of_slices', sa.Integer(), nullable=False),
    sa.Column('meta_data', sa.JSON(), nullable=True),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], name='scans_dataset_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='scans_pkey')
    )

    # Stack metadata and frames have nowhere to go in the old schema.
    op.execute('DELETE FROM image_metadata WHERE stack_id IS NOT NULL')
    op.drop_constraint('ck_image_metadata_one_owner', 'image_metadata', type_='check')
    op.drop_constraint('uq_image_metadata_stack_key', 'image_metadata', type_='unique')
    op.drop_constraint('image_metadata_stack_id_fkey', 'image_metadata', type_='foreignkey')
    op.drop_index('ix_image_metadata_stack_id', table_name='image_metadata')
    op.drop_column('image_metadata', 'stack_id')
    op.alter_column('image_metadata', 'image_id', existing_type=sa.Integer(), nullable=False)

    op.execute("DELETE FROM images WHERE kind = 'frame'")
    op.drop_constraint('ck_images_frame_columns', 'images', type_='check')
    op.drop_constraint('uq_images_stack_frame_index', 'images', type_='unique')
    op.drop_constraint('images_stack_id_fkey', 'images', type_='foreignkey')
    op.drop_index('ix_images_stack_id', table_name='images')
    op.drop_index('ix_images_kind', table_name='images')
    op.drop_column('images', 'overview_geometry')
    op.drop_column('images', 'frame_position')
    op.drop_column('images', 'frame_index')
    op.drop_column('images', 'stack_id')
    op.drop_column('images', 'kind')

    op.drop_index('ix_stacks_dataset_id', table_name='stacks')
    op.drop_table('stacks')

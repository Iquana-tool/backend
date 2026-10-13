"""Scene graph: given attributes and named relationships between objects.

New tables:

* ``object_attribute_definitions`` -- per dataset, what an attribute is (type, unit,
  vocabulary), like ``dataset_metadata_keys`` for images;
* ``object_attribute_label_scope`` -- the labels an attribute applies to (none = all);
* ``object_attributes`` -- one value per (object, attribute), with provenance;
* ``relationship_types`` -- the relationship names a dataset uses;
* ``object_relationships`` -- directed edges between two objects.

Everything cascades with its dataset, its object or its definition.

Also: ``contours.model_class`` (the class a model predicted, in its own terms) and
``model_label_mappings.attribute_id`` / ``attribute_value``, so a model class can map to
a label *and* an attribute value ("Acropora" -> Coral fragment, species = Acropora).

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-13 16:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0009'
down_revision: Union[str, Sequence[str], None] = '0008'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _user_fk(column: str, table: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint([column], ['users.username'], name=f'{table}_{column}_fkey',
                                   onupdate='CASCADE', ondelete='SET NULL')


def upgrade() -> None:
    op.create_table('object_attribute_definitions',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=64), nullable=False),
    sa.Column('value_type', sa.String(length=16), nullable=False),
    sa.Column('unit', sa.String(length=16), nullable=True),
    sa.Column('options', sa.JSON(), nullable=False),
    sa.Column('description', sa.String(length=256), nullable=True),
    sa.Column('created_by', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], name='object_attribute_definitions_dataset_id_fkey', ondelete='CASCADE'),
    _user_fk('created_by', 'object_attribute_definitions'),
    sa.PrimaryKeyConstraint('id', name='object_attribute_definitions_pkey'),
    sa.UniqueConstraint('dataset_id', 'name', name='uq_object_attribute_definitions_name')
    )
    op.create_index('ix_object_attribute_definitions_dataset_id', 'object_attribute_definitions', ['dataset_id'], unique=False)

    op.create_table('object_attribute_label_scope',
    sa.Column('attribute_id', sa.Integer(), nullable=False),
    sa.Column('label_id', sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(['attribute_id'], ['object_attribute_definitions.id'], name='object_attribute_label_scope_attribute_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['label_id'], ['labels.id'], name='object_attribute_label_scope_label_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('attribute_id', 'label_id', name='object_attribute_label_scope_pkey')
    )
    op.create_index('ix_object_attribute_label_scope_label_id', 'object_attribute_label_scope', ['label_id'], unique=False)

    op.create_table('object_attributes',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('contour_id', sa.Integer(), nullable=False),
    sa.Column('attribute_id', sa.Integer(), nullable=False),
    sa.Column('value', sa.String(length=512), nullable=True),
    sa.Column('value_num', sa.Float(), nullable=True),
    sa.Column('source', sa.String(length=16), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('model_key', sa.String(length=255), nullable=True),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('author_username', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['contour_id'], ['contours.id'], name='object_attributes_contour_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['attribute_id'], ['object_attribute_definitions.id'], name='object_attributes_attribute_id_fkey', ondelete='CASCADE'),
    _user_fk('author_username', 'object_attributes'),
    sa.PrimaryKeyConstraint('id', name='object_attributes_pkey'),
    sa.UniqueConstraint('contour_id', 'attribute_id', name='uq_object_attributes_contour_attribute')
    )
    op.create_index('ix_object_attributes_contour_id', 'object_attributes', ['contour_id'], unique=False)
    op.create_index('ix_object_attributes_attribute_id', 'object_attributes', ['attribute_id'], unique=False)
    op.create_index('ix_object_attributes_value_num', 'object_attributes', ['value_num'], unique=False)

    op.create_table('relationship_types',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=64), nullable=False),
    sa.Column('symmetric', sa.Boolean(), nullable=False),
    sa.Column('description', sa.String(length=256), nullable=True),
    sa.Column('created_by', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], name='relationship_types_dataset_id_fkey', ondelete='CASCADE'),
    _user_fk('created_by', 'relationship_types'),
    sa.PrimaryKeyConstraint('id', name='relationship_types_pkey'),
    sa.UniqueConstraint('dataset_id', 'name', name='uq_relationship_types_name')
    )
    op.create_index('ix_relationship_types_dataset_id', 'relationship_types', ['dataset_id'], unique=False)

    op.create_table('object_relationships',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('dataset_id', sa.Integer(), nullable=False),
    sa.Column('type_id', sa.Integer(), nullable=False),
    sa.Column('subject_id', sa.Integer(), nullable=False),
    sa.Column('object_id', sa.Integer(), nullable=False),
    sa.Column('source', sa.String(length=16), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('model_key', sa.String(length=255), nullable=True),
    sa.Column('confidence', sa.Float(), nullable=True),
    sa.Column('author_username', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.CheckConstraint('subject_id <> object_id', name='ck_object_relationships_no_self_loop'),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], name='object_relationships_dataset_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['type_id'], ['relationship_types.id'], name='object_relationships_type_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['subject_id'], ['contours.id'], name='object_relationships_subject_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['object_id'], ['contours.id'], name='object_relationships_object_id_fkey', ondelete='CASCADE'),
    _user_fk('author_username', 'object_relationships'),
    sa.PrimaryKeyConstraint('id', name='object_relationships_pkey'),
    sa.UniqueConstraint('type_id', 'subject_id', 'object_id', name='uq_object_relationships_edge')
    )
    op.create_index('ix_object_relationships_dataset_id', 'object_relationships', ['dataset_id'], unique=False)
    op.create_index('ix_object_relationships_type_id', 'object_relationships', ['type_id'], unique=False)
    op.create_index('ix_object_relationships_subject_id', 'object_relationships', ['subject_id'], unique=False)
    op.create_index('ix_object_relationships_object_id', 'object_relationships', ['object_id'], unique=False)

    op.add_column('contours', sa.Column('model_class', sa.String(length=255), nullable=True))
    op.add_column('model_label_mappings', sa.Column('attribute_id', sa.Integer(), nullable=True))
    op.add_column('model_label_mappings', sa.Column('attribute_value', sa.String(length=512), nullable=True))
    op.create_foreign_key('model_label_mappings_attribute_id_fkey', 'model_label_mappings',
                          'object_attribute_definitions', ['attribute_id'], ['id'], ondelete='SET NULL')


def downgrade() -> None:
    op.drop_constraint('model_label_mappings_attribute_id_fkey', 'model_label_mappings', type_='foreignkey')
    op.drop_column('model_label_mappings', 'attribute_value')
    op.drop_column('model_label_mappings', 'attribute_id')
    op.drop_column('contours', 'model_class')
    op.drop_table('object_relationships')
    op.drop_table('relationship_types')
    op.drop_table('object_attributes')
    op.drop_table('object_attribute_label_scope')
    op.drop_table('object_attribute_definitions')

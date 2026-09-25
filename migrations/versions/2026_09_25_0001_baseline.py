"""Baseline: the schema as it stood when Alembic was introduced.

Every later revision builds on this one, and it has two jobs:

* **An empty database** gets the whole schema, created from the frozen table
  definitions below.
* **A database from before Alembic** -- built by the old boot path, which ran
  ``create_all`` and patched columns in by hand -- is *adopted*: whatever it lacks
  is added, and the known differences between such databases and a fresh one are
  normalised, so that every database leaves this revision with the same schema.
  Adoption adds and never removes, apart from one redundant constraint. It refuses,
  rather than guesses, when a database needs a data migration first (a NOT NULL
  column it cannot fill).

The adoption covers what the old boot path could leave behind:

* tables it had not got round to creating yet;
* nullable columns added to a model after its table existed (``create_all`` never
  ALTERs), e.g. ``image_metadata.value_num``;
* indexes on such columns, which the hand-written ALTERs never created;
* ``datasets_name_key``, the unique constraint ``create_all`` built for
  ``datasets.name`` next to the ``uq_datasets_name`` index the boot path added. The
  index is what the models declare now, so the constraint goes.

Foreign keys are not added to existing tables: legacy rows may not satisfy them.
``app.database.migrate`` logs any difference that survives adoption.

The table definitions are a snapshot, deliberately not imported from the models:
the models move on, and this revision has to keep describing the schema of
2026-09-25. Constraint and index names are spelled out in full for the same reason.

Revision ID: 0001
Revises:
Create Date: 2026-09-25 13:51:11.590569

"""
from logging import getLogger
from typing import Sequence, Union

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

# Not under "alembic": MLflow turns that logger down to WARNING when it is imported.
logger = getLogger("migrations.baseline")

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

schema = sa.MetaData()

sa.Table(
    "instance_settings",
    schema,
    sa.Column("key", sa.String(length=64), nullable=False),
    sa.Column("value", sa.Text(), nullable=True),
    sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    sa.Column("updated_by", sa.String(), nullable=True),
    sa.PrimaryKeyConstraint("key", name="instance_settings_pkey"),
)
sa.Table(
    "users",
    schema,
    sa.Column("username", sa.String(), nullable=False),
    sa.Column("hashed_password", sa.String(), nullable=False),
    sa.Column("global_role", sa.String(length=20), nullable=False),
    sa.Column("is_active", sa.Boolean(), nullable=False),
    sa.PrimaryKeyConstraint("username", name="users_pkey"),
)
sa.Table(
    "datasets",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("name", sa.String(length=50), nullable=False),
    sa.Column("description", sa.String(length=255), nullable=True),
    sa.Column("dataset_type", sa.String(length=20), nullable=False),
    sa.Column("folder_path", sa.String(length=255), nullable=False),
    sa.Column("created_by", sa.String(), nullable=False),
    sa.Column("require_independent_review", sa.Boolean(), nullable=False),
    sa.ForeignKeyConstraint(["created_by"], ["users.username"], name="datasets_created_by_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="datasets_pkey"),
    sa.Index("uq_datasets_name", "name", unique=True),
)
sa.Table(
    "user_model_favorites",
    schema,
    sa.Column("username", sa.String(), nullable=False),
    sa.Column("task", sa.String(), nullable=False),
    sa.Column("model_registry_key", sa.String(), nullable=False),
    sa.ForeignKeyConstraint(["username"], ["users.username"], name="user_model_favorites_username_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("username", "task", name="user_model_favorites_pkey"),
)
sa.Table(
    "annotation_queues",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("dataset_id", sa.Integer(), nullable=False),
    sa.Column("username", sa.String(), nullable=False),
    sa.Column("strategy", sa.String(length=32), nullable=False),
    sa.Column("image_order", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(), nullable=False),
    sa.Column("updated_at", sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name="annotation_queues_dataset_id_fkey", ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["username"], ["users.username"], name="annotation_queues_username_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="annotation_queues_pkey"),
    sa.UniqueConstraint("dataset_id", "username", name="uq_annotation_queue_dataset_user"),
    sa.Index("ix_annotation_queues_dataset_id", "dataset_id"),
)
sa.Table(
    "dataset_calibration_defaults",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("dataset_id", sa.Integer(), nullable=False),
    sa.Column("kind", sa.String(length=32), nullable=False),
    sa.Column("defaults", sa.JSON(), nullable=False),
    sa.Column("updated_by", sa.String(), nullable=True),
    sa.Column("updated_at", sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name="dataset_calibration_defaults_dataset_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="dataset_calibration_defaults_pkey"),
    sa.UniqueConstraint("dataset_id", "kind", name="uq_dataset_calibration_defaults"),
    sa.Index("ix_dataset_calibration_defaults_dataset_id", "dataset_id"),
)
sa.Table(
    "dataset_invites",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("token_hash", sa.String(length=64), nullable=False),
    sa.Column("dataset_id", sa.Integer(), nullable=False),
    sa.Column("role", sa.String(length=20), nullable=False),
    sa.Column("created_by", sa.String(), nullable=False),
    sa.Column("created_at", sa.DateTime(), nullable=False),
    sa.Column("expires_at", sa.DateTime(), nullable=True),
    sa.Column("max_uses", sa.Integer(), nullable=True),
    sa.Column("uses", sa.Integer(), nullable=False),
    sa.Column("revoked_at", sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(["created_by"], ["users.username"], name="dataset_invites_created_by_fkey", ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name="dataset_invites_dataset_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="dataset_invites_pkey"),
    sa.Index("ix_dataset_invites_token_hash", "token_hash", unique=True),
)
sa.Table(
    "dataset_members",
    schema,
    sa.Column("dataset_id", sa.Integer(), nullable=False),
    sa.Column("username", sa.String(), nullable=False),
    sa.Column("role", sa.String(length=20), nullable=False),
    sa.Column("extra_permissions", sa.JSON(), nullable=False),
    sa.Column("denied_permissions", sa.JSON(), nullable=False),
    sa.Column("granted_by", sa.String(), nullable=True),
    sa.Column("granted_at", sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name="dataset_members_dataset_id_fkey", ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["granted_by"], ["users.username"], name="dataset_members_granted_by_fkey", ondelete="SET NULL"),
    sa.ForeignKeyConstraint(["username"], ["users.username"], name="dataset_members_username_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("dataset_id", "username", name="dataset_members_pkey"),
)
sa.Table(
    "dataset_metadata_keys",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("dataset_id", sa.Integer(), nullable=False),
    sa.Column("key", sa.String(length=64), nullable=False),
    sa.Column("value_type", sa.String(length=16), nullable=False),
    sa.Column("unit", sa.String(length=16), nullable=True),
    sa.Column("options", sa.JSON(), nullable=False),
    sa.Column("description", sa.String(length=256), nullable=True),
    sa.Column("created_by", sa.String(), nullable=True),
    sa.Column("created_at", sa.DateTime(), nullable=False),
    sa.Column("updated_at", sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name="dataset_metadata_keys_dataset_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="dataset_metadata_keys_pkey"),
    sa.UniqueConstraint("dataset_id", "key", name="uq_dataset_metadata_keys_dataset_key"),
    sa.Index("ix_dataset_metadata_keys_dataset_id", "dataset_id"),
)
sa.Table(
    "dataset_model_routing_configs",
    schema,
    sa.Column("dataset_id", sa.Integer(), nullable=False),
    sa.Column("bindings", sa.JSON(), nullable=False),
    sa.Column("updated_by", sa.String(), nullable=True),
    sa.Column("created_at", sa.DateTime(), nullable=False),
    sa.Column("updated_at", sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name="dataset_model_routing_configs_dataset_id_fkey", ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["updated_by"], ["users.username"], name="dataset_model_routing_configs_updated_by_fkey", ondelete="SET NULL"),
    sa.PrimaryKeyConstraint("dataset_id", name="dataset_model_routing_configs_pkey"),
)
sa.Table(
    "images",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("dataset_id", sa.Integer(), nullable=False),
    sa.Column("file_name", sa.String(), nullable=False),
    sa.Column("file_path", sa.String(), nullable=False),
    sa.Column("thumbnail_file_path", sa.String(), nullable=False),
    sa.Column("description", sa.String(), nullable=True),
    sa.Column("width", sa.Integer(), nullable=False),
    sa.Column("height", sa.Integer(), nullable=False),
    sa.Column("color_mode", sa.String(), nullable=False),
    sa.Column("scale_x", sa.Float(), nullable=False),
    sa.Column("scale_y", sa.Float(), nullable=False),
    sa.Column("unit", sa.String(length=10), nullable=True),
    sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name="images_dataset_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="images_pkey"),
    sa.Index("ix_images_dataset_id", "dataset_id"),
)
sa.Table(
    "inference_jobs",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("dataset_id", sa.Integer(), nullable=False),
    sa.Column("created_by", sa.String(), nullable=True),
    sa.Column("name", sa.String(length=80), nullable=True),
    sa.Column("status", sa.String(length=16), nullable=False),
    sa.Column("write_mode", sa.String(length=16), nullable=False),
    sa.Column("plan_steps", sa.JSON(), nullable=False),
    sa.Column("options", sa.JSON(), nullable=False),
    sa.Column("image_ids", sa.JSON(), nullable=False),
    sa.Column("total_units", sa.Integer(), nullable=False),
    sa.Column("contours_created", sa.Integer(), nullable=False),
    sa.Column("contours_suppressed", sa.Integer(), nullable=False),
    sa.Column("contours_deleted", sa.Integer(), nullable=False),
    sa.Column("contours_unparented", sa.Integer(), nullable=False),
    sa.Column("celery_task_id", sa.String(), nullable=True),
    sa.Column("error", sa.Text(), nullable=True),
    sa.Column("created_at", sa.DateTime(), nullable=False),
    sa.Column("started_at", sa.DateTime(), nullable=True),
    sa.Column("finished_at", sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(["created_by"], ["users.username"], name="inference_jobs_created_by_fkey", ondelete="SET NULL"),
    sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name="inference_jobs_dataset_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="inference_jobs_pkey"),
    sa.Index("ix_inference_jobs_dataset_id", "dataset_id"),
    sa.Index("ix_inference_jobs_status", "status"),
)
sa.Table(
    "labels",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("dataset_id", sa.Integer(), nullable=True),
    sa.Column("parent_id", sa.Integer(), nullable=True),
    sa.Column("name", sa.String(), nullable=False),
    sa.Column("value", sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name="labels_dataset_id_fkey", ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["parent_id"], ["labels.id"], name="labels_parent_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="labels_pkey"),
)
sa.Table(
    "quantification_profiles",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("dataset_id", sa.Integer(), nullable=False),
    sa.Column("name", sa.String(length=128), nullable=False),
    sa.Column("is_default", sa.Boolean(), nullable=False),
    sa.Column("entries", sa.JSON(), nullable=False),
    sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name="quantification_profiles_dataset_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="quantification_profiles_pkey"),
)
sa.Table(
    "scans",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("dataset_id", sa.Integer(), nullable=False),
    sa.Column("name", sa.String(), nullable=False),
    sa.Column("folder_path", sa.String(), nullable=False),
    sa.Column("type", sa.String(), nullable=True),
    sa.Column("description", sa.String(), nullable=True),
    sa.Column("number_of_slices", sa.Integer(), nullable=False),
    sa.Column("meta_data", sa.JSON(), nullable=True),
    sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], name="scans_dataset_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="scans_pkey"),
)
sa.Table(
    "image_calibrations",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("image_id", sa.Integer(), nullable=False),
    sa.Column("kind", sa.String(length=32), nullable=False),
    sa.Column("params", sa.JSON(), nullable=False),
    sa.Column("source", sa.String(length=24), nullable=False),
    sa.Column("created_by", sa.String(), nullable=True),
    sa.Column("created_at", sa.DateTime(), nullable=False),
    sa.Column("updated_at", sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(["image_id"], ["images.id"], name="image_calibrations_image_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="image_calibrations_pkey"),
    sa.UniqueConstraint("image_id", "kind", name="uq_image_calibrations_image_kind"),
    sa.Index("ix_image_calibrations_image_id", "image_id"),
)
sa.Table(
    "image_metadata",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("image_id", sa.Integer(), nullable=False),
    sa.Column("key", sa.String(length=64), nullable=False),
    sa.Column("value", sa.String(length=512), nullable=False),
    sa.Column("value_num", sa.Float(), nullable=True),
    sa.Column("created_by", sa.String(), nullable=True),
    sa.Column("created_at", sa.DateTime(), nullable=False),
    sa.Column("updated_at", sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(["image_id"], ["images.id"], name="image_metadata_image_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="image_metadata_pkey"),
    sa.UniqueConstraint("image_id", "key", name="uq_image_metadata_image_key"),
    sa.Index("ix_image_metadata_image_id", "image_id"),
    sa.Index("ix_image_metadata_key_value", "key", "value"),
    sa.Index("ix_image_metadata_value_num", "value_num"),
)
sa.Table(
    "inference_job_items",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("job_id", sa.Integer(), nullable=False),
    sa.Column("level", sa.Integer(), nullable=False),
    sa.Column("step_index", sa.Integer(), nullable=False),
    sa.Column("image_id", sa.Integer(), nullable=False),
    sa.Column("status", sa.String(length=16), nullable=False),
    sa.Column("contours_created", sa.Integer(), nullable=False),
    sa.Column("contours_suppressed", sa.Integer(), nullable=False),
    sa.Column("contours_unparented", sa.Integer(), nullable=False),
    sa.Column("duration_ms", sa.Float(), nullable=True),
    sa.Column("error", sa.Text(), nullable=True),
    sa.Column("finished_at", sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(["image_id"], ["images.id"], name="inference_job_items_image_id_fkey", ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["job_id"], ["inference_jobs.id"], name="inference_job_items_job_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="inference_job_items_pkey"),
    sa.Index("ix_inference_job_items_image_id", "image_id"),
    sa.Index("ix_inference_job_items_job_id", "job_id"),
    sa.Index("ix_inference_job_items_status", "status"),
)
sa.Table(
    "masks",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("image_id", sa.Integer(), nullable=False),
    sa.Column("fully_annotated", sa.Boolean(), nullable=False),
    sa.Column("file_path", sa.String(), nullable=False),
    sa.ForeignKeyConstraint(["image_id"], ["images.id"], name="masks_image_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="masks_pkey"),
    sa.Index("ix_masks_image_id", "image_id"),
)
sa.Table(
    "annotation_actions",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("mask_id", sa.Integer(), nullable=False),
    sa.Column("username", sa.String(), nullable=False),
    sa.Column("group_id", sa.String(length=64), nullable=True),
    sa.Column("action_type", sa.String(length=16), nullable=False),
    sa.Column("payload", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(), nullable=False),
    sa.Column("undone", sa.Boolean(), nullable=False),
    sa.ForeignKeyConstraint(["mask_id"], ["masks.id"], name="annotation_actions_mask_id_fkey", ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["username"], ["users.username"], name="annotation_actions_username_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="annotation_actions_pkey"),
    sa.Index("ix_annotation_actions_group_id", "group_id"),
    sa.Index("ix_annotation_actions_mask_id", "mask_id"),
    sa.Index("ix_annotation_actions_stack", "mask_id", "username", "undone"),
    sa.Index("ix_annotation_actions_username", "username"),
)
sa.Table(
    "contours",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("mask_id", sa.Integer(), nullable=False),
    sa.Column("parent_id", sa.Integer(), nullable=True),
    sa.Column("temporary", sa.Boolean(), nullable=False),
    sa.Column("added_by", sa.String(length=255), nullable=False),
    sa.Column("author_username", sa.String(), nullable=True),
    sa.Column("created_at", sa.DateTime(), nullable=False),
    sa.Column("confidence_score", sa.Float(), nullable=False),
    sa.Column("label_id", sa.Integer(), nullable=True),
    sa.Column("area", sa.Float(), nullable=False),
    sa.Column("perimeter", sa.Float(), nullable=False),
    sa.Column("circularity", sa.Float(), nullable=False),
    sa.Column("diameter", sa.Float(), nullable=False),
    sa.Column("x", sa.JSON(), nullable=False),
    sa.Column("y", sa.JSON(), nullable=False),
    sa.ForeignKeyConstraint(["author_username"], ["users.username"], name="contours_author_username_fkey", ondelete="SET NULL"),
    sa.ForeignKeyConstraint(["label_id"], ["labels.id"], name="contours_label_id_fkey", ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["mask_id"], ["masks.id"], name="contours_mask_id_fkey", ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["parent_id"], ["contours.id"], name="contours_parent_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="contours_pkey"),
    sa.Index("ix_contours_author_username", "author_username"),
    sa.Index("ix_contours_label_id", "label_id"),
    sa.Index("ix_contours_mask_id", "mask_id"),
    sa.Index("ix_contours_parent_id", "parent_id"),
)
sa.Table(
    "annotation_rejections",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("mask_id", sa.Integer(), nullable=False),
    sa.Column("contour_id", sa.Integer(), nullable=True),
    sa.Column("reason", sa.String(length=32), nullable=False),
    sa.Column("note", sa.String(length=1000), nullable=True),
    sa.Column("created_by", sa.String(), nullable=True),
    sa.Column("created_at", sa.DateTime(), nullable=False),
    sa.Column("resolved_at", sa.DateTime(), nullable=True),
    sa.Column("resolved_by", sa.String(), nullable=True),
    sa.Column("resolution", sa.String(length=16), nullable=True),
    sa.ForeignKeyConstraint(["contour_id"], ["contours.id"], name="annotation_rejections_contour_id_fkey", ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["created_by"], ["users.username"], name="annotation_rejections_created_by_fkey", ondelete="SET NULL"),
    sa.ForeignKeyConstraint(["mask_id"], ["masks.id"], name="annotation_rejections_mask_id_fkey", ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["resolved_by"], ["users.username"], name="annotation_rejections_resolved_by_fkey", ondelete="SET NULL"),
    sa.PrimaryKeyConstraint("id", name="annotation_rejections_pkey"),
    sa.Index("ix_annotation_rejections_mask_id", "mask_id"),
    sa.Index("ix_annotation_rejections_open", "mask_id", "resolved_at"),
)
sa.Table(
    "contour_metrics",
    schema,
    sa.Column("contour_id", sa.Integer(), nullable=False),
    sa.Column("metric_key", sa.String(length=64), nullable=False),
    sa.Column("component", sa.SmallInteger(), nullable=False),
    sa.Column("value", sa.Float(), nullable=False),
    sa.Column("unit", sa.String(length=16), nullable=True),
    sa.Column("computed_at", sa.DateTime(), nullable=False),
    sa.Column("stale", sa.Boolean(), nullable=False),
    sa.ForeignKeyConstraint(["contour_id"], ["contours.id"], name="contour_metrics_contour_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("contour_id", "metric_key", "component", name="contour_metrics_pkey"),
    sa.Index("ix_contour_metrics_metric_key", "metric_key"),
)
sa.Table(
    "embeddings",
    schema,
    sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
    sa.Column("image_id", sa.Integer(), nullable=True),
    sa.Column("contour_id", sa.Integer(), nullable=True),
    sa.Column("kind", sa.String(length=32), nullable=False),
    sa.Column("model_id", sa.String(length=128), nullable=False),
    sa.Column("dim", sa.Integer(), nullable=False),
    sa.Column("vector", pgvector.sqlalchemy.Vector(dim=768), nullable=False),
    sa.Column("created_at", sa.DateTime(), nullable=False),
    sa.CheckConstraint("(image_id IS NOT NULL) <> (contour_id IS NOT NULL)", name="ck_embeddings_one_subject"),
    sa.ForeignKeyConstraint(["contour_id"], ["contours.id"], name="embeddings_contour_id_fkey", ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["image_id"], ["images.id"], name="embeddings_image_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("id", name="embeddings_pkey"),
    sa.Index("ix_embeddings_vector_hnsw", "vector", postgresql_using="hnsw", postgresql_ops={"vector": "vector_cosine_ops"}),
    sa.Index("uq_embeddings_contour_kind_model", "contour_id", "kind", "model_id", unique=True, postgresql_where=sa.text("contour_id IS NOT NULL")),
    sa.Index("uq_embeddings_image_kind_model", "image_id", "kind", "model_id", unique=True, postgresql_where=sa.text("image_id IS NOT NULL")),
)
sa.Table(
    "reviewer_contour_association",
    schema,
    sa.Column("reviewer_id", sa.String(), nullable=False),
    sa.Column("contour_id", sa.Integer(), nullable=False),
    sa.ForeignKeyConstraint(["contour_id"], ["contours.id"], name="reviewer_contour_association_contour_id_fkey", ondelete="CASCADE"),
    sa.ForeignKeyConstraint(["reviewer_id"], ["users.username"], name="reviewer_contour_association_reviewer_id_fkey", ondelete="CASCADE"),
    sa.PrimaryKeyConstraint("reviewer_id", "contour_id", name="reviewer_contour_association_pkey"),
    sa.Index("ix_reviewer_contour_association_contour_id", "contour_id"),
)


def upgrade() -> None:
    bind = op.get_bind()
    existing = set(sa.inspect(bind).get_table_names())
    adopted = [table for table in schema.sorted_tables if table.name in existing]

    # Every refusal comes before any change. A refusal would roll the whole
    # transaction back anyway, but this way the log never shows work being done
    # that did not happen.
    inspector = sa.inspect(bind)
    for table in adopted:
        _refuse_unfillable_columns(inspector, table)

    if adopted:
        logger.info("Adopting %d existing tables.", len(adopted))
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    for table in adopted:
        _add_missing_columns(bind, table)
    schema.create_all(bind, tables=[table for table in schema.sorted_tables
                                    if table.name not in existing])
    if "datasets" in existing and not _has_index(bind, "datasets", "uq_datasets_name"):
        _deduplicate_dataset_names(bind)
    for table in adopted:
        _add_missing_indexes(bind, table)
    if "datasets_name_key" in {constraint["name"] for constraint
                               in sa.inspect(bind).get_unique_constraints("datasets")}:
        logger.info("Dropping datasets_name_key; uq_datasets_name enforces the same rule.")
        op.drop_constraint("datasets_name_key", "datasets", type_="unique")


def downgrade() -> None:
    # Going below the baseline means having no schema at all. The pgvector
    # extension stays installed: it holds no data, and other schemas may use it.
    schema.drop_all(op.get_bind())


def _refuse_unfillable_columns(inspector, table: sa.Table) -> None:
    """Stop when the database needs a data migration this revision cannot do.

    A missing NOT NULL column without a default cannot be added to a table that
    has rows, and a stray NOT NULL column without a default makes every insert
    fail. Both mean the database skipped one of the ``scripts/migrate_*.py`` data
    migrations -- the only known case being the roles migration.
    """
    present = {column["name"]: column for column in inspector.get_columns(table.name)}
    for column in table.columns:
        if column.name not in present and not column.nullable and column.server_default is None:
            raise RuntimeError(
                f"{table.name}.{column.name} is missing, and as a NOT NULL column without a "
                "default it cannot be added automatically. This database predates a data "
                "migration: run scripts/migrate_roles.py, then start the backend again."
            )
    frozen = {column.name for column in table.columns}
    for name, column in present.items():
        if name not in frozen and not column["nullable"] and column.get("default") is None:
            raise RuntimeError(
                f"{table.name}.{name} is not part of the schema, and as a NOT NULL column "
                "without a default it makes every insert fail. This database predates a "
                "data migration: run scripts/migrate_roles.py, then start the backend again."
            )


def _add_missing_columns(bind, table: sa.Table) -> None:
    present = {column["name"] for column in sa.inspect(bind).get_columns(table.name)}
    for column in table.columns:
        if column.name in present:
            continue
        logger.info("Adding missing column %s.%s.", table.name, column.name)
        default = column.server_default.arg if column.server_default is not None else None
        op.add_column(table.name, sa.Column(column.name, column.type,
                                            nullable=column.nullable, server_default=default))


def _has_index(bind, table: str, name: str) -> bool:
    return name in {index["name"] for index in sa.inspect(bind).get_indexes(table)}


def _add_missing_indexes(bind, table: sa.Table) -> None:
    present = {index["name"] for index in sa.inspect(bind).get_indexes(table.name)}
    for index in table.indexes:
        if index.name not in present:
            logger.info("Creating missing index %s.", index.name)
            index.create(bind)


def _deduplicate_dataset_names(bind) -> None:
    """Rename duplicate dataset names so ``uq_datasets_name`` can be built.

    Only a database that never booted the code which made names unique can still
    hold duplicates. The oldest dataset keeps its name and later ones get
    " (dup <id>)" -- the scheme that code used.
    """
    seen: set[str] = set()
    for dataset_id, name in bind.execute(sa.text("SELECT id, name FROM datasets ORDER BY id")):
        name = name or ""
        if name not in seen:
            seen.add(name)
            continue
        counter = 0
        while True:
            suffix = f" (dup {dataset_id})" if counter == 0 else f" (dup {dataset_id}_{counter})"
            new_name = name[: 50 - len(suffix)] + suffix
            if new_name not in seen:
                break
            counter += 1
        logger.warning("Renaming duplicate dataset name %r (id %s) to %r.",
                       name, dataset_id, new_name)
        bind.execute(sa.text("UPDATE datasets SET name = :name WHERE id = :id"),
                     {"name": new_name, "id": dataset_id})
        seen.add(new_name)


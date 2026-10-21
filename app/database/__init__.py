import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone

import sqlite3

from sqlalchemy import MetaData, create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL, DATA_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key enforcement on every SQLite connection.

    SQLite disables foreign key constraints by default, which means the
    ``ON DELETE CASCADE`` rules declared on our tables are silently ignored.
    Without this, deleting a dataset/image/label/mask leaves orphaned rows
    (images, labels, masks, contours) behind. The check keeps this a no-op for
    non-SQLite backends.
    """
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


#: Names for constraints a model does not name itself. They copy PostgreSQL's own
#: defaults, which is what every such constraint in a database built before Alembic
#: already carries -- so a migration can refer to an existing constraint by name (to
#: drop or alter it) and hit. There is deliberately no "ck" entry: a convention that
#: uses ``%(constraint_name)s`` is applied on top of explicit names as well, and would
#: rename ``ck_embeddings_one_subject``.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "%(table_name)s_%(column_0_N_name)s_key",
    "fk": "%(table_name)s_%(column_0_N_name)s_fkey",
    "pk": "%(table_name)s_pkey",
}

# Define the declarative general
database = declarative_base(metadata=MetaData(naming_convention=NAMING_CONVENTION))

engine = create_engine(DATABASE_URL,
                       pool_size=20,  # Default is usually 5
                       max_overflow=50,  # Increase from default 10
                       pool_pre_ping=True,  # Validate connections
                       pool_recycle=3600,  # Recycle after 1 hour
                       )

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def import_models():
    """Import every model module so the metadata describes every table.

    Tables referenced only by string in relationships (e.g. ``dataset_members``)
    are otherwise never imported, which leaves SQLAlchemy unable to resolve the
    mapper, ``create_all`` without the table, and Alembic's autogenerate proposing
    to drop it.
    """
    from app.database import (  # noqa: F401  (imported for their side effects)
        ai_suggestions,
        annotation_actions,
        annotation_queues,
        contour_metrics,
        contours,
        credentials,
        dataset_calibration_defaults,
        dataset_members,
        dataset_metadata_keys,
        datasets,
        derived_metrics,
        embeddings,
        image_calibrations,
        image_metadata,
        images,
        instance_settings,
        dataset_model_routing_configs,
        inference_jobs,
        labels,
        masks,
        model_favorites,
        model_label_mappings,
        organizations,
        quantification_profiles,
        rejections,
        scene_graph,
        stacks,
        activity_log_settings,
        user_events,
        users,
    )


#: Columns added to existing tables after their first release, as
#: (table, column, DDL type). ``create_all`` creates missing *tables* but never
#: ALTERs an existing one, so a nullable column added to a shipped model has to be
#: patched in here for databases that predate it. Each entry is idempotent — it is
#: only applied when the column is absent.
#:
#: Frozen: this only serves the SQLite path of :func:`init_db`. PostgreSQL schema
#: changes are Alembic revisions under ``migrations/versions``.
_ADDED_COLUMNS = [
    ("annotation_rejections", "resolution", "VARCHAR(16)"),
    # Added with the metadata type system; a dev database that ran the untyped
    # first cut of image_metadata has the table but not this column.
    ("image_metadata", "value_num", "FLOAT"),
    # Dataset activity summary: who finished a mask and when an approval was given.
    ("masks", "fully_annotated_by", "VARCHAR"),
    ("masks", "fully_annotated_at", "TIMESTAMP"),
    ("reviewer_contour_association", "reviewed_at", "TIMESTAMP"),
    # SQLite's create_all() does not change tables already present in an older
    # development database. Keep these additions in step with the Alembic
    # revisions without rebuilding tables or rewriting annotation rows.
    ("users", "display_name", "VARCHAR(100)"),
    ("users", "email", "VARCHAR(254)"),
    ("users", "preferences", "JSON NOT NULL DEFAULT '{}'"),
    ("users", "created_at", "TIMESTAMP"),
    ("users", "last_login_at", "TIMESTAMP"),
    ("users", "tokens_valid_after", "TIMESTAMP"),
    ("users", "must_change_password", "BOOLEAN NOT NULL DEFAULT FALSE"),
    ("datasets", "organization_id",
     "INTEGER REFERENCES organizations(id) ON DELETE RESTRICT"),
    ("ai_suggestions", "final_iou", "FLOAT"),
    ("contours", "model_class", "VARCHAR(255)"),
    ("image_metadata", "stack_id",
     "INTEGER REFERENCES stacks(id) ON DELETE CASCADE"),
    ("images", "kind", "VARCHAR(16) NOT NULL DEFAULT 'image'"),
    ("images", "stack_id", "INTEGER REFERENCES stacks(id) ON DELETE CASCADE"),
    ("images", "frame_index", "INTEGER"),
    ("images", "frame_position", "FLOAT"),
    ("images", "overview_geometry", "JSON"),
]

# SQLite indexes for the columns added above and for older tables whose indexes
# were not created by create_all(). The vector HNSW index is intentionally absent:
# it is PostgreSQL-only and SQLite uses the NumPy cosine-search path.
_SQLITE_INDEXES = [
    ("reviewer_contour_association", "ix_reviewer_contour_association_contour_id",
     "CREATE INDEX IF NOT EXISTS ix_reviewer_contour_association_contour_id "
     "ON reviewer_contour_association (contour_id)", ()),
    ("contours", "ix_contours_author_username",
     "CREATE INDEX IF NOT EXISTS ix_contours_author_username ON contours (author_username)", ()),
    ("contours", "ix_contours_label_id",
     "CREATE INDEX IF NOT EXISTS ix_contours_label_id ON contours (label_id)", ()),
    ("contours", "ix_contours_mask_id",
     "CREATE INDEX IF NOT EXISTS ix_contours_mask_id ON contours (mask_id)", ()),
    ("contours", "ix_contours_parent_id",
     "CREATE INDEX IF NOT EXISTS ix_contours_parent_id ON contours (parent_id)", ()),
    ("contours", "ix_contours_suggestion_id",
     "CREATE INDEX IF NOT EXISTS ix_contours_suggestion_id ON contours (suggestion_id)", ()),
    ("users", "users_email_key",
     "CREATE UNIQUE INDEX IF NOT EXISTS users_email_key ON users (email)", ("email",)),
    ("datasets", "ix_datasets_organization_id",
     "CREATE INDEX IF NOT EXISTS ix_datasets_organization_id ON datasets (organization_id)", ()),
    ("images", "ix_images_dataset_id",
     "CREATE INDEX IF NOT EXISTS ix_images_dataset_id ON images (dataset_id)", ()),
    ("images", "ix_images_kind",
     "CREATE INDEX IF NOT EXISTS ix_images_kind ON images (kind)", ()),
    ("images", "ix_images_stack_id",
     "CREATE INDEX IF NOT EXISTS ix_images_stack_id ON images (stack_id)", ()),
    ("images", "uq_images_stack_frame_index",
     "CREATE UNIQUE INDEX IF NOT EXISTS uq_images_stack_frame_index "
     "ON images (stack_id, frame_index)", ("stack_id", "frame_index")),
    ("masks", "ix_masks_image_id",
     "CREATE INDEX IF NOT EXISTS ix_masks_image_id ON masks (image_id)", ()),
    ("image_metadata", "ix_image_metadata_stack_id",
     "CREATE INDEX IF NOT EXISTS ix_image_metadata_stack_id ON image_metadata (stack_id)", ()),
    ("image_metadata", "uq_image_metadata_stack_key",
     "CREATE UNIQUE INDEX IF NOT EXISTS uq_image_metadata_stack_key "
     "ON image_metadata (stack_id, key)", ("stack_id",)),
    ("organizations", "uq_organizations_default",
     "CREATE UNIQUE INDEX IF NOT EXISTS uq_organizations_default "
     "ON organizations (is_default) WHERE is_default", ()),
]


def _default_organization_name(connection) -> str:
    """Return the same display name chosen by migration 0005."""
    configured_name = None
    if inspect(connection).has_table("instance_settings"):
        configured_name = connection.execute(text(
            "SELECT value FROM instance_settings WHERE key = 'instance_name'"
        )).scalar()
    configured_name = (configured_name or os.getenv("INSTANCE_NAME") or "").strip()
    return configured_name[:100] or "Default organisation"


def _backfill_default_organization(connection):
    """Match migration 0005 for a database gaining organization_id for the first time.

    This runs only when the old datasets table lacked organization_id. It does not
    repair memberships or assign intentionally personal datasets on later boots.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    organization_id = connection.execute(text(
        "SELECT id FROM organizations WHERE is_default = 1"
    )).scalar()
    if organization_id is None:
        connection.execute(text(
            "INSERT INTO organizations (name, is_default, created_at) "
            "VALUES (:name, 1, :now)"
        ), {"name": _default_organization_name(connection), "now": now})
        organization_id = connection.execute(text(
            "SELECT id FROM organizations WHERE is_default = 1"
        )).scalar_one()
    connection.execute(text(
        "INSERT OR IGNORE INTO organization_members "
        "(organization_id, username, role, joined_at) "
        "SELECT :organization_id, username, "
        "CASE WHEN global_role = 'admin' THEN 'admin' ELSE 'member' END, :now "
        "FROM users"
    ), {"organization_id": organization_id, "now": now})
    connection.execute(text(
        "UPDATE datasets SET organization_id = :organization_id "
        "WHERE organization_id IS NULL"
    ), {"organization_id": organization_id})


def _ensure_columns(target_engine: Engine | None = None):
    """Bring older SQLite tables up to the additive schema expected by current models.

    SQLite DDL must share a real transaction with the organization backfill. Python's
    sqlite3 legacy transaction mode does not begin a transaction for DDL, so use an
    explicit BEGIN IMMEDIATE and roll back the whole upgrade if any step fails.
    """
    db_engine = target_engine or engine
    with db_engine.connect() as connection:
        try:
            if db_engine.dialect.name == "sqlite":
                # Lock before inspection so another startup cannot leave us with a
                # stale plan that tries to add the same column after it commits.
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                connection.begin()

            inspector = inspect(connection)
            existing_tables = set(inspector.get_table_names())
            columns_by_table = {
                table: {column["name"] for column in inspector.get_columns(table)}
                for table in existing_tables
            }
            existing_indexes = {
                table: {index["name"] for index in inspector.get_indexes(table)}
                for table in existing_tables
            }
            unique_column_sets = {
                table: {tuple(constraint["column_names"]) for constraint in
                        inspector.get_unique_constraints(table)}
                for table in existing_tables
            }
            legacy_dataset_organization = (
                db_engine.dialect.name == "sqlite"
                and "datasets" in existing_tables
                and "organization_id" not in columns_by_table["datasets"]
            )
            scheduled_columns = {
                (table, column)
                for table, column, _ddl in _ADDED_COLUMNS
                if table in existing_tables and column not in columns_by_table[table]
            }
            has_work = any(
                table in existing_tables and column not in columns_by_table[table]
                for table, column, _ in _ADDED_COLUMNS
            )
            if db_engine.dialect.name == "sqlite":
                has_work = has_work or any(
                    table in existing_tables
                    and name not in existing_indexes[table]
                    and all(
                        column in columns_by_table[table] or (table, column) in scheduled_columns
                        for column in required_columns
                    )
                    and not (required_columns and tuple(required_columns) in unique_column_sets[table])
                    for table, name, _ddl, required_columns in _SQLITE_INDEXES
                )
            if not has_work:
                connection.commit()
                return

            for table, column, ddl in _ADDED_COLUMNS:
                if table not in existing_tables or column in columns_by_table[table]:
                    continue
                logger.info("Adding missing column %s.%s", table, column)
                connection.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                columns_by_table[table].add(column)

            if db_engine.dialect.name == "sqlite":
                for table, name, ddl, required_columns in _SQLITE_INDEXES:
                    if table not in existing_tables or name in existing_indexes[table]:
                        continue
                    if not all(column in columns_by_table[table] for column in required_columns):
                        continue
                    if required_columns and tuple(required_columns) in unique_column_sets[table]:
                        continue
                    connection.exec_driver_sql(ddl)
                    existing_indexes[table].add(name)

                if legacy_dataset_organization:
                    _backfill_default_organization(connection)

            connection.commit()
        except Exception:
            connection.rollback()
            raise


def _ensure_dataset_name_uniqueness(target_engine: Engine | None = None):
    """Ensure dataset names are unique across existing databases.

    Deduplicates pre-existing identical names (by suffixing duplicates with
    their id) and creates a unique index so concurrent creations/imports fail
    fast with IntegrityError rather than corrupting dataset directories or queries.
    """
    db_engine = target_engine or engine
    inspector = inspect(db_engine)
    existing_tables = set(inspector.get_table_names())
    if "datasets" not in existing_tables:
        return

    with db_engine.begin() as connection:
        rows = connection.execute(text("SELECT id, name FROM datasets ORDER BY id ASC")).fetchall()
        seen_names = set()
        for ds_id, ds_name in rows:
            ds_name = ds_name or ""
            if ds_name in seen_names:
                suffix = f" (dup {ds_id})"
                base = ds_name[: 50 - len(suffix)]
                new_name = f"{base}{suffix}"
                counter = 1
                while new_name in seen_names:
                    suffix = f" (dup {ds_id}_{counter})"
                    base = ds_name[: 50 - len(suffix)]
                    new_name = f"{base}{suffix}"
                    counter += 1

                logger.warning(
                    "Deduplicating dataset id=%s: renaming duplicate '%s' to '%s'",
                    ds_id,
                    ds_name,
                    new_name,
                )
                connection.execute(
                    text("UPDATE datasets SET name = :new_name WHERE id = :id"),
                    {"new_name": new_name, "id": ds_id},
                )
                seen_names.add(new_name)
            else:
                seen_names.add(ds_name)

        connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_datasets_name ON datasets (name)"))


def init_db(target_engine: Engine | None = None):
    """Bring the schema in line with the models.

    PostgreSQL -- every real deployment -- is migrated by Alembic: the database is
    upgraded to the newest revision under ``migrations/versions``, and a database
    from before Alembic is adopted on the way (see the baseline revision).

    Anything else is the SQLite used by tests and throwaway dev databases, which
    keeps building its schema straight from the models. Revisions are written for
    PostgreSQL only, so a SQLite database never gets later ALTERs; delete it and
    let it be recreated instead.
    """
    logger.debug("\tInitializing database")
    import_models()
    db_engine = target_engine or engine
    if db_engine.dialect.name == "postgresql":
        from app.database.migrate import upgrade_to_head
        upgrade_to_head(db_engine)
        return
    build_schema_from_models(db_engine)


def build_schema_from_models(target_engine: Engine | None = None):
    """The pre-Alembic schema path: ``create_all`` plus the hand-written patches.

    What ``init_db`` does for SQLite, and what ``scripts/migrate_roles.py`` needs on
    a PostgreSQL database too old for the baseline revision to adopt. Refused on a
    database Alembic already manages, where ``create_all`` would create tables
    ahead of the revisions that are meant to create them.
    """
    import_models()
    db_engine = target_engine or engine
    if db_engine.dialect.name == "postgresql" and inspect(db_engine).has_table("alembic_version"):
        raise RuntimeError("This database is managed by Alembic; use init_db() instead.")
    database.metadata.create_all(bind=db_engine)
    _ensure_columns(db_engine)
    _ensure_dataset_name_uniqueness(db_engine)


def get_session():
    session = SessionLocal()
    logging.info(f"DB connections: {engine.pool.checkedout()}")
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_context_session():
    session = SessionLocal()
    logging.info(f"DB connections: {engine.pool.checkedout()}")
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

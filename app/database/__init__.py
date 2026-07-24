import logging
import os
from contextlib import contextmanager

import sqlite3

from sqlalchemy import create_engine, event, inspect, text
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


# Define the declarative general
database = declarative_base()

engine = create_engine(DATABASE_URL,
                       pool_size=20,  # Default is usually 5
                       max_overflow=50,  # Increase from default 10
                       pool_pre_ping=True,  # Validate connections
                       pool_recycle=3600,  # Recycle after 1 hour
                       )

database.metadata.create_all(engine)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _import_models():
    """Import every model module so `create_all` sees the full metadata.

    Tables referenced only by string in relationships (e.g. ``dataset_members``)
    are otherwise never imported, which leaves SQLAlchemy unable to resolve the
    mapper and the table missing from a fresh database.
    """
    from app.database import (  # noqa: F401  (imported for their side effects)
        annotation_queues,
        contour_metrics,
        contours,
        dataset_members,
        datasets,
        images,
        labels,
        masks,
        quantification_profiles,
        rejections,
        scans,
        users,
    )


#: Columns added to existing tables after their first release, as
#: (table, column, DDL type). ``create_all`` creates missing *tables* but never
#: ALTERs an existing one, so a nullable column added to a shipped model has to be
#: patched in here for databases that predate it. Each entry is idempotent — it is
#: only applied when the column is absent.
_ADDED_COLUMNS = [
    ("annotation_rejections", "resolution", "VARCHAR(16)"),
]


def _ensure_columns():
    """Add late-arriving nullable columns to already-created tables.

    Mirrors ``scripts/migrate_roles.py`` but runs on every boot so a dev database
    stays in step with the models without a manual migration step. Safe on a fresh
    database: ``create_all`` has already made the columns, so every check is a hit.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as connection:
        for table, column, ddl in _ADDED_COLUMNS:
            if table not in existing_tables:
                continue
            columns = {col["name"] for col in inspector.get_columns(table)}
            if column in columns:
                continue
            logger.info("Adding missing column %s.%s", table, column)
            connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


def init_db():
    logger.debug("\tInitializing database")
    _import_models()
    database.metadata.create_all(bind=engine)
    _ensure_columns()


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

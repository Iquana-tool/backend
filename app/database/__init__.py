import logging
import os
from contextlib import contextmanager

import sqlite3

from sqlalchemy import create_engine, event
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


def init_db():
    logger.debug("\tInitializing database")
    database.metadata.create_all(bind=engine)


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

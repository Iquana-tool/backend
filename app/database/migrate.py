"""Keeping a PostgreSQL database on the newest Alembic revision.

``init_db`` calls :func:`upgrade_to_head` on every boot, so pulling new code and
restarting is the whole upgrade -- nobody runs a migration script by hand. The same
call covers all three states a database can be in: empty, from before Alembic, and
already versioned. The baseline revision knows how to adopt the first two (see
``migrations/versions/*_baseline.py``).

Every process that builds the app gets here, and a deployment can run several
(uvicorn workers, a second replica). The upgrade therefore runs in one transaction
under a PostgreSQL advisory lock: the first process migrates, the others wait for it
and then find nothing left to do. PostgreSQL DDL is transactional, so a revision that
fails rolls back whole instead of leaving a half-migrated schema behind.
"""
from __future__ import annotations

import os
from logging import getLogger

import pgvector.sqlalchemy  # noqa: F401 -- teaches reflection the "vector" column type
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine

from app.database import database
from config import AUTO_MIGRATE, ROOT_DIR

logger = getLogger(__name__)

ALEMBIC_INI = os.path.join(ROOT_DIR, "alembic.ini")

#: Key for ``pg_advisory_xact_lock``. Arbitrary, but fixed: every process has to
#: ask for the same one, and it should not collide with another application's lock
#: on a shared server.
_MIGRATION_LOCK_KEY = 7_140_526_031


def alembic_config(connection: Connection | None = None) -> Config:
    """The Alembic configuration, optionally bound to an open connection.

    A bound connection makes the migration run inside the caller's transaction,
    and under the caller's lock, instead of opening a connection of its own.
    """
    config = Config(ALEMBIC_INI)
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def head_revision() -> str:
    """The newest revision the code ships."""
    return ScriptDirectory.from_config(alembic_config()).get_current_head()


def current_revision(connection: Connection) -> str | None:
    """The revision the database is stamped with, or None when it has none."""
    return MigrationContext.configure(connection).get_current_revision()


def schema_drift(connection: Connection) -> list:
    """Every difference between the database and the models, as Alembic sees it.

    Empty when the schema matches. Used after adopting a database from before
    Alembic, where it catches whatever the old ``create_all`` boot path left
    behind that the baseline did not know to fix.
    """
    return compare_metadata(MigrationContext.configure(connection), database.metadata)


def upgrade_to_head(db_engine: Engine) -> None:
    """Upgrade the database to the newest revision, adopting it first if needed."""
    head = head_revision()

    if not AUTO_MIGRATE:
        with db_engine.connect() as connection:
            current = current_revision(connection)
        if current != head:
            logger.warning(
                "Database schema is at revision %s but this code expects %s, and "
                "IQUANA_AUTO_MIGRATE is off. Run `alembic upgrade head` from backend/.",
                current, head,
            )
        return

    with db_engine.begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _MIGRATION_LOCK_KEY})
        current = current_revision(connection)
        if current == head:
            return

        # An empty alembic_version table is left behind by e.g. a failed first
        # upgrade, and says nothing about the database holding data.
        adopting = current is None and bool(
            set(inspect(connection).get_table_names()) - {"alembic_version"})
        if adopting:
            logger.warning("Database has no Alembic revision; adopting it at the baseline.")
        logger.info("Migrating database schema from %s to %s.",
                    current or ("an unversioned database" if adopting else "an empty database"),
                    head)
        command.upgrade(alembic_config(connection), "head")

        if adopting:
            for difference in schema_drift(connection):
                logger.warning("Schema still differs from the models after adoption: %s",
                               difference)

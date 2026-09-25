"""Checks for the Alembic migrations and the boot-time upgrade.

Revisions are written for PostgreSQL, so these run against a real server: each test
gets a throwaway database there, created and dropped around it. The server comes
from ``MIGRATION_TEST_DATABASE_URL``, or ``DATABASE_URL`` when that is PostgreSQL;
without one the tests skip, which keeps the usual SQLite run of the suite working.

The first test is the one that catches a model change without a revision: a fresh
database upgraded to head has to match the models exactly.
"""
import logging
import os
import uuid

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from alembic import command

from app.database import build_schema_from_models, init_db
from app.database import migrate
from app.database.migrate import alembic_config, current_revision, head_revision, schema_drift
from config import DATABASE_URL


def _server_url():
    url = make_url(os.getenv("MIGRATION_TEST_DATABASE_URL") or DATABASE_URL)
    return url if url.get_backend_name() == "postgresql" else None


@pytest.fixture
def pg_engine():
    """An engine on a fresh, empty PostgreSQL database that is dropped afterwards."""
    server = _server_url()
    if server is None:
        pytest.skip("Migrations are PostgreSQL-only; set MIGRATION_TEST_DATABASE_URL to run them.")
    name = f"iquana_migration_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(server.set(database="postgres"), isolation_level="AUTOCOMMIT",
                          connect_args={"connect_timeout": 5})
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{name}"'))
    except OperationalError as exc:
        admin.dispose()
        pytest.skip(f"PostgreSQL is not reachable: {exc}")

    engine = create_engine(server.set(database=name), poolclass=NullPool)
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def _make_legacy(engine):
    """Turn a database at the baseline into what the pre-Alembic boot path left.

    Built from the baseline revision rather than from the models, so the shape stays
    that of 2026-09-25 however far the models move on. The quirks are the ones real
    databases were found with: no revision stamp, ``create_all``'s unique constraint
    next to ``uq_datasets_name``, and a column patched in by ALTER without its index.
    """
    with engine.begin() as connection:
        command.upgrade(alembic_config(connection), "0001")
        connection.execute(text("DROP TABLE alembic_version"))
        connection.execute(text("ALTER TABLE datasets ADD CONSTRAINT datasets_name_key UNIQUE (name)"))
        connection.execute(text("DROP INDEX ix_image_metadata_value_num"))
        connection.execute(text("ALTER TABLE image_metadata DROP COLUMN value_num"))
        connection.execute(text(
            "INSERT INTO users (username, hashed_password, global_role, is_active) "
            "VALUES ('alice', 'x', 'admin', true)"))
        connection.execute(text(
            "INSERT INTO datasets (name, dataset_type, folder_path, created_by, "
            "require_independent_review) VALUES ('Reef', 'image', '/tmp/reef', 'alice', false)"))


def test_fresh_database_reaches_head_matching_the_models(pg_engine):
    init_db(target_engine=pg_engine)

    with pg_engine.connect() as connection:
        assert current_revision(connection) == head_revision()
        # A difference here means a model changed without a revision to match.
        assert schema_drift(connection) == []


def test_second_boot_changes_nothing(pg_engine):
    init_db(target_engine=pg_engine)
    init_db(target_engine=pg_engine)

    with pg_engine.connect() as connection:
        assert current_revision(connection) == head_revision()


def test_baseline_downgrades_to_nothing_and_back(pg_engine):
    init_db(target_engine=pg_engine)
    with pg_engine.begin() as connection:
        command.downgrade(alembic_config(connection), "base")
    assert set(inspect(pg_engine).get_table_names()) == {"alembic_version"}

    init_db(target_engine=pg_engine)
    with pg_engine.connect() as connection:
        assert schema_drift(connection) == []


def test_adopts_a_database_from_before_alembic(pg_engine):
    _make_legacy(pg_engine)

    init_db(target_engine=pg_engine)

    inspector = inspect(pg_engine)
    assert "datasets_name_key" not in {c["name"] for c in inspector.get_unique_constraints("datasets")}
    assert "ix_image_metadata_value_num" in {i["name"] for i in inspector.get_indexes("image_metadata")}
    with pg_engine.connect() as connection:
        assert current_revision(connection) == head_revision()
        assert schema_drift(connection) == []
        assert connection.execute(text("SELECT name FROM datasets")).scalars().all() == ["Reef"]


def test_refuses_a_database_that_needs_the_roles_migration(pg_engine):
    with pg_engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE users (username VARCHAR PRIMARY KEY, "
            "hashed_password VARCHAR NOT NULL, is_admin BOOLEAN NOT NULL)"))
        connection.execute(text("INSERT INTO users VALUES ('alice', 'x', true)"))

    with pytest.raises(RuntimeError, match="migrate_roles"):
        init_db(target_engine=pg_engine)

    # The refusal rolled the whole upgrade back: nothing created, nothing stamped.
    assert set(inspect(pg_engine).get_table_names()) == {"users"}


def test_auto_migrate_off_only_warns(pg_engine, monkeypatch, caplog):
    monkeypatch.setattr(migrate, "AUTO_MIGRATE", False)

    with caplog.at_level(logging.WARNING, logger="app.database.migrate"):
        init_db(target_engine=pg_engine)

    assert "IQUANA_AUTO_MIGRATE is off" in caplog.text
    assert inspect(pg_engine).get_table_names() == []


def test_schema_from_models_is_refused_once_alembic_manages_the_database(pg_engine):
    init_db(target_engine=pg_engine)

    with pytest.raises(RuntimeError, match="managed by Alembic"):
        build_schema_from_models(pg_engine)

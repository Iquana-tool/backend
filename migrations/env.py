"""Alembic environment for the iquana backend.

Revisions target PostgreSQL only -- the database every deployment runs. SQLite
(tests, throwaway dev databases) builds its schema from the models instead; see
``app.database.init_db``.

There are two ways in:

* the CLI (``python -m alembic ...`` from backend/), which opens its own
  connection to ``DATABASE_URL``;
* ``app.database.migrate.upgrade_to_head`` on boot, which hands over a connection
  that is already inside a transaction and holds the migration lock.
"""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

import pgvector.sqlalchemy  # noqa: F401 -- teaches reflection the "vector" column type
from app.database import database, import_models
from app.database.embeddings import VectorType
from config import DATABASE_URL

config = context.config
app_connection = config.attributes.get("connection")

# Only the CLI configures logging. On boot the app has set up its own, and
# fileConfig would reset it.
if app_connection is None and config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

import_models()
target_metadata = database.metadata


def render_item(type_, obj, autogen_context):
    """Write ``VectorType`` into revisions as the pgvector type it is on PostgreSQL.

    A revision must not import the models: it has to go on describing the schema
    as it was when it was written, whatever the model says later.
    """
    if type_ == "type" and isinstance(obj, VectorType):
        autogen_context.imports.add("import pgvector.sqlalchemy")
        return f"pgvector.sqlalchemy.Vector(dim={obj.dim})"
    return False


def run_migrations(connection) -> None:
    if connection.dialect.name != "postgresql":
        raise SystemExit(
            f"Migrations are written for PostgreSQL, not {connection.dialect.name}. "
            "A SQLite database builds its schema from the models on boot instead."
        )
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    raise SystemExit(
        "Offline (--sql) mode is not supported: the baseline revision inspects the "
        "live database to adopt it."
    )

if app_connection is not None:
    run_migrations(app_connection)
else:
    engine = create_engine(DATABASE_URL, poolclass=pool.NullPool)
    with engine.connect() as cli_connection:
        run_migrations(cli_connection)

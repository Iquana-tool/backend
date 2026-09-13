"""One-shot copy of the dev SQLite database into the configured Postgres database.

Both backends are described by the very same SQLAlchemy models, so this copies each
table *through the shared metadata* rather than as raw text: reading a row applies the
SQLite result processors (e.g. a JSON column's TEXT is ``json.loads``-ed back into a
Python object) and writing it applies the Postgres bind processors (that object is
re-serialised as Postgres ``json``). Column order, types and NULLs therefore round-trip
without hand-written per-table SQL.

Order matters: rows are inserted in ``metadata.sorted_tables`` order (a foreign-key
topological sort) so a child never lands before its parent. After loading, each table's
identity sequence is fast-forwarded past the copied ids -- otherwise the *next* ORM
INSERT would reuse id 1 and collide.

  * Source  = the SQLite file passed as ``--source`` (default: data/database.db).
  * Target  = whatever ``DATABASE_URL`` points at (must already be Postgres). The target
              schema is created via ``init_db()`` first, so a fresh empty database is fine.

The target must be empty. Re-running is a no-op unless you pass ``--wipe``, which
TRUNCATEs every table (CASCADE) before loading -- use it to redo a botched copy.

Usage (cwd = backend/, DATABASE_URL already set to Postgres in .env):
    backend/.venv/Scripts/python.exe scripts/copy_sqlite_to_postgres.py
    backend/.venv/Scripts/python.exe scripts/copy_sqlite_to_postgres.py --wipe
    backend/.venv/Scripts/python.exe scripts/copy_sqlite_to_postgres.py --source data/database.db
"""
import argparse
import logging
import os
import sys

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from sqlalchemy import Integer, create_engine, insert, inspect, select, text  # noqa: E402

from app.database import database, engine, init_db  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("copy_sqlite_to_postgres")


def _eval_default(column):
    """Produce a value from a column's Python-side default, or None if it has none.

    Scalar defaults return their constant; callable defaults (e.g. ``created_at``'s
    ``_utcnow``) are invoked. Used to backfill NOT NULL columns whose *legacy* rows
    predate the column and were left NULL in SQLite -- a NULL that SQLite kept but
    PostgreSQL's NOT NULL rejects. Columns without a default are left untouched (a
    genuine NULL there is bad data worth surfacing, not silently papering over).
    """
    default = column.default
    if default is None:
        return None
    if getattr(default, "is_scalar", False):
        return default.arg
    if getattr(default, "is_callable", False):
        fn = default.arg
        try:
            return fn()
        except TypeError:  # a default that expects the execution context
            return fn(None)
    return None


def _backfill_not_null_defaults(table, rows) -> int:
    """Fill NULLs in NOT NULL columns that carry a default, in place. Returns fill count."""
    defaulted = [c for c in table.columns if not c.nullable and c.default is not None]
    if not defaulted:
        return 0
    filled = 0
    for row in rows:
        for column in defaulted:
            if row.get(column.name) is None:
                row[column.name] = _eval_default(column)
                filled += 1
    return filled


def _reset_sequence(target_conn, table) -> None:
    """Advance a table's Postgres identity sequence past the largest copied id.

    Copied rows carry their original primary keys, but that never moves the SERIAL
    sequence, so the next ORM insert would try id 1 again. ``pg_get_serial_sequence``
    returns NULL for tables without an owned sequence (composite-key association
    tables), which we skip.
    """
    for pk in table.primary_key.columns:
        if not isinstance(pk.type, Integer):
            continue
        seq = target_conn.execute(
            text("SELECT pg_get_serial_sequence(:t, :c)"),
            {"t": table.name, "c": pk.name},
        ).scalar()
        if seq is None:
            continue
        target_conn.execute(
            text(
                f"SELECT setval(:seq, (SELECT COALESCE(MAX({pk.name}), 0) + 1 FROM {table.name}), false)"
            ),
            {"seq": seq},
        )
        logger.info("  reset sequence %s", seq)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        default=os.path.join(_BACKEND_ROOT, "data", "database.db"),
        help="Path to the source SQLite file (default: backend/data/database.db).",
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="TRUNCATE every target table (CASCADE) before loading. Destroys existing "
        "Postgres data -- only use to redo a copy.",
    )
    args = parser.parse_args()

    if engine.dialect.name != "postgresql":
        raise SystemExit(
            f"Target DATABASE_URL is {engine.dialect.name!r}, not postgresql. "
            "Point DATABASE_URL at Postgres before running."
        )
    if not os.path.exists(args.source):
        raise SystemExit(f"Source SQLite file not found: {args.source}")

    source_engine = create_engine("sqlite:///" + args.source.replace("\\", "/"))

    # Build the schema on the target (no-op for tables that already exist).
    init_db()

    tables = list(database.metadata.sorted_tables)
    source_tables = set(inspect(source_engine).get_table_names())

    with source_engine.connect() as src, engine.begin() as dst:
        # Refuse to load into a non-empty target unless --wipe was given.
        if not args.wipe:
            for table in tables:
                if table.name not in source_tables:
                    continue
                if dst.execute(select(text("1")).select_from(table).limit(1)).first():
                    raise SystemExit(
                        f"Target table {table.name!r} is not empty. Re-run with --wipe "
                        "to replace all target data, or point at a fresh database."
                    )

        if args.wipe:
            names = ", ".join(f'"{t.name}"' for t in tables)
            logger.info("Wiping %d target tables (CASCADE).", len(tables))
            dst.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))

        total = 0
        for table in tables:
            if table.name not in source_tables:
                logger.info("Skipping %s (absent from source).", table.name)
                continue
            # Order by primary key so a self-referential FK (contours.parent_id ->
            # contours.id) sees parents before children within the single INSERT.
            query = select(table).order_by(*table.primary_key.columns)
            rows = [dict(row._mapping) for row in src.execute(query)]
            filled = _backfill_not_null_defaults(table, rows)
            if rows:
                dst.execute(insert(table), rows)
                _reset_sequence(dst, table)
            suffix = f" ({filled} NULL default(s) backfilled)" if filled else ""
            logger.info("Copied %-32s %5d rows%s", table.name, len(rows), suffix)
            total += len(rows)

    logger.info("Done. Copied %d rows across %d tables into Postgres.", total, len(tables))


if __name__ == "__main__":
    main()

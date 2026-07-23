"""Migrate an existing database to the role-based access model.

The project has no Alembic setup — schema comes from ``metadata.create_all``, which
creates missing *tables* but never adds columns to existing ones. This script does
the rest, and is safe to re-run.

What it does:
  1. Adds the new columns to existing tables (``users.global_role``,
     ``users.is_active``, ``datasets.require_independent_review``,
     ``contours.author_username``, ``contours.created_at``).
  2. Backfills ``users.global_role`` from the old ``users.is_admin`` flag, then drops
     that column — it is ``NOT NULL`` with no default and the model no longer declares
     it, so leaving it behind makes every new-user INSERT fail.
  3. Copies ``dataset_user_association`` rows into ``dataset_members``, giving each
     existing collaborator a role (curator by default, since sharing previously
     implied unrestricted access).
  4. Gives every dataset's creator an ``owner`` membership row.
  5. Optionally seeds ``contours.author_username`` from ``added_by`` where that
     value happens to be a known username.

Usage (cwd = backend/):
    backend/.venv/Scripts/python.exe scripts/migrate_roles.py [--dry-run]
                                                              [--shared-role curator]
                                                              [--seed-authors]
"""
import argparse
import logging
import os
import sys

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from sqlalchemy import inspect, text  # noqa: E402

from app.database import engine, init_db  # noqa: E402
from app.schemas.permissions import DatasetRole, GlobalRole  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("migrate_roles")

#: column additions, as (table, column, DDL type + default)
_NEW_COLUMNS = [
    ("users", "global_role", f"VARCHAR(20) NOT NULL DEFAULT '{GlobalRole.MEMBER.value}'"),
    ("users", "is_active", "BOOLEAN NOT NULL DEFAULT 1"),
    ("datasets", "require_independent_review", "BOOLEAN NOT NULL DEFAULT 0"),
    ("contours", "author_username", "VARCHAR"),
    ("contours", "created_at", "DATETIME"),
]


def _existing_columns(connection, table: str) -> set[str]:
    inspector = inspect(connection)
    if table not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table)}


def add_missing_columns(connection, dry_run: bool) -> int:
    added = 0
    for table, column, ddl in _NEW_COLUMNS:
        columns = _existing_columns(connection, table)
        if not columns:
            logger.info("Table %s does not exist yet; create_all will make it.", table)
            continue
        if column in columns:
            continue
        logger.info("Adding %s.%s", table, column)
        if not dry_run:
            connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
        added += 1
    return added


def backfill_global_roles(connection, dry_run: bool) -> int:
    """Turn the old boolean admin flag into a global role."""
    if "is_admin" not in _existing_columns(connection, "users"):
        logger.info("No legacy users.is_admin column; nothing to backfill.")
        return 0
    admins = connection.execute(
        text("SELECT COUNT(*) FROM users WHERE is_admin = 1")
    ).scalar() or 0
    logger.info("Promoting %s legacy admin account(s) to global_role='admin'.", admins)
    if not dry_run and admins:
        connection.execute(text(
            "UPDATE users SET global_role = :role WHERE is_admin = 1"
        ), {"role": GlobalRole.ADMIN.value})
    return admins


def drop_legacy_is_admin(connection, dry_run: bool) -> int:
    """Remove ``users.is_admin`` once ``global_role`` carries the same information.

    The column is declared ``BOOLEAN NOT NULL`` with no default, and ``Users`` no longer
    maps it. Any INSERT the ORM builds therefore omits it and SQLite rejects the row with
    ``NOT NULL constraint failed: users.is_admin`` — which surfaces as a 500 from
    ``POST /auth/register``. Must run *after* backfill_global_roles().
    """
    if "is_admin" not in _existing_columns(connection, "users"):
        return 0
    logger.info("Dropping legacy users.is_admin (superseded by global_role).")
    if not dry_run:
        connection.execute(text("ALTER TABLE users DROP COLUMN is_admin"))
    return 1


def migrate_shares(connection, shared_role: DatasetRole, dry_run: bool) -> int:
    """Copy the old flat share table into role-carrying membership rows."""
    inspector = inspect(connection)
    if "dataset_user_association" not in inspector.get_table_names():
        logger.info("No legacy dataset_user_association table; nothing to migrate.")
        return 0

    rows = connection.execute(text(
        "SELECT dataset_id, user_name FROM dataset_user_association"
    )).fetchall()
    migrated = 0
    for dataset_id, username in rows:
        already = connection.execute(text(
            "SELECT 1 FROM dataset_members WHERE dataset_id = :d AND username = :u"
        ), {"d": dataset_id, "u": username}).scalar()
        if already:
            continue
        logger.info("Granting %s the %s role on dataset %s.", username, shared_role.value, dataset_id)
        if not dry_run:
            connection.execute(text(
                "INSERT INTO dataset_members "
                "(dataset_id, username, role, extra_permissions, denied_permissions, granted_at) "
                "VALUES (:d, :u, :r, '[]', '[]', CURRENT_TIMESTAMP)"
            ), {"d": dataset_id, "u": username, "r": shared_role.value})
        migrated += 1
    return migrated


def create_owner_memberships(connection, dry_run: bool) -> int:
    """Give each dataset creator an explicit owner row so ownership can transfer."""
    rows = connection.execute(text("SELECT id, created_by FROM datasets")).fetchall()
    created = 0
    for dataset_id, created_by in rows:
        if created_by is None:
            logger.warning("Dataset %s has no creator; skipping owner row.", dataset_id)
            continue
        existing = connection.execute(text(
            "SELECT role FROM dataset_members WHERE dataset_id = :d AND username = :u"
        ), {"d": dataset_id, "u": created_by}).scalar()
        if existing == DatasetRole.OWNER.value:
            continue
        logger.info("Making %s the owner of dataset %s.", created_by, dataset_id)
        if not dry_run:
            if existing is None:
                connection.execute(text(
                    "INSERT INTO dataset_members "
                    "(dataset_id, username, role, extra_permissions, denied_permissions, granted_at) "
                    "VALUES (:d, :u, :r, '[]', '[]', CURRENT_TIMESTAMP)"
                ), {"d": dataset_id, "u": created_by, "r": DatasetRole.OWNER.value})
            else:
                # The creator was also listed as a share; promote rather than duplicate.
                connection.execute(text(
                    "UPDATE dataset_members SET role = :r "
                    "WHERE dataset_id = :d AND username = :u"
                ), {"d": dataset_id, "u": created_by, "r": DatasetRole.OWNER.value})
        created += 1
    return created


def seed_contour_authors(connection, dry_run: bool) -> int:
    """Best-effort backfill of contours.author_username from added_by.

    `added_by` records what produced the geometry ("SAM2", "manual", sometimes a
    username), so only values matching an actual account are usable. Everything
    else is left NULL, which the permission code treats as "author unknown" and
    therefore editable by anyone who may edit the dataset's annotations.
    """
    if "author_username" not in _existing_columns(connection, "contours"):
        # Only reachable under --dry-run, where add_missing_columns() reported the
        # column instead of creating it. Counting rows would fail on "no such column".
        logger.info("contours.author_username does not exist yet; "
                    "re-run without --dry-run to add it and seed authors.")
        return 0

    updated = connection.execute(text(
        "SELECT COUNT(*) FROM contours WHERE author_username IS NULL "
        "AND added_by IN (SELECT username FROM users)"
    )).scalar() or 0
    logger.info("Seeding author_username for %s contour(s) whose added_by is a known user.", updated)
    if not dry_run and updated:
        connection.execute(text(
            "UPDATE contours SET author_username = added_by "
            "WHERE author_username IS NULL AND added_by IN (SELECT username FROM users)"
        ))
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing anything.")
    parser.add_argument("--shared-role", default=DatasetRole.CURATOR.value,
                        choices=[role.value for role in DatasetRole if role is not DatasetRole.OWNER],
                        help="Role to give users who previously had a dataset shared with them. "
                             "Defaults to curator, which is closest to the unrestricted access "
                             "sharing used to grant.")
    parser.add_argument("--seed-authors", action="store_true",
                        help="Also copy contours.added_by into author_username where it names a real user.")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN - no changes will be written.")

    # Create any tables that do not exist yet (dataset_members, dataset_invites,
    # annotation_rejections) before touching them.
    init_db()

    with engine.begin() as connection:
        added = add_missing_columns(connection, args.dry_run)
        promoted = backfill_global_roles(connection, args.dry_run)
        dropped = drop_legacy_is_admin(connection, args.dry_run)
        shares = migrate_shares(connection, DatasetRole(args.shared_role), args.dry_run)
        owners = create_owner_memberships(connection, args.dry_run)
        authors = seed_contour_authors(connection, args.dry_run) if args.seed_authors else 0
        if args.dry_run:
            connection.rollback()

    logger.info(
        "Done. columns_added=%s admins_promoted=%s is_admin_dropped=%s shares_migrated=%s "
        "owner_rows=%s authors_seeded=%s",
        added, promoted, dropped, shares, owners, authors,
    )
    logger.info(
        "The legacy dataset_user_association table is left in place; drop it once you "
        "have verified the migrated memberships."
    )


if __name__ == "__main__":
    main()

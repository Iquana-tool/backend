"""Migrate an existing database to the generalised image-calibration model.

The project has no Alembic setup — schema comes from ``metadata.create_all``, which
creates missing *tables* but never backfills them. ``image_calibrations`` is a new
table, so ``init_db()`` makes it; this script does the data part, and is safe to
re-run.

What it does:
  1. Creates ``image_calibrations`` (via ``init_db``) if it does not exist yet.
  2. Backfills a ``scale`` calibration row for every image that already carries a
     real-world scale in ``images.unit`` / ``scale_x`` / ``scale_y``. Those columns
     remain the value — the row records that a scale exists and where it came
     from, which is what the Calibrate tab shows. Images still on the ``px``
     default are correctly left with no row: they are uncalibrated, not
     calibrated at 1:1.
  3. Reports (and with ``--prune-orphans``, deletes) rows whose kind is no longer
     registered, which is how a kind that gets renamed or dropped is cleaned up.

Nothing is destructive without ``--prune-orphans``. Existing rows are never
overwritten, so re-running after real calibration work does not reset provenance
to "migrated".

Usage (cwd = backend/):
    backend/.venv/Scripts/python.exe scripts/migrate_calibrations.py [--dry-run]
                                                                    [--prune-orphans]
"""
import argparse
import json
import logging
import os
import sys

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from sqlalchemy import bindparam, inspect, text  # noqa: E402

from app.database import engine, init_db  # noqa: E402
from app.services.calibration import registry  # noqa: E402
from app.services.calibration.registry import CalibrationSource  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("migrate_calibrations")


def _table_exists(connection, table: str) -> bool:
    return table in inspect(connection).get_table_names()


def backfill_scale_rows(connection, dry_run: bool) -> int:
    """Give every already-scaled image a ``scale`` calibration row.

    ``unit = 'px'`` is the column default and means "no physical scale set", so it
    is excluded — writing a row for those would turn every untouched image into a
    calibrated one. Non-positive scale values are excluded for the same reason:
    they cannot describe a real pixel size.
    """
    rows = connection.execute(text(
        "SELECT i.id, i.scale_x, i.scale_y, i.unit "
        "FROM images i "
        "LEFT JOIN image_calibrations c "
        "  ON c.image_id = i.id AND c.kind = 'scale' "
        "WHERE c.id IS NULL "
        "  AND i.unit IS NOT NULL AND i.unit != 'px' AND TRIM(i.unit) != '' "
        "  AND i.scale_x > 0 AND i.scale_y > 0"
    )).fetchall()

    if not rows:
        logger.info("No images need a scale calibration row.")
        return 0

    logger.info("Backfilling scale calibration rows for %d image(s).", len(rows))
    if dry_run:
        return len(rows)

    for image_id, scale_x, scale_y, unit in rows:
        params = {"scale_x": float(scale_x), "scale_y": float(scale_y), "unit": str(unit).strip()}
        connection.execute(text(
            "INSERT INTO image_calibrations "
            "(image_id, kind, params, source, created_by, created_at, updated_at) "
            "VALUES (:image_id, 'scale', :params, :source, NULL, "
            "        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ), {
            "image_id": image_id,
            "params": json.dumps(params),
            # The scale predates provenance tracking, so the honest answer to
            # "where did this come from" is only "someone set it by hand".
            "source": CalibrationSource.MANUAL,
        })
    return len(rows)


def report_orphan_kinds(connection, prune: bool, dry_run: bool) -> int:
    """Find rows whose ``kind`` is no longer in the registry.

    Such a row is inert — the service iterates registered kinds, so it is never
    read or applied — but it is also invisible, which is worse than being noisy.

    ``LEGACY_KINDS`` are excluded from pruning even though they are unregistered:
    they are real calibrations awaiting conversion by
    ``scripts/migrate_response_calibration.py``, and deleting them here would
    destroy measurements rather than tidy up after a rename.
    """
    known = {kind.key for kind in registry.all_kinds()} | set(registry.LEGACY_KINDS)
    rows = connection.execute(text(
        "SELECT kind, COUNT(*) FROM image_calibrations GROUP BY kind"
    )).fetchall()

    for kind, count in rows:
        if kind in registry.LEGACY_KINDS:
            logger.warning(
                "%d row(s) still use the superseded calibration kind %r. Convert them "
                "with scripts/migrate_response_calibration.py; they are not pruned here.",
                count, kind)

    orphans = [(kind, count) for kind, count in rows if kind not in known]
    if not orphans:
        return 0

    for kind, count in orphans:
        logger.warning("%d row(s) use unregistered calibration kind %r.", count, kind)
    if not prune:
        logger.warning("Re-run with --prune-orphans to delete them.")
        return 0

    total = sum(count for _, count in orphans)
    logger.info("Deleting %d orphaned calibration row(s).", total)
    if not dry_run:
        # `expanding` turns the single bind into one placeholder per element, which
        # is what a NOT IN over a Python list needs.
        connection.execute(
            text("DELETE FROM image_calibrations WHERE kind NOT IN :known")
            .bindparams(bindparam("known", expanding=True)),
            {"known": sorted(known)},
        )
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing anything.")
    parser.add_argument("--prune-orphans", action="store_true",
                        help="Delete calibration rows whose kind is no longer registered.")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN - no changes will be written.")

    logger.info("Registered calibration kinds: %s",
                ", ".join(kind.key for kind in registry.all_kinds()))

    # Creates image_calibrations if this database predates it.
    init_db()

    with engine.begin() as connection:
        if not _table_exists(connection, "image_calibrations"):
            logger.error("image_calibrations still does not exist after init_db(); aborting.")
            sys.exit(1)
        backfilled = backfill_scale_rows(connection, args.dry_run)
        pruned = report_orphan_kinds(connection, args.prune_orphans, args.dry_run)
        if args.dry_run:
            connection.rollback()

    logger.info("Done. scale_rows_backfilled=%s orphan_rows_pruned=%s", backfilled, pruned)


if __name__ == "__main__":
    main()

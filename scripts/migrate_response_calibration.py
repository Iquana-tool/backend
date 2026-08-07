"""Merge the separate ``intensity`` and ``color`` calibrations into ``response``.

Tone and colour started as two calibration kinds applied in sequence. They are now
one kind estimated by a chosen strategy, because they were always two estimates of
the same thing — how a camera under some light turned reference values into pixel
values — and keeping them apart meant they could be set inconsistently or stacked
on top of a card-based calibration that already accounted for both.

This script converts existing rows. The resulting ``response`` row uses the
``two_patch`` strategy, which is exactly what the old pair was, so the corrected
pixels are unchanged for every calibration that used the default gamma of 1.0.

Mapping:
  * intensity + colour  -> black/white/gamma from one, gains from the other
  * intensity alone     -> the same, with unit gains (tone-only, no cast correction)
  * colour alone        -> black 0 / white 255 with the gains, which is what a
                           gain-only correction always was
  * colour in matrix mode -> not convertible to per-channel anchors; reported and
                           left in place for a human to decide on

Old rows are kept unless ``--drop-legacy`` is passed, so a converted database can
still be inspected against what it came from. Run this BEFORE
``migrate_calibrations.py --prune-orphans`` — that script now refuses to prune
these kinds, but the ordering is worth keeping in mind.

Usage (cwd = backend/):
    backend/.venv/Scripts/python.exe scripts/migrate_response_calibration.py [--dry-run]
                                                                            [--drop-legacy]
"""
import argparse
import json
import logging
import os
import sys

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from sqlalchemy import text  # noqa: E402

from app.database import engine, init_db  # noqa: E402
from app.exceptions import InvalidCalibrationError  # noqa: E402
from app.services.calibration import registry  # noqa: E402
from app.services.calibration.registry import CalibrationSource  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("migrate_response_calibration")

#: Ranked weakest to strongest. When the two legacy rows disagree about how the
#: calibration was obtained, the merged row keeps the weaker claim — "measured"
#: should not be inherited by a value that was half typed in.
_SOURCE_RANK = {
    CalibrationSource.DATASET: 0,
    CalibrationSource.MANUAL: 1,
    CalibrationSource.FILE_METADATA: 2,
    CalibrationSource.MEASURED: 3,
}


def _load_params(raw) -> dict:
    """Params come back as a dict on SQLite/JSON columns, or a string if stored raw."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def _merge_params(intensity: dict | None, color: dict | None) -> dict:
    """Build the ``response`` parameters the two legacy rows are equivalent to."""
    merged: dict = {"strategy": "two_patch"}

    if intensity:
        merged["black_level"] = intensity.get("black_level")
        merged["white_level"] = intensity.get("white_level")
        merged["gamma"] = intensity.get("gamma", 1.0)
        for key in ("black_observed_rgb", "white_observed_rgb"):
            if intensity.get(key) is not None:
                merged[key] = intensity[key]
    else:
        # A gain-only correction is a two-patch calibration whose references were
        # the ends of the range: nothing is rescaled, only balanced.
        merged["black_level"] = 0.0
        merged["white_level"] = 255.0
        merged["gamma"] = 1.0

    if color:
        merged["gains"] = color.get("gains")
        if color.get("observed_rgb") is not None:
            # The old colour patch was sampled with the intensity stage already
            # applied, so it is not a raw neutral reading. Kept as provenance under
            # its own name rather than as `neutral_rgb`, which the estimator would
            # re-derive gains from and get a different answer.
            merged["legacy_observed_rgb"] = color["observed_rgb"]
    else:
        merged["gains"] = [1.0, 1.0, 1.0]

    return merged


def _pick_source(rows: list[dict]) -> str:
    sources = [row["source"] for row in rows if row.get("source")]
    if not sources:
        return CalibrationSource.MANUAL
    return min(sources, key=lambda value: _SOURCE_RANK.get(value, 1))


def collect_legacy(connection) -> dict[int, dict[str, dict]]:
    """Group every legacy calibration row by image."""
    rows = connection.execute(text(
        "SELECT image_id, kind, params, source, created_by "
        "FROM image_calibrations WHERE kind IN ('intensity', 'color')"
    )).fetchall()

    by_image: dict[int, dict[str, dict]] = {}
    for image_id, kind, params, source, created_by in rows:
        by_image.setdefault(image_id, {})[kind] = {
            "params": _load_params(params),
            "source": source,
            "created_by": created_by,
        }
    return by_image


def convert(connection, dry_run: bool) -> tuple[int, int]:
    """Write a merged ``response`` row per image that has legacy calibrations."""
    by_image = collect_legacy(connection)
    if not by_image:
        logger.info("No legacy intensity/color calibrations to convert.")
        return 0, 0

    logger.info("Found legacy calibrations on %d image(s).", len(by_image))
    kind = registry.get_kind("response")
    converted = 0
    skipped = 0

    for image_id, rows in sorted(by_image.items()):
        existing = connection.execute(text(
            "SELECT id FROM image_calibrations WHERE image_id = :i AND kind = 'response'"
        ), {"i": image_id}).scalar()
        if existing:
            logger.info("Image %s already has a response calibration; leaving it alone.",
                        image_id)
            skipped += 1
            continue

        color = rows.get("color", {}).get("params")
        if color and color.get("mode") == "matrix":
            logger.warning(
                "Image %s has a colour calibration in matrix mode, which has no "
                "per-channel equivalent. Left in place for manual conversion.", image_id)
            skipped += 1
            continue

        merged = _merge_params(rows.get("intensity", {}).get("params"), color)
        try:
            # Run it through the real validator so a converted row is exactly as
            # trustworthy as a freshly written one, anchors and all.
            normalized = kind.normalize(merged)
        except InvalidCalibrationError as exc:
            logger.warning("Image %s: cannot convert (%s). Left in place.", image_id, exc)
            skipped += 1
            continue

        source = _pick_source(list(rows.values()))
        created_by = next(
            (row["created_by"] for row in rows.values() if row.get("created_by")), None
        )
        logger.info("Image %s -> response (%s)", image_id, kind.describe(normalized))

        if not dry_run:
            connection.execute(text(
                "INSERT INTO image_calibrations "
                "(image_id, kind, params, source, created_by, created_at, updated_at) "
                "VALUES (:image_id, 'response', :params, :source, :created_by, "
                "        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ), {
                "image_id": image_id,
                "params": json.dumps(normalized),
                "source": source,
                "created_by": created_by,
            })
        converted += 1

    return converted, skipped


def drop_legacy_rows(connection, dry_run: bool) -> int:
    """Remove the old rows once their images carry a response calibration."""
    rows = connection.execute(text(
        "SELECT COUNT(*) FROM image_calibrations c "
        "WHERE c.kind IN ('intensity', 'color') "
        "  AND EXISTS (SELECT 1 FROM image_calibrations r "
        "              WHERE r.image_id = c.image_id AND r.kind = 'response')"
    )).scalar() or 0
    if not rows:
        logger.info("No converted legacy rows to drop.")
        return 0

    logger.info("Dropping %d converted legacy row(s).", rows)
    if not dry_run:
        connection.execute(text(
            "DELETE FROM image_calibrations "
            "WHERE kind IN ('intensity', 'color') "
            "  AND EXISTS (SELECT 1 FROM image_calibrations r "
            "              WHERE r.image_id = image_calibrations.image_id "
            "                AND r.kind = 'response')"
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing anything.")
    parser.add_argument("--drop-legacy", action="store_true",
                        help="Delete the old intensity/color rows once converted. "
                             "Omit on the first run so the two can be compared.")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN - no changes will be written.")

    # Creates dataset_calibration_defaults if this database predates it.
    init_db()

    with engine.begin() as connection:
        converted, skipped = convert(connection, args.dry_run)
        dropped = drop_legacy_rows(connection, args.dry_run) if args.drop_legacy else 0
        if args.dry_run:
            connection.rollback()

    logger.info("Done. converted=%s skipped=%s legacy_rows_dropped=%s",
                converted, skipped, dropped)
    if converted and not args.drop_legacy:
        logger.info("Legacy rows were kept. Re-run with --drop-legacy once you have "
                    "checked the converted values.")


if __name__ == "__main__":
    main()

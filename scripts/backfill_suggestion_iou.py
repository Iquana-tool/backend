"""Fill ``ai_suggestions.final_iou`` for suggestions edited before revision 0011.

For every suggestion that was edited by hand but has no IoU yet, compare the logged
outline with the object's current outline and store the result. Suggestions whose
object has since been deleted are skipped: the final outline is gone, so there is
nothing to compare against (they count as rejected anyway).

Idempotent (only touches rows where ``final_iou`` is NULL), commits in batches and
prints a summary.

Usage (cwd = backend/):
    .venv/bin/python scripts/backfill_suggestion_iou.py [--dataset-id N] [--dry-run]
"""
import argparse
import logging
import os
import sys

# Make ``import app...`` and ``import config`` work regardless of where python is invoked
# from: put the backend project root (parent of this scripts/ dir) on sys.path.
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from app.database import get_context_session  # noqa: E402
from app.database.ai_suggestions import AiSuggestions  # noqa: E402
from app.database.contours import Contours  # noqa: E402
from app.services.provenance import outline_iou  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_suggestion_iou")

BATCH_SIZE = 500


def backfill(dataset_id: int | None, dry_run: bool) -> None:
    updated = skipped = 0
    with get_context_session() as session:
        # Collect the ids up front and work through them in chunks: committing while
        # a streamed result is still open would invalidate its cursor on PostgreSQL.
        query = (session.query(AiSuggestions.id)
                 .join(Contours, Contours.id == AiSuggestions.contour_id)
                 .filter(AiSuggestions.edited_at.is_not(None),
                         AiSuggestions.final_iou.is_(None)))
        if dataset_id is not None:
            query = query.filter(AiSuggestions.dataset_id == dataset_id)
        ids = [row.id for row in query.order_by(AiSuggestions.id)]

        for start in range(0, len(ids), BATCH_SIZE):
            chunk = ids[start:start + BATCH_SIZE]
            rows = (session.query(AiSuggestions, Contours)
                    .join(Contours, Contours.id == AiSuggestions.contour_id)
                    .filter(AiSuggestions.id.in_(chunk)))
            for suggestion, contour in rows:
                iou = outline_iou(list(suggestion.x or []), list(suggestion.y or []),
                                  list(contour.x or []), list(contour.y or []))
                if iou is None:
                    skipped += 1
                    continue
                suggestion.final_iou = iou
                updated += 1
            if dry_run:
                session.rollback()
            else:
                session.commit()
            logger.info("Processed %d of %d suggestion(s).", min(start + BATCH_SIZE, len(ids)),
                        len(ids))

    logger.info("%s %d suggestion(s); skipped %d without a usable outline.",
                "Would update" if dry_run else "Updated", updated, skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill ai_suggestions.final_iou.")
    parser.add_argument("--dataset-id", type=int, default=None,
                        help="Only backfill suggestions of this dataset. Default: all datasets.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and report, but do not write.")
    args = parser.parse_args()
    backfill(args.dataset_id, args.dry_run)


if __name__ == "__main__":
    main()

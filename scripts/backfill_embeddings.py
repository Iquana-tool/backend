"""Backfill the embedding store for existing images and contours.

Embeds every image / contour (optionally scoped to one dataset) that has no embedding of a
required kind yet, by calling the ai-service ``embed`` surface and upserting the results.
What gets computed is driven by the enabled retrieval strategies' ``required_kinds``
(``image_cls`` for images, ``region_mean`` for contours).

Run this to populate the store the first time, or after enabling a new retrieval strategy.
Idempotent -- already-embedded subjects are skipped, so re-running only fills gaps. Requires a
reachable ai-service embed surface (``EMBED_BACKEND_URL``) with the embedder model registered.

Usage (cwd = backend/):
    backend/.venv/Scripts/python.exe scripts/backfill_embeddings.py [--dataset-id N] [--dry-run]
"""
import argparse
import logging
import os
import sys

# Make ``import app...`` / ``import config`` work regardless of invocation cwd.
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from app.database import get_context_session  # noqa: E402
from app.services.embedding_lifecycle import (  # noqa: E402
    backfill_embeddings,
    required_kinds,
)
from config import EMBEDDING_MODEL_KEY  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_embeddings")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill embeddings for images and contours.")
    parser.add_argument("--dataset-id", type=int, default=None,
                        help="Restrict to one dataset (default: all datasets).")
    parser.add_argument("--model-key", type=str, default=EMBEDDING_MODEL_KEY,
                        help=f"Embedder registry key to call (default: {EMBEDDING_MODEL_KEY}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be embedded without calling the service or writing.")
    args = parser.parse_args()

    kinds = required_kinds()
    if not kinds:
        logger.warning("No enabled retrieval strategy requires any embedding kind; nothing to do.")
        return
    logger.info("Required embedding kinds: %s", sorted(kinds))

    with get_context_session() as session:
        if args.dry_run:
            # Reuse the internal selectors to count without embedding.
            from app.services.embedding_lifecycle import (
                _contours_missing_kind,
                _images_missing_kind,
                _region_kinds_needed,
                _required_image_kinds,
            )
            n_images = len(_images_missing_kind(session, args.dataset_id)) if _required_image_kinds() else 0
            n_contours = len(_contours_missing_kind(session, args.dataset_id)) if _region_kinds_needed() else 0
            logger.info("[dry-run] would embed %d image(s) and %d contour(s).", n_images, n_contours)
            return

        counts = backfill_embeddings(
            session, dataset_id=args.dataset_id, model_registry_key=args.model_key
        )
        session.commit()
        logger.info("Backfilled %d image(s) and %d contour(s).", counts["images"], counts["contours"])


if __name__ == "__main__":
    main()

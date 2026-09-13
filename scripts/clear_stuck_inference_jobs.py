"""Finalise inference jobs that no worker is going to finish.

A job row says `pending` / `running` / `cancelling` only while a worker is expected to move
it along. If the message was lost -- broker down at submit time, worker not running, or a
task published to a queue nobody consumes -- the row stays that way forever, and because
those statuses are not terminal it also blocks every future run on the dataset.

The UI can do this itself now (cancel finalises an unclaimed run, and "Remove from history"
deletes it), so this script is the way out for rows that predate that fix, or for clearing a
whole instance at once.

Annotations already written by the job are never touched: a cancelled run is a partial run,
not an undone one.

Usage:
    uv run python scripts/clear_stuck_inference_jobs.py            # list what is stuck
    uv run python scripts/clear_stuck_inference_jobs.py --apply    # finalise it
    uv run python scripts/clear_stuck_inference_jobs.py --apply --dataset-id 3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import get_context_session, init_db  # noqa: E402
from app.database.inference_jobs import (  # noqa: E402
    InferenceJobItems,
    InferenceJobs,
    TERMINAL_JOB_STATUSES,
)
from app.services.inference.tasks import abandon_pending, finish  # noqa: E402

_STUCK_MESSAGE = (
    "Cancelled by scripts/clear_stuck_inference_jobs.py: no worker was running to finish it."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Actually finalise the jobs (default is a dry run).")
    parser.add_argument("--dataset-id", type=int, default=None,
                        help="Limit to one dataset.")
    args = parser.parse_args()

    init_db()
    with get_context_session() as db:
        query = db.query(InferenceJobs).filter(
            InferenceJobs.status.notin_(tuple(TERMINAL_JOB_STATUSES))
        )
        if args.dataset_id is not None:
            query = query.filter(InferenceJobs.dataset_id == args.dataset_id)
        jobs = query.order_by(InferenceJobs.id).all()

        if not jobs:
            print("No unfinished inference jobs.")
            return 0

        for job in jobs:
            done = db.query(InferenceJobItems).filter(
                InferenceJobItems.job_id == job.id,
                InferenceJobItems.status.in_(("done", "failed")),
            ).count()
            print(f"job {job.id}  dataset {job.dataset_id}  status={job.status}  "
                  f"{done}/{job.total_units} units  created={job.contours_created} contours")

        if not args.apply:
            print(f"\n{len(jobs)} job(s) would be cancelled. Re-run with --apply.")
            return 0

        for job in jobs:
            abandon_pending(db, job.id)
            finish(db, job, "cancelled", _STUCK_MESSAGE)
        print(f"\nCancelled {len(jobs)} job(s). The contours they wrote were kept.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

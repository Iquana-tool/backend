"""Batch inference: annotate a whole dataset with a hand-picked orchestration of models.

The counterpart to the annotation canvas, where a model runs on the one image in front of
you. Here a user binds each label to a model, picks how the predictions meet the annotations
that already exist (patch or replace), and hands the whole dataset to Celery.

Everything is gated on `AI_BATCH_INFER`. Starting a *replace* run additionally needs
`MASK_DELETE`: it deletes existing contours and their child objects before predicting, which
is a destructive act on annotations, not an AI capability -- and `confirm_replace` must be
set explicitly, so no default-valued field can ever trigger one.
"""
import asyncio
from logging import getLogger

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from starlette import status as http_status

from app.database import get_session
from app.database.inference_jobs import InferenceJobItems, InferenceJobs, TERMINAL_JOB_STATUSES
from app.schemas.auth_user import AuthenticatedUser
from app.schemas.inference import (
    DatasetModelRoutingRead,
    DatasetModelRoutingWrite,
    InferenceJobCreate,
    InferenceJobItemRead,
    InferenceJobSnapshot,
    ModelCatalog,
    ModelRoutingSuggestRequest,
    ModelRoutingSuggestResult,
    ReplacePreview,
    ScopeCounts,
    WriteMode,
)
from app.schemas.permissions import Permission
from app.services.auth import get_current_user
from app.services.celery_app import BACKEND_QUEUE, celery_app
from app.services.inference import configuration, planning, progress, tasks
from app.services.permissions import ensure_permission, require

logger = getLogger(__name__)
router = APIRouter(prefix="/inference", tags=["batch_inference"])

#: Seconds between progress ticks on the SSE stream. Matches the training stream so the two
#: progress views feel the same; a unit takes far longer than this, so nothing is missed.
_STREAM_INTERVAL = 2.0


def _load_job(job_id: int, db: Session, user: AuthenticatedUser) -> InferenceJobs:
    """Fetch a job and authorize the caller against its dataset."""
    job = db.get(InferenceJobs, job_id)
    if job is None:
        raise HTTPException(http_status.HTTP_404_NOT_FOUND, f"Inference job {job_id} not found.")
    ensure_permission(user, job.dataset_id, Permission.AI_BATCH_INFER)
    return job


# --------------------------------------------------------------------------- #
# Planning surface
# --------------------------------------------------------------------------- #
@router.get("/models", response_model=ModelCatalog)
async def get_model_catalog(
    dataset_id: int,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Models a label in this dataset may be bound to, and the retrieval strategies."""
    if not (
        user.has_permission(dataset_id, Permission.AI_BATCH_INFER)
        or user.has_permission(dataset_id, Permission.AI_INTERACTIVE)
    ):
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail=f"Missing permission to view model catalog on dataset {dataset_id}.",
        )
    return await asyncio.to_thread(planning.model_catalog, db, dataset_id)


@router.get("/scope", response_model=ScopeCounts)
async def get_scope_counts(
    dataset_id: int,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(require(Permission.AI_BATCH_INFER)),
):
    """Image counts per selection, so the scope picker can label its options."""
    return planning.scope_counts(db, dataset_id)


@router.post("/replace-preview", response_model=ReplacePreview)
async def get_replace_preview(
    body: InferenceJobCreate,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Count what a replace run would delete, for the confirmation dialog.

    Takes the whole job body rather than a scope alone so the numbers shown are the numbers
    for the run that is about to start -- including how many contours `preserve_reviewed`
    would spare. Reads only; `confirm_replace` is irrelevant here.
    """
    ensure_permission(user, body.dataset_id, Permission.AI_BATCH_INFER)
    image_ids = planning.resolve_scope(db, body.dataset_id, body)
    return planning.replace_preview(
        db, image_ids, preserve_reviewed=body.options.preserve_reviewed
    )


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
@router.post("/jobs", response_model=InferenceJobSnapshot)
async def create_job(
    body: InferenceJobCreate,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Validate an orchestration, freeze it into a work list, and hand it to Celery."""
    ensure_permission(user, body.dataset_id, Permission.AI_BATCH_INFER)
    if body.options.write_mode == WriteMode.REPLACE:
        # Destroying annotations is not something AI assistance alone should authorize.
        ensure_permission(user, body.dataset_id, Permission.MASK_DELETE)

    job = planning.create_job(db, body.dataset_id, str(user.username), body)

    # Import here: the task module pulls in the whole execution path (AI clients, the
    # toolbox overlap helpers), which the API process has no reason to load until a job is
    # actually started.
    from app.services.inference.tasks import run_job

    try:
        task = run_job.apply_async((job.id,), queue=BACKEND_QUEUE)
    except Exception as exc:
        logger.exception("Could not enqueue inference job %s.", job.id)
        job.status = "failed"
        job.error = f"Could not reach the task broker: {exc}"
        db.commit()
        raise HTTPException(
            http_status.HTTP_503_SERVICE_UNAVAILABLE,
            "The job was created but could not be queued; is the Celery broker running?",
        )
    job.celery_task_id = task.id
    db.commit()
    return progress.snapshot(db, job)


@router.get("/jobs", response_model=list[InferenceJobSnapshot])
async def list_jobs(
    dataset_id: int,
    limit: int = Query(default=25, ge=1, le=100),
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(require(Permission.AI_BATCH_INFER)),
):
    """The dataset's run history, newest first."""
    jobs = (
        db.query(InferenceJobs)
        .filter(InferenceJobs.dataset_id == dataset_id)
        .order_by(InferenceJobs.id.desc())
        .limit(limit)
        .all()
    )
    return [progress.snapshot(db, job) for job in jobs]


@router.get("/jobs/{job_id}", response_model=InferenceJobSnapshot)
async def get_job(
    job_id: int,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """A single progress snapshot (the polling fallback for the stream below)."""
    return progress.snapshot(db, _load_job(job_id, db, user))


@router.get("/jobs/{job_id}/stream")
async def stream_job(
    job_id: int,
    request: Request,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Stream progress as Server-Sent Events until the run reaches a terminal state."""
    job = _load_job(job_id, db, user)

    async def event_generator():
        while True:
            if await request.is_disconnected():
                return
            db.expire_all()  # the worker writes from another connection
            current = db.get(InferenceJobs, job.id)
            if current is None:
                return
            payload = progress.snapshot(db, current)
            yield f"data: {payload.model_dump_json()}\n\n"
            if current.status in TERMINAL_JOB_STATUSES:
                return
            await asyncio.sleep(_STREAM_INTERVAL)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/jobs/{job_id}/items", response_model=list[InferenceJobItemRead])
async def get_job_items(
    job_id: int,
    item_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Work items of a run; the UI asks for `status=failed` to list what went wrong."""
    return progress.read_items(db, _load_job(job_id, db, user), status=item_status, limit=limit)


@router.post("/jobs/{job_id}/cancel", response_model=InferenceJobSnapshot)
async def cancel_job(
    job_id: int,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Ask a run to stop.

    For a run a worker has actually picked up, `cancelling` is a *request*: the worker checks
    the status between units and stops after the image it is on, rather than tearing a
    half-written mask apart. Annotations already written are kept -- a cancelled run is a
    partial run, not an undone one.

    Two cases must not wait for a worker, because no worker is ever going to come:

    * The run was never claimed (`started_at is None`) -- the broker was down, or no worker
      is running. There is nothing in flight to stop.
    * The run is already `cancelling` -- the user is asking a second time, which is the only
      signal available that the worker is gone.

    Both are finalised here and now. Without this a run could sit in `cancelling` forever,
    and since that status is not terminal it would block every future run on the dataset.
    """
    job = _load_job(job_id, db, user)
    if job.status in TERMINAL_JOB_STATUSES:
        return progress.snapshot(db, job)

    if job.celery_task_id:
        try:
            from celery.result import AsyncResult

            AsyncResult(job.celery_task_id, app=celery_app).revoke()
        except Exception:
            logger.warning("Could not revoke task %s for job %s.", job.celery_task_id, job_id)

    # ``run_job`` sets started_at before it publishes ``run_next``. If that follow-up
    # task is lost, a job can look started while every work item is still pending and
    # no worker will ever come back to finalize it. Treat that state like an unreachable
    # worker so the first cancel is enough to release the dataset.
    has_started_item = (
        db.query(InferenceJobItems.id)
        .filter(
            InferenceJobItems.job_id == job.id,
            InferenceJobItems.status.in_(("running", "done", "failed")),
        )
        .first()
        is not None
    )
    unreachable = job.started_at is None or job.status == "cancelling" or not has_started_item
    if unreachable:
        tasks.abandon_pending(db, job.id)
        tasks.finish(db, job, "cancelled")
    else:
        job.status = "cancelling"
        db.commit()
    return progress.snapshot(db, job)


@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: int,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Remove a run from the history. The annotations it wrote stay.

    Only a run a worker is actively walking is refused. A queued-but-never-claimed run, or one
    stuck asking to be cancelled, must always be deletable -- otherwise a broker that was down
    at submit time leaves a row that blocks the dataset forever with no way out of the UI.
    """
    job = _load_job(job_id, db, user)
    if job.status == "running":
        raise HTTPException(
            http_status.HTTP_409_CONFLICT,
            f"Job {job_id} is still running; stop it before deleting it.",
        )
    db.delete(job)
    db.commit()
    return {"success": True, "message": f"Deleted inference job {job_id}."}


# --------------------------------------------------------------------------- #
# Dataset Model Routing Policy
# --------------------------------------------------------------------------- #
@router.get("/config", response_model=Optional[DatasetModelRoutingRead])
def get_dataset_model_routing(
    dataset_id: int,
    response: Response = None,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Retrieve the single model routing policy for a dataset.

    Returns 204 No Content when no policy is configured.
    """
    if not (
        user.has_permission(dataset_id, Permission.AI_BATCH_INFER)
        or user.has_permission(dataset_id, Permission.AI_INTERACTIVE)
    ):
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail=f"Missing permission to view inference routing policy on dataset {dataset_id}.",
        )

    policy = configuration.get_routing_policy(db, dataset_id)
    if policy is None:
        if response is not None:
            response.status_code = http_status.HTTP_204_NO_CONTENT
        return None
    return policy


@router.put("/config", response_model=DatasetModelRoutingRead)
def update_dataset_model_routing(
    body: DatasetModelRoutingWrite,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Save or replace the model routing policy for a dataset."""
    ensure_permission(user, body.dataset_id, Permission.AI_BATCH_INFER)
    return configuration.upsert_routing_policy(db, body.dataset_id, user.username, body)


@router.delete("/config")
def delete_dataset_model_routing(
    dataset_id: int,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Explicitly delete the model routing policy for a dataset."""
    ensure_permission(user, dataset_id, Permission.AI_BATCH_INFER)
    deleted = configuration.clear_routing_policy(db, dataset_id)
    if not deleted:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"No inference routing policy found for dataset {dataset_id}.",
        )
    return {"success": True, "message": f"Deleted inference routing policy for dataset {dataset_id}."}


@router.post("/config/suggest", response_model=ModelRoutingSuggestResult)
def suggest_model_routing_step(
    body: ModelRoutingSuggestRequest,
    db: Session = Depends(get_session),
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Execute one routed model step for a single image with patch semantics."""
    ensure_permission(user, body.dataset_id, Permission.AI_INTERACTIVE)
    ensure_permission(user, body.dataset_id, Permission.ANNOTATION_CREATE)
    return configuration.execute_suggest_step(
        db=db,
        dataset_id=body.dataset_id,
        image_id=body.image_id,
        label_id=body.label_id,
        username=user.username,
        task=body.task,
        mask_id=body.mask_id,
    )

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.routes.services import instance_seg_router
from app.routes.services.instance_seg_router import (
    _CANONICAL_TO_PUBLIC_STATE,
    _TERMINAL_STATES,
    _empty_snapshot,
    _map_training_state_to_public_state,
    _parse_timestamp,
    _read_training_snapshot,
    _reconcile_run_with_celery,
    _snapshot_from_run,
)


def _mock_run(
    *,
    run_id="run-1",
    status="RUNNING",
    tags=None,
    params=None,
    metrics=None,
    start_time=1000,
    end_time=2000,
):
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id,
            status=status,
            start_time=start_time,
            end_time=end_time,
        ),
        data=SimpleNamespace(
            metrics=metrics or {},
            params=params or {"epochs": "5", "batch_size": "2"},
            tags={"celery_task_id": "task-1", **(tags or {})},
        ),
    )


def test_starting_tag_maps_to_public_starting_and_is_non_terminal():
    run = _mock_run(
        status="RUNNING",
        tags={"training_state": "starting", "queued_at": "1718000000.5"},
    )
    snapshot = _snapshot_from_run(None, run, "task-1")

    assert snapshot["state"] == "STARTING"
    assert snapshot["training_state"] == "starting"
    assert snapshot["state"] not in _TERMINAL_STATES
    assert snapshot["queued_at"] == 1718000000.5


def test_running_tag_maps_to_public_progress_and_is_non_terminal():
    run = _mock_run(
        status="RUNNING",
        tags={"training_state": "running", "started_at": "2026-08-18T10:00:00Z"},
    )
    snapshot = _snapshot_from_run(None, run, "task-1")

    assert snapshot["state"] == "PROGRESS"
    assert snapshot["training_state"] == "running"
    assert snapshot["state"] not in _TERMINAL_STATES
    assert snapshot["started_at"] == "2026-08-18T10:00:00Z"


def test_completed_tag_maps_to_public_success_and_is_terminal():
    run = _mock_run(
        status="FINISHED",
        tags={"training_state": "completed"},
    )
    snapshot = _snapshot_from_run(None, run, "task-1")

    assert snapshot["state"] == "SUCCESS"
    assert snapshot["training_state"] == "completed"
    assert snapshot["state"] in _TERMINAL_STATES


def test_failed_tag_maps_to_public_failed_and_preserves_message():
    run = _mock_run(
        status="FAILED",
        tags={
            "training_state": "failed",
            "status_message": "CUDA out of memory during batch 4",
        },
    )
    snapshot = _snapshot_from_run(None, run, "task-1")

    assert snapshot["state"] == "FAILED"
    assert snapshot["training_state"] == "failed"
    assert snapshot["state"] in _TERMINAL_STATES
    assert snapshot["message"] == "CUDA out of memory during batch 4"


def test_cancelled_tag_maps_to_public_cancelled_and_preserves_message():
    run = _mock_run(
        status="KILLED",
        tags={
            "training_state": "cancelled",
            "status_message": "Training cancelled by user.",
        },
    )
    snapshot = _snapshot_from_run(None, run, "task-1")

    assert snapshot["state"] == "CANCELLED"
    assert snapshot["training_state"] == "cancelled"
    assert snapshot["state"] in _TERMINAL_STATES
    assert snapshot["message"] == "Training cancelled by user."


def test_timed_out_tag_maps_to_public_timed_out_and_is_terminal():
    run = _mock_run(
        status="KILLED",
        tags={
            "training_state": "timed_out",
            "status_message": "Training did not start before the queue deadline.",
            "queued_at": "1718000000",
            "start_deadline": "1718000300",
        },
    )
    snapshot = _snapshot_from_run(None, run, "task-1")

    assert snapshot["state"] == "TIMED_OUT"
    assert snapshot["training_state"] == "timed_out"
    assert snapshot["state"] in _TERMINAL_STATES
    assert snapshot["message"] == "Training did not start before the queue deadline."
    assert snapshot["queued_at"] == 1718000000.0
    assert snapshot["start_deadline"] == 1718000300.0


def test_all_queue_metadata_fields_are_exposed_in_snapshot():
    run = _mock_run(
        status="RUNNING",
        tags={
            "training_state": "starting",
            "status_message": "Queued waiting for worker",
            "queued_at": "1718000100.25",
            "start_deadline": "1718000400.25",
            "started_at": "2026-08-18T10:05:00Z",
        },
    )
    snapshot = _snapshot_from_run(None, run, "task-1")

    expected_keys = {
        "task_id",
        "run_id",
        "state",
        "training_state",
        "message",
        "queued_at",
        "start_deadline",
        "started_at",
        "mlflow_status",
        "epoch",
        "total_epochs",
        "training_parameters",
        "loss",
        "validation_metrics",
        "validation_metrics_unavailable",
        "label_ids",
        "run_name",
        "start_time",
        "end_time",
    }
    assert set(snapshot.keys()) == expected_keys
    assert snapshot["queued_at"] == 1718000100.25
    assert snapshot["start_deadline"] == 1718000400.25
    assert snapshot["started_at"] == "2026-08-18T10:05:00Z"
    assert snapshot["message"] == "Queued waiting for worker"


def test_legacy_untagged_runs_use_fallback_mapping():
    running_run = _mock_run(status="RUNNING", tags={})
    finished_run = _mock_run(status="FINISHED", tags={})
    failed_run = _mock_run(status="FAILED", tags={})
    killed_run = _mock_run(status="KILLED", tags={})

    assert _snapshot_from_run(None, running_run)["state"] == "PROGRESS"
    assert _snapshot_from_run(None, finished_run)["state"] == "SUCCESS"
    assert _snapshot_from_run(None, failed_run)["state"] == "FAILED"
    # Legacy KILLED maps to CANCELLED, never inferring TIMED_OUT without tag
    assert _snapshot_from_run(None, killed_run)["state"] == "CANCELLED"


def test_ai_status_check_that_changes_tags_refetches_run(monkeypatch):
    initial_run = _mock_run(
        run_id="run-100",
        status="RUNNING",
        tags={"celery_task_id": "task-100", "training_state": "starting"},
    )
    refreshed_run = _mock_run(
        run_id="run-100",
        status="KILLED",
        tags={
            "celery_task_id": "task-100",
            "training_state": "timed_out",
            "status_message": "Training did not start before the queue deadline.",
        },
    )

    mock_get_status = AsyncMock(
        return_value={
            "task_id": "task-100",
            "state": "REVOKED",
            "training_state": "timed_out",
            "message": "Training did not start before the queue deadline.",
        }
    )
    monkeypatch.setattr(
        instance_seg_router.service, "get_training_task_status", mock_get_status
    )

    mock_client = MagicMock()
    mock_client.get_run.return_value = refreshed_run
    monkeypatch.setattr(instance_seg_router.MODEL_REGISTRY, "client", mock_client)

    result_run = asyncio.run(_reconcile_run_with_celery(initial_run))

    assert result_run.info.status == "KILLED"
    assert result_run.data.tags["training_state"] == "timed_out"
    snapshot = _snapshot_from_run(None, result_run)
    assert snapshot["state"] == "TIMED_OUT"
    assert snapshot["message"] == "Training did not start before the queue deadline."


def test_ai_status_unavailability_falls_back_without_false_terminal_result(
    monkeypatch,
):
    active_run = _mock_run(
        run_id="run-200",
        status="RUNNING",
        tags={"celery_task_id": "task-200", "training_state": "running"},
    )

    mock_get_status = AsyncMock(side_effect=Exception("AI Service connection timeout"))
    monkeypatch.setattr(
        instance_seg_router.service, "get_training_task_status", mock_get_status
    )

    result_run = asyncio.run(_reconcile_run_with_celery(active_run))

    assert result_run.info.status == "RUNNING"
    snapshot = _snapshot_from_run(None, result_run)
    assert snapshot["state"] == "PROGRESS"
    assert snapshot["state"] not in _TERMINAL_STATES


def test_sse_emits_one_terminal_snapshot_and_then_stops(monkeypatch):
    terminal_snapshot = {
        "task_id": "task-300",
        "run_id": "run-300",
        "state": "TIMED_OUT",
        "training_state": "timed_out",
        "message": "Training did not start before the queue deadline.",
    }

    mock_read_snapshot = AsyncMock(return_value=terminal_snapshot)
    monkeypatch.setattr(
        instance_seg_router, "_read_training_snapshot", mock_read_snapshot
    )

    # Test the stream generator logic directly
    async def run_stream():
        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)
        response = await instance_seg_router.get_training_status_stream(
            task_id="task-300", request=mock_request, user=None
        )
        events = []
        async for chunk in response.body_iterator:
            events.append(chunk)
        return events

    events = asyncio.run(run_stream())
    assert len(events) == 1
    assert "TIMED_OUT" in events[0]
    assert mock_read_snapshot.call_count == 1


def test_list_and_single_task_snapshots_have_the_same_lifecycle_fields():
    run = _mock_run(
        run_id="run-400",
        status="RUNNING",
        tags={
            "celery_task_id": "task-400",
            "training_state": "starting",
            "queued_at": "1718000000",
            "start_deadline": "1718000300",
        },
    )

    run_snapshot = _snapshot_from_run(None, run, "task-400")
    empty_snapshot = _empty_snapshot(
        "task-400",
        {
            "training_state": "starting",
            "queued_at": 1718000000.0,
            "start_deadline": 1718000300.0,
            "state": "PENDING",
        },
    )

    assert set(run_snapshot.keys()) == set(empty_snapshot.keys())
    assert empty_snapshot["state"] == "STARTING"
    assert empty_snapshot["training_state"] == "starting"
    assert empty_snapshot["queued_at"] == 1718000000.0
    assert empty_snapshot["start_deadline"] == 1718000300.0


def test_reconciliation_revoked_task_with_stale_running_tag_returns_cancelled(
    monkeypatch,
):
    initial_run = _mock_run(
        run_id="run-500",
        status="RUNNING",
        tags={"celery_task_id": "task-500", "training_state": "running"},
    )
    terminated_run = _mock_run(
        run_id="run-500",
        status="KILLED",
        tags={"celery_task_id": "task-500", "training_state": "cancelled"},
    )

    mock_get_status = AsyncMock(
        return_value={"task_id": "task-500", "state": "REVOKED"}
    )
    monkeypatch.setattr(
        instance_seg_router.service, "get_training_task_status", mock_get_status
    )

    mock_client = MagicMock()
    mock_client.get_run.side_effect = [initial_run, terminated_run]
    monkeypatch.setattr(instance_seg_router.MODEL_REGISTRY, "client", mock_client)

    result_run = asyncio.run(_reconcile_run_with_celery(initial_run))

    mock_client.set_tag.assert_called_with("run-500", "training_state", "cancelled")
    mock_client.set_terminated.assert_called_with("run-500", status="KILLED")
    snapshot = _snapshot_from_run(None, result_run)
    assert snapshot["state"] == "CANCELLED"
    assert snapshot["state"] in _TERMINAL_STATES



def test_terminal_mlflow_status_overrides_stale_running_tag():
    run_killed = _mock_run(
        status="KILLED",
        tags={"training_state": "running"},
    )
    run_failed = _mock_run(
        status="FAILED",
        tags={"training_state": "running"},
    )
    run_finished = _mock_run(
        status="FINISHED",
        tags={"training_state": "starting"},
    )

    assert _snapshot_from_run(None, run_killed)["state"] == "CANCELLED"
    assert _snapshot_from_run(None, run_failed)["state"] == "FAILED"
    assert _snapshot_from_run(None, run_finished)["state"] == "SUCCESS"


def test_cancel_training_error_does_not_mark_run_cancelled_locally(monkeypatch):
    active_run = _mock_run(
        run_id="run-600",
        status="RUNNING",
        tags={"celery_task_id": "task-600", "training_state": "running"},
    )

    mock_cancel = AsyncMock(side_effect=Exception("AI worker connection error"))
    monkeypatch.setattr(instance_seg_router.service, "cancel_training", mock_cancel)

    mock_find_run = MagicMock(return_value=active_run)
    monkeypatch.setattr(instance_seg_router, "_find_training_run", mock_find_run)

    mock_client = MagicMock()
    monkeypatch.setattr(instance_seg_router.MODEL_REGISTRY, "client", mock_client)

    with pytest.raises(instance_seg_router.HTTPException) as exc_info:
        asyncio.run(
            instance_seg_router.cancel_training_of_model(
                task_id="task-600", user=None
            )
        )

    assert exc_info.value.status_code == 502
    assert "Could not cancel training" in exc_info.value.detail
    mock_client.set_terminated.assert_not_called()


def test_reconcile_run_with_celery_skips_non_running_historical_runs(monkeypatch):
    finished_run = _mock_run(
        run_id="run-700",
        status="FINISHED",
        tags={"celery_task_id": "task-700", "training_state": "completed"},
    )

    mock_get_status = AsyncMock()
    monkeypatch.setattr(
        instance_seg_router.service, "get_training_task_status", mock_get_status
    )

    result_run = asyncio.run(_reconcile_run_with_celery(finished_run))

    assert result_run == finished_run
    mock_get_status.assert_not_called()

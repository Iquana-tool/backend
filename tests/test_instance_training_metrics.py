from types import SimpleNamespace

from app.routes.services.instance_seg_router import (
    _snapshot_from_run,
    _validation_metrics_from_run,
)


def _run(*, metrics, tags=None):
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id="run-1",
            status="FINISHED",
            start_time=1000,
            end_time=2000,
        ),
        data=SimpleNamespace(
            metrics=metrics,
            params={"epochs": "3", "batch_size": "2"},
            tags={"celery_task_id": "task-1", **(tags or {})},
        ),
    )


def test_validation_metrics_are_grouped_by_label_and_macro_average():
    run = _run(
        metrics={
            "loss": 0.25,
            "val_mask_iou_label_42": 0.8,
            "val_mask_f1_label_42": 0.88,
            "val_mask_precision_label_42": 0.9,
            "val_mask_recall_label_42": 0.86,
            "val_mask_iou_label_7": 0.6,
            "val_mask_f1_label_7": 0.75,
            "val_mask_precision_label_7": 0.8,
            "val_mask_recall_label_7": 0.7,
            "val_mask_iou_macro": 0.7,
            "val_mask_f1_macro": 0.815,
            "val_mask_precision_macro": 0.85,
            "val_mask_recall_macro": 0.78,
        }
    )

    assert _validation_metrics_from_run(run) == {
        "ap": None,
        "ap50": None,
        "ap75": None,
        "macro_iou": 0.7,
        "macro_f1": 0.815,
        "macro_precision": 0.85,
        "macro_recall": 0.78,
        "per_label": [
            {
                "label_id": 7,
                "iou": 0.6,
                "f1": 0.75,
                "precision": 0.8,
                "recall": 0.7,
            },
            {
                "label_id": 42,
                "iou": 0.8,
                "f1": 0.88,
                "precision": 0.9,
                "recall": 0.86,
            },
        ],
    }


def test_validation_metrics_expose_instance_average_precision_metrics():
    run = _run(
        metrics={
            "val_mask_ap": 0.73,
            "val_mask_ap50": 0.91,
            "val_mask_ap75": 0.68,
        }
    )

    assert _validation_metrics_from_run(run) == {
        "ap": 0.73,
        "ap50": 0.91,
        "ap75": 0.68,
        "macro_iou": None,
        "macro_f1": None,
        "macro_precision": None,
        "macro_recall": None,
        "per_label": [],
    }


def test_snapshot_keeps_validation_unavailable_reason():
    run = _run(
        metrics={"loss": 0.25},
        tags={"validation_metrics_unavailable": "not_enough_images"},
    )

    snapshot = _snapshot_from_run(None, run, "task-1")

    assert snapshot["validation_metrics"] is None
    assert snapshot["validation_metrics_unavailable"] == "not_enough_images"

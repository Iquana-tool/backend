"""How people treated each model's suggestions on a dataset.

Read from the AI suggestion log (``ai_suggestions``). Every suggestion falls into
exactly one bucket, so the three shares add up to 100%:

* **rejected** -- its object was deleted;
* **edited** -- the object still exists, but a person changed its outline by hand;
* **as_is** -- the object still exists with the outline the model produced.

"Accepted" means kept, not approved in review: the summary works for datasets that
are never reviewed, and an unreviewed suggestion counts as accepted while it stands.

A suggestion replaced by a later AI refinement is left out (``superseded_at``): the
refinement is a suggestion of its own, and counting both would count one object twice.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.database.ai_suggestions import AiSuggestions

#: Order the tools are listed in, most interactive first.
SOURCE_ORDER = ["prompted", "refine", "instance_suggestion", "instance_segmentation",
                "cross_image", "batch_inference"]


def get_suggestion_stats(dataset_id: int, since: datetime | None, db: Session) -> list[dict]:
    """Per-model counts of suggestions kept as-is, edited and rejected.

    Models are ordered by how many suggestions they made, then by name.
    """
    rejected = AiSuggestions.deleted_at.is_not(None)
    edited = AiSuggestions.deleted_at.is_(None) & AiSuggestions.edited_at.is_not(None)
    conditions = [AiSuggestions.dataset_id == dataset_id, AiSuggestions.superseded_at.is_(None)]
    if since is not None:
        conditions.append(AiSuggestions.created_at >= since)

    models: dict[str, dict] = {}
    rows = db.execute(
        select(
            func.coalesce(AiSuggestions.model_key, "unknown"),
            AiSuggestions.source,
            func.count(),
            func.sum(case((rejected, 1), else_=0)),
            func.sum(case((edited, 1), else_=0)),
            func.max(AiSuggestions.created_at),
        )
        .where(*conditions)
        .group_by(func.coalesce(AiSuggestions.model_key, "unknown"), AiSuggestions.source)
    )
    for model_key, source, total, n_rejected, n_edited, last in rows:
        entry = models.setdefault(model_key, {
            "model_key": model_key, "total": 0, "as_is": 0, "edited": 0, "rejected": 0,
            "sources": set(), "last_used": None,
        })
        n_rejected, n_edited = int(n_rejected or 0), int(n_edited or 0)
        entry["total"] += total
        entry["rejected"] += n_rejected
        entry["edited"] += n_edited
        entry["as_is"] += total - n_rejected - n_edited
        entry["sources"].add(source)
        if last is not None and (entry["last_used"] is None or last > entry["last_used"]):
            entry["last_used"] = last

    ordered = sorted(models.values(), key=lambda m: (-m["total"], m["model_key"]))
    for entry in ordered:
        entry["sources"] = sorted(entry["sources"],
                                  key=lambda s: SOURCE_ORDER.index(s) if s in SOURCE_ORDER else 99)
        entry["last_used"] = entry["last_used"].isoformat() if entry["last_used"] else None
    return ordered

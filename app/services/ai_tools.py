"""Which AI tools a dataset offers.

A dataset owner can switch individual AI tools off: to keep a dataset manual-only,
to stop expensive batch runs, or to control exactly which assistance a user study's
participants get. Everything is on by default; the dataset stores only what is off
(``datasets.disabled_ai_tools``, a comma-separated list), so tools added later are
on for existing datasets without a migration.

The switches are enforced here, server-side, on every entry point. The frontend
hides switched-off tools too, but only as a convenience.
"""
from __future__ import annotations

from enum import StrEnum

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.database.datasets import Datasets


class AiTool(StrEnum):
    #: Run AI on placed prompts (points, boxes, polygons) to create an object.
    PROMPTED = "prompted"
    #: Add prompts to an existing object to refine its outline with AI.
    REFINE = "refine"
    #: "Suggest similar": find more instances like the selected ones on the same image.
    INSTANCE_SUGGESTION = "instance_suggestion"
    #: Run an instance segmentation model over the whole image.
    INSTANCE_SEGMENTATION = "instance_segmentation"
    #: Suggest instances on an image from exemplars on other images.
    CROSS_IMAGE = "cross_image"
    #: Train a model on the dataset.
    TRAINING = "training"
    #: Run models over the whole dataset.
    BATCH_INFERENCE = "batch_inference"


def parse(raw: str | None) -> frozenset[AiTool]:
    """Parse a stored comma-separated list, ignoring names this build does not know."""
    tools = set()
    for name in (raw or "").split(","):
        name = name.strip()
        if name in AiTool._value2member_map_:
            tools.add(AiTool(name))
    return frozenset(tools)


def validate(names: list[str]) -> frozenset[AiTool]:
    """Parse user input strictly: an unknown name is an error, not silently dropped."""
    unknown = [name for name in names if name not in AiTool._value2member_map_]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown AI tool(s): {', '.join(unknown)}. "
                   f"Known: {', '.join(tool.value for tool in AiTool)}.",
        )
    return frozenset(AiTool(name) for name in names)


def format_tools(tools: frozenset[AiTool]) -> str:
    return ",".join(sorted(tool.value for tool in tools))


def disabled_tools(dataset_id: int, db: Session) -> frozenset[AiTool]:
    raw = db.query(Datasets.disabled_ai_tools).filter_by(id=dataset_id).scalar()
    return parse(raw)


def is_enabled(dataset_id: int | None, tool: AiTool, db: Session) -> bool:
    if dataset_id is None:
        return True
    return tool not in disabled_tools(dataset_id, db)


def disabled_message(tool: AiTool) -> str:
    return f"The AI tool '{tool.value}' is switched off for this dataset."


def ensure_tool_enabled(dataset_id: int | None, tool: AiTool, db: Session) -> None:
    """Raise 403 when `tool` is switched off for the dataset."""
    if not is_enabled(dataset_id, tool, db):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=disabled_message(tool))

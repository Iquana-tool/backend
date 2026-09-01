from pydantic import BaseModel, Field


class LabelMoveRequest(BaseModel):
    """A request to move a label to a different parent.

    Attributes:
        new_parent_id: The label it becomes a part of, or ``None`` for the top level.
        detach_affected: Accept demoting the annotated objects the move would strand to
            root level. Without it a move that would invalidate annotations is refused
            and reports what it would break.
    """

    new_parent_id: int | None = Field(default=None)
    detach_affected: bool = Field(default=False)


class LabelUpdate(BaseModel):
    """The fields a label may be edited in place.

    Deliberately narrow. ``parent_id`` moves go through ``/labels/{id}/move`` because
    they can invalidate annotations, and ``value`` is what mask encodings are written
    against, so neither is patchable here.
    """

    name: str = Field(min_length=1)

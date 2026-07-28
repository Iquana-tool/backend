from pydantic import BaseModel, Field


class RetrievalStrategyOption(BaseModel):
    """A selectable exemplar-retrieval strategy, for the frontend picker.

    Mirrors ``AnnotationQueueStrategyOption``: ``available=False`` strategies are shown as
    placeholders. ``required_kinds`` tells the caller which embedding kinds must be
    precomputed for the strategy to run (and drives what the lifecycle layer embeds).
    """

    key: str
    label: str
    description: str
    available: bool
    required_kinds: list[str] = Field(default_factory=list)

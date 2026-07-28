from pydantic import BaseModel, Field


class CrossImageSuggestRequest(BaseModel):
    """Body for the cross-image suggestion endpoint.

    This is the backend's own contract (image ids + a strategy), deliberately *not* the
    ai-service ``CrossImageSuggestionRequest`` (which carries resolved image URLs + masks): the
    backend resolves the exemplars from the retrieval store server-side. Keeping the public body
    a backend schema also keeps the route import-safe before the toolbox pin is bumped.
    """

    target_image_id: int = Field(..., description="The image being annotated.")
    strategy: str = Field("global_scene", description="Retrieval strategy key (see /strategies).")
    concept_label_id: int | None = Field(
        default=None, description="Restrict exemplars to this label; also adds a text prompt."
    )
    query_contour_id: int | None = Field(
        default=None,
        description="For region-based strategies: an existing contour whose region embedding is "
                    "the query (e.g. a just-annotated object of the concept).",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Max exemplars to retrieve/transfer from.")


class ExemplarInfo(BaseModel):
    """One exemplar the suggestion was transferred from (returned for transparency)."""

    contour_id: int
    image_id: int
    score: float


class CrossImageSuggestResponse(BaseModel):
    success: bool
    message: str
    exemplars: list[ExemplarInfo] = Field(default_factory=list)
    result: list = Field(default_factory=list, description="Suggested contours on the target image.")

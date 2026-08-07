"""Schemas for batch inference: the plan a user builds, and the progress they watch.

The central idea is the **plan**: a list of steps, each binding *one label* to *one model*.
Nothing forces one model per dataset -- a user who trained a specialist per class points each
label at its own model, and the run orchestrates them. Because a step names the label it is
responsible for, a multiclass model's output is filtered down to that label (see
`app.services.inference.execution.filter_for_step`), so pointing three labels at the same
multiclass model and pointing them at three specialists are the same plan shape.

Steps are *not* ordered by the user. They are ordered by the label hierarchy: the resolver
stamps each step with its label's depth and sorts by it, so root labels are fully annotated
across the dataset before any child-level model runs. Child predictions need their parents to
already exist -- that is what they are attached to.

These schemas live in the backend rather than in `iquana-toolbox` for the same reason
`permissions.py` does: the toolbox is a git-pinned dependency, and the plan vocabulary is
gateway-internal -- the AI service never sees a plan, only one image-sized request at a time.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class WriteMode(StrEnum):
    """What a run does with annotations that already exist on the images in scope."""

    PATCH = "patch"
    """Keep everything, add the predictions, drop predictions that duplicate an existing
    object (NMS). Existing annotations are never removed, so a patching run is additive and
    safe to repeat."""

    REPLACE = "replace"
    """Delete the existing contours in scope -- with their child objects -- and start from
    the model output alone. Destructive and irreversible; the API demands an explicit
    acknowledgement (see `InferenceJobCreate.confirm_replace`)."""


class UnparentedPolicy(StrEnum):
    """What to do with a child-level prediction that lies inside no parent instance."""

    DROP = "drop"
    """Discard it. The default: a nucleus outside every cell is far more likely to be a false
    positive than a real object that the parent model missed."""

    KEEP_AT_ROOT = "keep_at_root"
    """Write it at root level with no parent, for a human to re-parent or delete."""


class ImageSelection(StrEnum):
    """Which of the dataset's images a run covers."""

    ALL = "all"
    NOT_STARTED = "not_started"
    """Only images whose mask has no contours yet -- the usual choice for a first pass."""
    UNREVIEWED = "unreviewed"
    """Images that are not fully reviewed yet, i.e. everything except finished masks."""
    CUSTOM = "custom"
    """The explicit `image_ids` given in the request."""


#: Task surfaces a plan step may target. Both produce whole-image instance predictions;
#: `cross-image-suggestion` additionally pulls exemplars from the embedding store, which is
#: why it needs a retrieval strategy.
InferenceTask = Literal["instance-segmentation", "cross-image-suggestion"]


class InferenceStepRequest(BaseModel):
    """One label, and the model that should annotate it."""

    label_id: int = Field(..., description="The label this step is responsible for producing.")
    model_registry_key: str = Field(..., description="Registry key of the model to run.")
    task: InferenceTask = Field(
        default="instance-segmentation",
        description="Which AI-service surface the model is called on.",
    )
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Predictions below this confidence are discarded before merging.",
    )
    # --- cross-image-suggestion only -------------------------------------------------
    retrieval_strategy: Optional[str] = Field(
        default=None,
        description="Exemplar-retrieval strategy; required for cross-image steps.",
    )
    top_k: int = Field(
        default=5, ge=1, le=32,
        description="How many exemplars a cross-image step retrieves per image.",
    )

    @model_validator(mode="after")
    def _require_strategy_for_cross_image(self) -> "InferenceStepRequest":
        if self.task == "cross-image-suggestion" and not self.retrieval_strategy:
            raise ValueError("A cross-image step needs a retrieval_strategy.")
        return self


class ResolvedStep(InferenceStepRequest):
    """A step after the planner has resolved it against the dataset.

    Adds everything the worker and the UI need without another lookup: where the label sits
    in the hierarchy (`level`, and the parent label a child step attaches to) and the display
    names. Stored verbatim in `inference_jobs.plan_steps`.
    """

    level: int = Field(..., description="Depth of the label in the hierarchy; 0 is root.")
    parent_label_id: Optional[int] = Field(
        default=None, description="The label whose instances this step's output is nested under."
    )
    label_name: str = Field(..., description="Display name of the target label.")
    model_name: str = Field(..., description="Display name of the model.")
    model_label_ids: list[int] = Field(
        default_factory=list,
        description="Labels the model itself predicts. Non-empty means its output is filtered "
                    "down to this step's label; empty means the model is class-agnostic and "
                    "its output is labelled with this step's label.",
    )


class InferenceOptions(BaseModel):
    """How predictions are merged into the dataset. One setting set per run."""

    write_mode: WriteMode = Field(default=WriteMode.PATCH)
    nms_iou: float = Field(
        default=0.7, ge=0.05, le=0.99,
        description="Two objects overlapping by more than this are considered the same object; "
                    "the lower-confidence one is dropped. Applies within a run and, in patch "
                    "mode, against annotations that already exist.",
    )
    preserve_reviewed: bool = Field(
        default=True,
        description="Replace mode only: keep contours somebody has already approved instead of "
                    "deleting them. Their children survive with them.",
    )
    unparented: UnparentedPolicy = Field(default=UnparentedPolicy.DROP)
    min_parent_containment: float = Field(
        default=0.5, ge=0.05, le=1.0,
        description="Fraction of a child prediction that must lie inside a parent instance for "
                    "it to be nested under it.",
    )


class InferenceJobCreate(BaseModel):
    """Request body for starting a run."""

    dataset_id: int
    name: Optional[str] = Field(
        default=None, max_length=80, pattern=r"^[\w\-\s]{1,80}$",
        description="Optional human-readable name shown in the run history.",
    )
    steps: list[InferenceStepRequest] = Field(..., min_length=1)
    image_selection: ImageSelection = Field(default=ImageSelection.ALL)
    image_ids: list[int] = Field(
        default_factory=list, description="Explicit scope; only read for image_selection=custom."
    )
    options: InferenceOptions = Field(default_factory=InferenceOptions)
    confirm_replace: bool = Field(
        default=False,
        description="Must be true to start a replace run. The gateway refuses otherwise, so a "
                    "destructive run can never be the result of a default-valued field.",
    )

    @model_validator(mode="after")
    def _one_model_per_label(self) -> "InferenceJobCreate":
        """Reject two steps targeting the same label.

        Running two models at one label is expressible -- but it makes the run's outcome
        depend on which model happened to go first, since the second one's output is NMS'd
        against the first one's. Refusing it keeps the plan a function, not a race.
        """
        seen: set[int] = set()
        duplicates: set[int] = set()
        for step in self.steps:
            if step.label_id in seen:
                duplicates.add(step.label_id)
            seen.add(step.label_id)
        if duplicates:
            raise ValueError(f"Each label may appear at most once in a plan; got {sorted(duplicates)}.")
        return self

    @model_validator(mode="after")
    def _custom_selection_needs_images(self) -> "InferenceJobCreate":
        if self.image_selection == ImageSelection.CUSTOM and not self.image_ids:
            raise ValueError("image_selection='custom' needs a non-empty image_ids list.")
        return self


class LevelProgress(BaseModel):
    """Per-hierarchy-level progress, so the UI can show what has and has not started."""

    level: int
    label_names: list[str] = Field(default_factory=list)
    total: int = 0
    done: int = 0
    failed: int = 0


class ActivityEntry(BaseModel):
    """One image the run just finished, for the live feed."""

    image_id: int
    image_name: Optional[str] = None
    label_name: Optional[str] = None
    contours_created: int = 0
    contours_suppressed: int = 0
    finished_at: Optional[datetime] = None


class InferenceJobSnapshot(BaseModel):
    """Everything the progress view renders. Emitted per SSE tick and on plain GETs."""

    id: int
    dataset_id: int
    name: Optional[str] = None
    created_by: Optional[str] = None
    status: str
    write_mode: WriteMode
    options: InferenceOptions
    steps: list[ResolvedStep] = Field(default_factory=list)

    total_units: int = 0
    done_units: int = 0
    failed_units: int = 0
    image_count: int = 0

    contours_created: int = 0
    contours_suppressed: int = 0
    contours_deleted: int = 0
    contours_unparented: int = 0

    levels: list[LevelProgress] = Field(default_factory=list)
    current_step: Optional[ResolvedStep] = Field(
        default=None, description="The step the worker is on, for the 'now running' line."
    )
    recent_activity: list[ActivityEntry] = Field(
        default_factory=list,
        description="The last handful of finished images, newest first. Counts and names "
                    "only -- deliberately no geometry, so the stream stays a few hundred "
                    "bytes however many objects the model found.",
    )
    #: Seconds left, extrapolated from the mean duration of recently finished units. None
    #: until enough units have finished for the estimate to mean anything.
    eta_seconds: Optional[float] = None
    error: Optional[str] = None

    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class InferenceJobItemRead(BaseModel):
    """One work unit, for the failed-items list."""

    id: int
    level: int
    step_index: int
    image_id: int
    image_name: Optional[str] = None
    label_name: Optional[str] = None
    model_registry_key: Optional[str] = None
    status: str
    contours_created: int = 0
    contours_suppressed: int = 0
    contours_unparented: int = 0
    duration_ms: Optional[float] = None
    error: Optional[str] = None


class ScopeCounts(BaseModel):
    """Image counts per selection, so the scope picker can label its options."""

    total: int = 0
    not_started: int = 0
    unreviewed: int = 0


class ReplacePreview(BaseModel):
    """What a replace run would destroy. Rendered verbatim in the confirmation dialog."""

    images: int = Field(..., description="Images in scope.")
    contours: int = Field(..., description="Contours that would be deleted, children included.")
    reviewed_contours: int = Field(
        ..., description="Of those, how many somebody has already approved."
    )
    root_contours: int = Field(
        ..., description="Top-level objects; each takes its whole subtree with it."
    )
    protected_contours: int = Field(
        default=0,
        description="Contours that survive because preserve_reviewed is on (approved objects "
                    "and their descendants).",
    )


class ModelOption(BaseModel):
    """One selectable model in the per-label picker."""

    registry_key: str
    name: str
    task: InferenceTask
    description: Optional[str] = None
    usage_tip: Optional[str] = None
    badges: list[str] = Field(default_factory=list)
    architecture: Optional[str] = None
    label_ids: list[int] = Field(
        default_factory=list,
        description="Labels this model predicts. Empty means class-agnostic: it can be pointed "
                    "at any label, and its output is labelled with whichever label it is bound to.",
    )
    trained_on_dataset: bool = Field(
        default=False,
        description="Whether the model was trained on the dataset being annotated. Models that "
                    "were are sorted first in the picker.",
    )


class ModelCatalog(BaseModel):
    """The picker's entire input: one call instead of one per task."""

    models: list[ModelOption] = Field(default_factory=list)
    retrieval_strategies: list[dict] = Field(
        default_factory=list,
        description="Selectable exemplar-retrieval strategies for cross-image steps. Empty when "
                    "the embedding store is not populated, which is also why those models are "
                    "not offered.",
    )

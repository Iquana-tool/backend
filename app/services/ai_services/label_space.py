"""LLM-assisted label-space generation.

Provider-agnostic: built on LiteLLM (one call signature for ~100 providers,
selected by the ``LABEL_SPACE_LLM_MODEL`` prefix, e.g. ``anthropic/...``,
``openai/...``, ``gemini/...``, ``ollama/...``) and Instructor, which coerces
the model output into a validated :class:`LabelSpaceDraft` and retries on
schema-validation errors.

Configuration (see ``config.py`` / ``.env``):
    LABEL_SPACE_LLM_MODEL     LiteLLM model id, default ``anthropic/claude-opus-4-8``.
    LABEL_SPACE_LLM_API_KEY   API key for the matching provider. Generation is
                              disabled until this is set.
    LABEL_SPACE_LLM_API_BASE  Optional base URL (self-hosted / Azure / Ollama).
"""
from __future__ import annotations

from logging import getLogger

from fastapi import HTTPException

from app.schemas.label_space import DraftLabel, LabelSpaceDraft
from config import (
    LABEL_SPACE_LLM_API_BASE,
    LABEL_SPACE_LLM_API_KEY,
    LABEL_SPACE_LLM_MODEL,
)

logger = getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert in annotation schemas for image segmentation.
Given a plain-language description of what a user wants to segment, design a clear
label space organised by physical containment.

Rules:
- Nesting means PART OF, never "kind of". A child label must be a part or component of
  its parent: "Nucleus" under "Cell" means a nucleus is part of a cell. If you cannot say
  "an X is part of a Y", X must not be nested under Y.
- Subtypes are NOT nested. "Acropora" is a kind of coral, not a part of one, so it belongs
  at the top level alongside "Coral" rather than underneath it. The target system cannot
  record "is a kind of" at all — never express one by nesting.
- Whole objects go at the top level; their components go underneath them.
- Sibling labels must be mutually exclusive: one object is one of them, never two.
- Every label name MUST be unique across the ENTIRE hierarchy (not just among its siblings).
  The target system requires dataset-wide unique names. If two parents would share a part
  name, qualify it (e.g. "Car Wheel" vs "Truck Wheel").
- Use concise, Title Case names a domain expert would recognise.
- Do not exceed the requested maximum depth or total label count.
- Do not invent a "Background" label unless the user explicitly asks for one.
- Add a short `description` to each label explaining what it covers.
Return only the structured hierarchy."""


def _count(labels: list[DraftLabel]) -> int:
    return sum(1 + _count(child.children) for child in labels) if labels else 0


def _depth(labels: list[DraftLabel], current: int = 1) -> int:
    best = 0 if not labels else current
    for label in labels:
        if label.children:
            best = max(best, _depth(label.children, current + 1))
    return best


class LabelSpaceService:
    """Generates and refines draft label hierarchies via an LLM."""

    @staticmethod
    def is_enabled() -> bool:
        return bool(LABEL_SPACE_LLM_API_KEY)

    @staticmethod
    def model() -> str:
        return LABEL_SPACE_LLM_MODEL

    def _client(self):
        # Imported lazily so the rest of the app runs without the optional deps installed.
        try:
            import instructor
            import litellm
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise HTTPException(
                status_code=503,
                detail="Label-space generation requires the 'litellm' and 'instructor' packages.",
            ) from exc
        return instructor.from_litellm(litellm.completion)

    def _create(self, messages: list[dict], model: str | None, max_depth: int, max_labels: int) -> LabelSpaceDraft:
        if not self.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Label-space generation is not configured. Set LABEL_SPACE_LLM_API_KEY on the server.",
            )
        client = self._client()
        kwargs = dict(
            model=model or LABEL_SPACE_LLM_MODEL,
            response_model=LabelSpaceDraft,
            messages=messages,
            api_key=LABEL_SPACE_LLM_API_KEY,
            max_retries=2,
            temperature=0.2,
        )
        if LABEL_SPACE_LLM_API_BASE:
            kwargs["api_base"] = LABEL_SPACE_LLM_API_BASE
        try:
            draft: LabelSpaceDraft = client.chat.completions.create(**kwargs)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise any provider error
            logger.exception("Label-space generation failed")
            raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

        return self._validate(draft, max_depth, max_labels)

    @staticmethod
    def _validate(draft: LabelSpaceDraft, max_depth: int, max_labels: int) -> LabelSpaceDraft:
        if not draft.labels:
            raise HTTPException(status_code=422, detail="The model returned an empty label space.")
        if _count(draft.labels) > max_labels:
            raise HTTPException(
                status_code=422,
                detail=f"Generated {_count(draft.labels)} labels, exceeding the limit of {max_labels}.",
            )
        if _depth(draft.labels) > max_depth:
            raise HTTPException(
                status_code=422,
                detail=f"Generated hierarchy is deeper than the allowed {max_depth} levels.",
            )
        # Enforce dataset-wide unique names (the persistence layer requires it).
        seen: set[str] = set()
        duplicates: set[str] = set()

        def walk(labels: list[DraftLabel]) -> None:
            for label in labels:
                key = label.name.strip().lower()
                if key in seen:
                    duplicates.add(label.name)
                seen.add(key)
                walk(label.children)

        walk(draft.labels)
        if duplicates:
            raise HTTPException(
                status_code=422,
                detail=f"The model produced duplicate label names: {', '.join(sorted(duplicates))}.",
            )
        return draft

    def generate(self, description: str, max_depth: int, max_labels: int, model: str | None = None) -> LabelSpaceDraft:
        user = (
            f"Description of what to segment:\n{description}\n\n"
            f"Constraints: maximum depth {max_depth}, maximum {max_labels} labels total."
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return self._create(messages, model, max_depth, max_labels)

    def refine(
        self,
        current_draft: LabelSpaceDraft,
        message: str,
        description: str | None,
        max_depth: int,
        max_labels: int,
        model: str | None = None,
    ) -> LabelSpaceDraft:
        context = f"Original description:\n{description}\n\n" if description else ""
        user = (
            f"{context}Here is the current draft label space (JSON):\n"
            f"{current_draft.model_dump_json(indent=2)}\n\n"
            f"Revise it according to this instruction:\n{message}\n\n"
            f"Constraints: maximum depth {max_depth}, maximum {max_labels} labels total. "
            f"Return the full revised hierarchy."
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        return self._create(messages, model, max_depth, max_labels)

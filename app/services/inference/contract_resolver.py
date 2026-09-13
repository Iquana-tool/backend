"""Effective inference input contract resolver.

Resolves model-declared contracts and per-task legacy defaults with provenance tracking.
"""
from __future__ import annotations

import json
from typing import Any, Literal

from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.model_info import ModelInfo
from iquana_toolbox.schemas.training import HyperParameter

Provenance = Literal["declared", "legacy_default"]

#: Explicit per-task legacy default contracts preserving existing platform behavior.
LEGACY_TASK_DEFAULTS: dict[str, InputContract] = {
    "instance-segmentation": InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[],
        notes="Legacy default contract for autonomous instance segmentation.",
    ),
    "cross-image-suggestion": InputContract(
        task="cross-image-suggestion",
        conditioning=ConditioningSpec(
            kind="instances",
            unit="instance",
            min_units=1,
            max_units=32,
            user_selectable_count=True,
        ),
        parameters=[],
        notes="Legacy default contract for cross-image exemplar transfer.",
    ),
    "instance-suggestion": InputContract(
        task="instance-suggestion",
        conditioning=ConditioningSpec(
            kind="instances",
            unit="instance",
            min_units=1,
            max_units=32,
            user_selectable_count=True,
        ),
        parameters=[],
        notes="Legacy default contract for interactive instance suggestion.",
    ),
    "prompted-segmentation": InputContract(
        task="prompted-segmentation",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[],
        notes="Legacy default contract for point/box prompted segmentation.",
    ),
}


def _extract_raw_contracts(model_info: dict[str, Any] | ModelInfo | None) -> list[Any] | None:
    """Extract raw contract items from a ModelInfo instance or dictionary."""
    if model_info is None:
        return None

    if isinstance(model_info, ModelInfo):
        if not isinstance(model_info.input_contracts, list):
            raise ValueError(
                f"Malformed declared input_contracts: expected list, got {type(model_info.input_contracts).__name__}"
            )
        return model_info.input_contracts

    if not isinstance(model_info, dict):
        raise ValueError(f"model_info must be a ModelInfo, dict, or None, got {type(model_info).__name__}")

    # Registered model tags take precedence over artifact-level metadata so synchronized
    # updates are visible without re-logging artifacts.
    tags = model_info.get("tags")
    if isinstance(tags, dict) and "input_contracts" in tags:
        tag_val = tags["input_contracts"]
        if tag_val is None:
            raise ValueError("Malformed declared input_contracts in tags: expected list, got None")
        if isinstance(tag_val, str):
            try:
                tag_val = json.loads(tag_val)
            except Exception as exc:
                raise ValueError(f"Malformed input_contracts tag JSON: {exc}") from exc
        if not isinstance(tag_val, list):
            raise ValueError(
                f"Malformed declared input_contracts in tags: expected list, got {type(tag_val).__name__ if tag_val is not None else 'None'}"
            )
        return tag_val
    elif isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict) and tag.get("key") == "input_contracts":
                val = tag.get("value")
                if val is None:
                    raise ValueError("Malformed declared input_contracts in tag value: expected list, got None")
                if isinstance(val, str):
                    try:
                        val = json.loads(val)
                    except Exception as exc:
                        raise ValueError(f"Malformed input_contracts tag JSON: {exc}") from exc
                if not isinstance(val, list):
                    raise ValueError(
                        f"Malformed declared input_contracts in tag value: expected list, got {type(val).__name__ if val is not None else 'None'}"
                    )
                return val

    if "input_contracts" in model_info:
        raw = model_info["input_contracts"]
        if raw is None:
            raise ValueError("Malformed declared input_contracts: expected list, got None")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"Malformed declared input_contracts JSON: {exc}") from exc
        if not isinstance(raw, list):
            raise ValueError(
                f"Malformed declared input_contracts: expected list, got {type(raw).__name__ if raw is not None else 'None'}"
            )
        return raw

    return None


def resolve_input_contract(
    model_info: dict[str, Any] | ModelInfo | None,
    task: str,
) -> tuple[InputContract, Provenance]:
    """Resolve the effective InputContract and provenance for a model and task.

    Args:
        model_info: ModelInfo schema object, dictionary, or None.
        task: Target task surface name (e.g. "instance-segmentation", "cross-image-suggestion").

    Returns:
        A tuple of `(effective_contract, provenance)` where provenance is `"declared"`
        if the model provided a valid contract for `task`, or `"legacy_default"` if
        resolved from fallback task defaults.

    Raises:
        ValueError: If declared contracts contain duplicate tasks, malformed data, or
            if `task` is unknown and has no declared contract.
    """
    raw_list = _extract_raw_contracts(model_info)

    if raw_list is not None:
        if not isinstance(raw_list, list):
            raise ValueError(f"Malformed declared input contracts: expected list, got {type(raw_list).__name__}")
        parsed_contracts: list[InputContract] = []
        seen_tasks: set[str] = set()

        for item in raw_list:
            if isinstance(item, InputContract):
                contract = item
            elif isinstance(item, dict):
                try:
                    contract = InputContract.model_validate(item)
                except Exception as exc:
                    raise ValueError(f"Malformed declared input contract: {exc}") from exc
            else:
                raise ValueError(f"Invalid input contract item type: {type(item).__name__}")

            if contract.task in seen_tasks:
                raise ValueError(f"Duplicate declared input contract for task '{contract.task}'")
            seen_tasks.add(contract.task)
            parsed_contracts.append(contract)

        for contract in parsed_contracts:
            if contract.task == task:
                return contract, "declared"

    if task in LEGACY_TASK_DEFAULTS:
        return LEGACY_TASK_DEFAULTS[task], "legacy_default"

    raise ValueError(f"Unknown task '{task}' with no declared contract or legacy default.")

"""Generic inference input validator and normalizer (Issue #27).

Validates submitted inference inputs against a model's effective InputContract,
filling defaults, validating types, ranges, options, and enforcing conditioning
cardinality bounds and semantics.
"""
from __future__ import annotations

import math
from typing import Any

from iquana_toolbox.schemas.input_contract import InputContract


def validate_and_normalize_inputs(
    contract: InputContract,
    raw_inputs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate and normalize inference inputs against an :class:`InputContract`.

    Args:
        contract: The effective InputContract for the selected model and task.
        raw_inputs: Generic inputs dictionary containing "conditioning" and/or "parameters",
            or None.

    Returns:
        A canonical normalized dictionary:
        {
            "conditioning": {
                "count": int,
                "strategy": str | None,
                "concept_text": str | None,
            },
            "parameters": { ... }
        }

    Raises:
        ValueError: If input structure, parameter types, bounds, options, or conditioning
            cardinalities violate the contract.
    """
    if raw_inputs is None:
        raw_inputs = {}

    if not isinstance(raw_inputs, dict):
        raise ValueError(f"inputs must be a dictionary, got {type(raw_inputs).__name__}")

    # Reject unknown top-level envelope keys
    allowed_top_level = {"conditioning", "parameters"}
    unknown_top_level = set(raw_inputs.keys()) - allowed_top_level
    if unknown_top_level:
        raise ValueError(
            f"Unknown keys in inputs envelope: {sorted(unknown_top_level)}. "
            f"Allowed keys: {sorted(allowed_top_level)}"
        )

    raw_params = raw_inputs.get("parameters")
    if raw_params is None:
        raw_params = {}
    elif not isinstance(raw_params, dict):
        raise ValueError(f"parameters must be a dictionary, got {type(raw_params).__name__}")

    raw_cond = raw_inputs.get("conditioning")
    if raw_cond is None:
        raw_cond = {}
    elif not isinstance(raw_cond, dict):
        raise ValueError(f"conditioning must be a dictionary, got {type(raw_cond).__name__}")

    # Reject unknown conditioning keys
    allowed_cond_keys = {"count", "strategy", "concept_text", "query_contour_id"}
    unknown_cond_keys = set(raw_cond.keys()) - allowed_cond_keys
    if unknown_cond_keys:
        raise ValueError(
            f"Unknown conditioning keys: {sorted(unknown_cond_keys)}. "
            f"Allowed keys: {sorted(allowed_cond_keys)}"
        )

    # ----------------------------------------------------------------------- #
    # 1. Parameter validation and default normalization
    # ----------------------------------------------------------------------- #
    declared_by_key = {p.key: p for p in contract.parameters}

    # Reject unknown parameter keys
    unknown_params = set(raw_params.keys()) - set(declared_by_key.keys())
    if unknown_params:
        raise ValueError(
            f"Unknown parameter(s) {sorted(unknown_params)} for task '{contract.task}'. "
            f"Declared parameters: {sorted(declared_by_key.keys())}"
        )

    normalized_parameters: dict[str, Any] = {}
    for key, spec in declared_by_key.items():
        if key in raw_params and raw_params[key] is not None:
            val = raw_params[key]
        else:
            val = spec.default_value

        spec_type = spec.type
        if spec_type == "bool":
            if not isinstance(val, bool):
                raise ValueError(
                    f"Parameter '{key}' must be a bool, got {type(val).__name__} ({val!r})"
                )
            normalized_val: Any = bool(val)
        elif spec_type == "int":
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(
                    f"Parameter '{key}' must be an int, got {type(val).__name__} ({val!r})"
                )
            normalized_val = int(val)
        elif spec_type == "float":
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(
                    f"Parameter '{key}' must be a finite float, got {type(val).__name__} ({val!r})"
                )
            normalized_val = float(val)
        elif spec_type == "str":
            if not isinstance(val, str):
                raise ValueError(
                    f"Parameter '{key}' must be a str, got {type(val).__name__} ({val!r})"
                )
            normalized_val = str(val)
        else:
            raise ValueError(f"Unsupported parameter type '{spec_type}' for '{key}'")

        if spec.options is not None:
            if normalized_val not in spec.options:
                raise ValueError(
                    f"Parameter '{key}' value {normalized_val!r} is not in allowed options: {spec.options}"
                )

        if spec_type in {"int", "float"}:
            if spec.min_value is not None and normalized_val < spec.min_value:
                raise ValueError(
                    f"Parameter '{key}' value {normalized_val} is less than min_value {spec.min_value}"
                )
            if spec.max_value is not None and normalized_val > spec.max_value:
                raise ValueError(
                    f"Parameter '{key}' value {normalized_val} is greater than max_value {spec.max_value}"
                )

        normalized_parameters[key] = normalized_val

    # ----------------------------------------------------------------------- #
    # 2. Conditioning validation and normalization
    # ----------------------------------------------------------------------- #
    cond_kind = contract.conditioning.kind
    normalized_conditioning: dict[str, Any]

    query_contour_id = raw_cond.get("query_contour_id")
    if query_contour_id is not None:
        if isinstance(query_contour_id, bool) or not isinstance(query_contour_id, int):
            raise ValueError(
                f"query_contour_id must be an integer, got {type(query_contour_id).__name__}"
            )

    if cond_kind == "none":
        if raw_cond.get("strategy") is not None:
            raise ValueError("Conditioning kind 'none' does not accept a retrieval strategy.")
        if raw_cond.get("concept_text") is not None:
            raise ValueError("Conditioning kind 'none' does not accept concept_text.")
        if query_contour_id is not None:
            raise ValueError("Conditioning kind 'none' does not accept query_contour_id.")
        count_val = raw_cond.get("count")
        if count_val is not None and count_val != 0:
            raise ValueError(f"Conditioning kind 'none' does not accept count > 0 (got {count_val}).")
        normalized_conditioning = {
            "count": 0,
            "strategy": None,
            "concept_text": None,
            "query_contour_id": None,
        }

    elif cond_kind == "concept_text":
        if raw_cond.get("strategy") is not None:
            raise ValueError("Conditioning kind 'concept_text' does not accept a retrieval strategy.")
        if query_contour_id is not None:
            raise ValueError("Conditioning kind 'concept_text' does not accept query_contour_id.")
        count_val = raw_cond.get("count")
        if count_val is not None and count_val != 0:
            raise ValueError(f"Conditioning kind 'concept_text' does not accept count > 0 (got {count_val}).")
        text = raw_cond.get("concept_text")
        if text is not None and not isinstance(text, str):
            raise ValueError(f"concept_text must be a string, got {type(text).__name__}")
        normalized_conditioning = {
            "count": 0,
            "strategy": None,
            "concept_text": text,
            "query_contour_id": None,
        }

    else:  # instances, reference_images, embeddings
        # Validate count
        raw_count = raw_cond.get("count")
        if raw_count is None:
            # Default to min_units (or 1 if min_units is 0/None)
            count = contract.conditioning.min_units if contract.conditioning.min_units > 0 else 1
        else:
            if isinstance(raw_count, bool) or not isinstance(raw_count, int):
                raise ValueError(f"Conditioning count must be an integer, got {type(raw_count).__name__}")
            count = raw_count

        if count < contract.conditioning.min_units:
            raise ValueError(
                f"Conditioning count {count} is below min_units {contract.conditioning.min_units} "
                f"for task '{contract.task}'."
            )
        if contract.conditioning.max_units is not None and count > contract.conditioning.max_units:
            raise ValueError(
                f"Conditioning count {count} exceeds max_units {contract.conditioning.max_units} "
                f"for task '{contract.task}'."
            )

        # Validate strategy
        strategy = raw_cond.get("strategy")
        if strategy is not None and not isinstance(strategy, str):
            raise ValueError(f"retrieval strategy must be a string, got {type(strategy).__name__}")

        if cond_kind == "instances":
            if strategy is not None:
                raise ValueError("Conditioning kind 'instances' does not accept a retrieval strategy.")
            if query_contour_id is not None:
                raise ValueError("Conditioning kind 'instances' does not accept query_contour_id.")

        elif cond_kind == "reference_images":
            effective_strat = strategy or "global_scene"
            from app.services.exemplar_retrieval import is_region_based_strategy
            if query_contour_id is not None and not is_region_based_strategy(effective_strat):
                raise ValueError(
                    f"Retrieval strategy '{effective_strat}' does not accept query_contour_id; "
                    "query_contour_id is only accepted for region-based retrieval strategies."
                )

        elif cond_kind == "embeddings":
            if strategy is not None:
                raise ValueError("Conditioning kind 'embeddings' does not accept a retrieval strategy.")
            emb_kinds = contract.conditioning.embedding_kinds or ["image_cls"]
            if query_contour_id is not None and "region_mean" not in emb_kinds:
                raise ValueError(
                    "Embedding conditioning without 'region_mean' does not accept query_contour_id."
                )

        # Validate concept_text if provided
        concept_text = raw_cond.get("concept_text")
        if concept_text is not None and not isinstance(concept_text, str):
            raise ValueError(f"concept_text must be a string, got {type(concept_text).__name__}")

        normalized_conditioning = {
            "count": count,
            "strategy": strategy,
            "concept_text": concept_text,
            "query_contour_id": query_contour_id,
        }

    return {
        "conditioning": normalized_conditioning,
        "parameters": normalized_parameters,
    }

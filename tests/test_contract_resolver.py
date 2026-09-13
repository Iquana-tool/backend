"""Unit tests for backend effective input contract resolver (Issue #27)."""
from __future__ import annotations

import json
import pytest

from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.model_info import ModelInfo
from iquana_toolbox.schemas.training import HyperParameter

from app.services.inference.contract_resolver import (
    LEGACY_TASK_DEFAULTS,
    resolve_input_contract,
)


def test_resolve_input_contract_none_model_info():
    """None model_info resolves to legacy default with 'legacy_default' provenance."""
    contract, provenance = resolve_input_contract(None, "instance-segmentation")
    assert provenance == "legacy_default"
    assert contract == LEGACY_TASK_DEFAULTS["instance-segmentation"]
    assert contract.task == "instance-segmentation"
    assert contract.conditioning.kind == "none"


def test_resolve_input_contract_empty_contracts():
    """Empty input_contracts in dict resolves to legacy default."""
    contract, provenance = resolve_input_contract({"input_contracts": []}, "cross-image-suggestion")
    assert provenance == "legacy_default"
    assert contract == LEGACY_TASK_DEFAULTS["cross-image-suggestion"]
    assert contract.conditioning.kind == "instances"
    assert contract.conditioning.unit == "instance"
    assert contract.conditioning.min_units == 1
    assert contract.conditioning.max_units == 32
    assert contract.conditioning.user_selectable_count is True
    assert contract.parameters == []


def test_resolve_input_contract_tag_overrides_artifact_contract():
    """Registered model tag contracts override stale artifact-level contracts."""
    stale_artifact_contract = {
        "task": "instance-segmentation",
        "conditioning": {"kind": "none", "user_selectable_count": False},
        "parameters": [
            {
                "key": "threshold",
                "label": "Stale Old Threshold",
                "type": "float",
                "default_value": 0.1,
            }
        ],
    }
    updated_tag_contract = {
        "task": "instance-segmentation",
        "conditioning": {"kind": "none", "user_selectable_count": False},
        "parameters": [
            {
                "key": "threshold",
                "label": "Updated Tag Threshold",
                "type": "float",
                "default_value": 0.85,
            }
        ],
    }
    model_info = {
        "name": "model_with_stale_artifact",
        "input_contracts": [stale_artifact_contract],
        "tags": {"input_contracts": json.dumps([updated_tag_contract])},
    }
    contract, provenance = resolve_input_contract(model_info, "instance-segmentation")
    assert provenance == "declared"
    assert contract.parameters[0].label == "Updated Tag Threshold"
    assert contract.parameters[0].default_value == 0.85


def test_resolve_input_contract_declared_dict():
    """Dict model_info with declared contract resolves with 'declared' provenance."""
    custom_contract = {
        "task": "instance-segmentation",
        "conditioning": {"kind": "none", "user_selectable_count": False},
        "parameters": [
            {
                "key": "threshold",
                "label": "Custom Threshold",
                "type": "float",
                "default_value": 0.65,
                "min_value": 0.1,
                "max_value": 0.9,
            }
        ],
    }
    model_info = {"name": "custom_m2f", "input_contracts": [custom_contract]}
    contract, provenance = resolve_input_contract(model_info, "instance-segmentation")
    assert provenance == "declared"
    assert contract.task == "instance-segmentation"
    assert len(contract.parameters) == 1
    assert contract.parameters[0].default_value == 0.65


def test_resolve_input_contract_declared_model_info_object():
    """ModelInfo Pydantic object with input_contracts resolves with 'declared' provenance."""
    custom_contract = InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[
            HyperParameter(
                key="threshold",
                label="Threshold",
                type="float",
                default_value=0.7,
            )
        ],
    )
    info = ModelInfo(
        registry_key="custom-m2f",
        name="Custom M2F",
        description="desc",
        usage_tip="tip",
        status="ready",
        input_contracts=[custom_contract],
    )
    contract, provenance = resolve_input_contract(info, "instance-segmentation")
    assert provenance == "declared"
    assert contract == custom_contract


def test_resolve_input_contract_from_tags_json_dict():
    """Tags dict carrying stringified input_contracts JSON parses and resolves correctly."""
    custom_contract = {
        "task": "cross-image-suggestion",
        "conditioning": {
            "kind": "reference_images",
            "unit": "image",
            "min_units": 1,
            "max_units": 2,
            "requires_complete_annotation": True,
            "user_selectable_count": True,
        },
        "parameters": [],
    }
    model_info = {
        "name": "sam3",
        "tags": {"input_contracts": json.dumps([custom_contract])},
    }
    contract, provenance = resolve_input_contract(model_info, "cross-image-suggestion")
    assert provenance == "declared"
    assert contract.conditioning.max_units == 2


def test_resolve_input_contract_from_tags_list():
    """Tags list format carrying stringified input_contracts JSON parses and resolves correctly."""
    custom_contract = {
        "task": "instance-suggestion",
        "conditioning": {
            "kind": "instances",
            "unit": "instance",
            "min_units": 1,
            "user_selectable_count": True,
        },
        "parameters": [],
    }
    model_info = {
        "name": "sam3",
        "tags": [{"key": "input_contracts", "value": json.dumps([custom_contract])}],
    }
    contract, provenance = resolve_input_contract(model_info, "instance-suggestion")
    assert provenance == "declared"
    assert contract.task == "instance-suggestion"


def test_resolve_input_contract_declared_other_task_falls_back():
    """Declared contracts that don't match requested task fall back to legacy default."""
    model_info = {
        "input_contracts": [
            {
                "task": "prompted-segmentation",
                "conditioning": {"kind": "none", "user_selectable_count": False},
                "parameters": [],
            }
        ]
    }
    contract, provenance = resolve_input_contract(model_info, "instance-segmentation")
    assert provenance == "legacy_default"
    assert contract == LEGACY_TASK_DEFAULTS["instance-segmentation"]


def test_resolve_input_contract_duplicate_declared_tasks_raises():
    """Duplicate task declarations in input_contracts must raise ValueError."""
    model_info = {
        "input_contracts": [
            {
                "task": "instance-segmentation",
                "conditioning": {"kind": "none", "user_selectable_count": False},
                "parameters": [],
            },
            {
                "task": "instance-segmentation",
                "conditioning": {"kind": "none", "user_selectable_count": False},
                "parameters": [],
            },
        ]
    }
    with pytest.raises(ValueError, match="Duplicate declared input contract for task 'instance-segmentation'"):
        resolve_input_contract(model_info, "instance-segmentation")


def test_resolve_input_contract_malformed_declared_raises():
    """Malformed declared contract must raise ValueError."""
    model_info = {
        "input_contracts": [
            {
                "task": "instance-segmentation",
                "conditioning": {"kind": "none", "unit": "instance"},  # invalid: none cannot have unit
            }
        ]
    }
    with pytest.raises(ValueError, match="Malformed declared input contract"):
        resolve_input_contract(model_info, "instance-segmentation")


def test_resolve_input_contract_unknown_task_raises():
    """Unknown task with no declared contract and no legacy default raises ValueError."""
    with pytest.raises(ValueError, match="Unknown task 'unknown-specialist'"):
        resolve_input_contract(None, "unknown-specialist")


def test_resolve_input_contract_malformed_metadata_non_list_raises():
    """Declaring non-list input_contracts in dict or stringified JSON must raise ValueError."""
    with pytest.raises(ValueError, match="Malformed declared input_contracts: expected list"):
        resolve_input_contract({"input_contracts": {"task": "instance-segmentation"}}, "instance-segmentation")

    with pytest.raises(ValueError, match="Malformed declared input_contracts: expected list"):
        resolve_input_contract({"input_contracts": 123}, "instance-segmentation")

    with pytest.raises(ValueError, match="Malformed declared input_contracts: expected list"):
        resolve_input_contract({"input_contracts": '{"task": "instance-segmentation"}'}, "instance-segmentation")

    # Explicit null in dict or JSON string
    with pytest.raises(ValueError, match="Malformed declared input_contracts: expected list, got None"):
        resolve_input_contract({"input_contracts": None}, "instance-segmentation")

    with pytest.raises(ValueError, match="Malformed declared input_contracts: expected list, got None"):
        resolve_input_contract({"input_contracts": "null"}, "instance-segmentation")

    # Tuple is rejected (list-only metadata)
    with pytest.raises(ValueError, match="Malformed declared input_contracts: expected list, got tuple"):
        resolve_input_contract({"input_contracts": ()}, "instance-segmentation")


def test_resolve_input_contract_malformed_metadata_invalid_json_string_raises():
    """Invalid JSON string in input_contracts must raise ValueError."""
    with pytest.raises(ValueError, match="Malformed declared input_contracts JSON"):
        resolve_input_contract({"input_contracts": "invalid-json"}, "instance-segmentation")


def test_resolve_input_contract_malformed_tags_non_list_raises():
    """Non-list tag value for input_contracts must raise ValueError."""
    with pytest.raises(ValueError, match="Malformed declared input_contracts in tags"):
        resolve_input_contract({"tags": {"input_contracts": 123}}, "instance-segmentation")

    with pytest.raises(ValueError, match="Malformed declared input_contracts in tags: expected list, got None"):
        resolve_input_contract({"tags": {"input_contracts": None}}, "instance-segmentation")

    with pytest.raises(ValueError, match="Malformed declared input_contracts in tags: expected list, got None"):
        resolve_input_contract({"tags": {"input_contracts": "null"}}, "instance-segmentation")

    with pytest.raises(ValueError, match="Malformed declared input_contracts in tag value"):
        resolve_input_contract({"tags": [{"key": "input_contracts", "value": 123}]}, "instance-segmentation")

    with pytest.raises(ValueError, match="Malformed declared input_contracts in tag value: expected list, got None"):
        resolve_input_contract({"tags": [{"key": "input_contracts", "value": None}]}, "instance-segmentation")

    with pytest.raises(ValueError, match="Malformed declared input_contracts in tag value: expected list, got None"):
        resolve_input_contract({"tags": [{"key": "input_contracts", "value": "null"}]}, "instance-segmentation")


def test_resolve_input_contract_invalid_model_info_type_raises():
    """Non-dict and non-ModelInfo model_info (other than None) must raise ValueError."""
    with pytest.raises(ValueError, match="model_info must be a ModelInfo, dict, or None"):
        resolve_input_contract("invalid-model-info-string", "instance-segmentation")

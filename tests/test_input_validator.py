"""Unit tests for backend generic inference input validator (Issue #27)."""
from __future__ import annotations

import pytest

from iquana_toolbox.schemas.input_contract import ConditioningSpec, InputContract
from iquana_toolbox.schemas.training import HyperParameter

from app.services.inference.input_validator import validate_and_normalize_inputs


def test_validate_and_normalize_inputs_fills_defaults():
    """Omitted parameters receive declared defaults."""
    contract = InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[
            HyperParameter(
                key="threshold",
                label="Threshold",
                type="float",
                default_value=0.5,
                min_value=0.0,
                max_value=1.0,
            ),
            HyperParameter(
                key="mode",
                label="Mode",
                type="str",
                default_value="standard",
                options=["fast", "standard", "accurate"],
            ),
        ],
    )
    res = validate_and_normalize_inputs(contract, None)
    assert res == {
        "conditioning": {"count": 0, "strategy": None, "concept_text": None, "query_contour_id": None},
        "parameters": {"threshold": 0.5, "mode": "standard"},
    }


def test_validate_and_normalize_inputs_rejects_unknown_top_level_key():
    """Unknown keys in top-level inputs envelope are rejected."""
    contract = InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[],
    )
    with pytest.raises(ValueError, match="Unknown keys in inputs envelope"):
        validate_and_normalize_inputs(contract, {"bogus": 123})


def test_validate_and_normalize_inputs_rejects_unknown_param():
    """Unknown parameter keys are rejected."""
    contract = InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[HyperParameter(key="threshold", label="Threshold", type="float", default_value=0.5)],
    )
    with pytest.raises(ValueError, match="Unknown parameter.*invalid_param"):
        validate_and_normalize_inputs(contract, {"parameters": {"invalid_param": 0.5}})


def test_validate_and_normalize_inputs_type_validation():
    """Parameter types are validated strictly and float integers coerced."""
    contract = InputContract(
        task="dummy",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[
            HyperParameter(key="flag", label="Flag", type="bool", default_value=True),
            HyperParameter(key="count", label="Count", type="int", default_value=5),
            HyperParameter(key="rate", label="Rate", type="float", default_value=1),
            HyperParameter(key="name", label="Name", type="str", default_value="test"),
        ],
    )
    # Int default coerced to float
    res = validate_and_normalize_inputs(contract, {})
    assert res["parameters"]["rate"] == 1.0
    assert isinstance(res["parameters"]["rate"], float)

    # Bool type check
    with pytest.raises(ValueError, match="must be a bool"):
        validate_and_normalize_inputs(contract, {"parameters": {"flag": "true"}})
    with pytest.raises(ValueError, match="must be a bool"):
        validate_and_normalize_inputs(contract, {"parameters": {"flag": 1}})

    # Int type check
    with pytest.raises(ValueError, match="must be an int"):
        validate_and_normalize_inputs(contract, {"parameters": {"count": True}})
    with pytest.raises(ValueError, match="must be an int"):
        validate_and_normalize_inputs(contract, {"parameters": {"count": 2.5}})

    # Float type check
    with pytest.raises(ValueError, match="must be a finite float"):
        validate_and_normalize_inputs(contract, {"parameters": {"rate": "1.0"}})
    with pytest.raises(ValueError, match="must be a finite float"):
        validate_and_normalize_inputs(contract, {"parameters": {"rate": float("nan")}})

    # Str type check
    with pytest.raises(ValueError, match="must be a str"):
        validate_and_normalize_inputs(contract, {"parameters": {"name": 123}})


def test_validate_and_normalize_inputs_bounds_and_options():
    """Min/max bounds and allowed options are strictly enforced."""
    contract = InputContract(
        task="dummy",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[
            HyperParameter(key="threshold", label="Threshold", type="float", default_value=0.5, min_value=0.1, max_value=0.9),
            HyperParameter(key="mode", label="Mode", type="str", default_value="a", options=["a", "b"]),
        ],
    )
    with pytest.raises(ValueError, match="less than min_value"):
        validate_and_normalize_inputs(contract, {"parameters": {"threshold": 0.05}})
    with pytest.raises(ValueError, match="greater than max_value"):
        validate_and_normalize_inputs(contract, {"parameters": {"threshold": 0.95}})
    with pytest.raises(ValueError, match="not in allowed options"):
        validate_and_normalize_inputs(contract, {"parameters": {"mode": "c"}})


def test_validate_and_normalize_inputs_conditioning_none_rejects_irrelevant_keys():
    """Conditioning kind 'none' rejects strategy, concept_text, and count > 0."""
    contract = InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[],
    )
    with pytest.raises(ValueError, match="does not accept a retrieval strategy"):
        validate_and_normalize_inputs(contract, {"conditioning": {"strategy": "global_scene"}})
    with pytest.raises(ValueError, match="does not accept concept_text"):
        validate_and_normalize_inputs(contract, {"conditioning": {"concept_text": "cell"}})
    with pytest.raises(ValueError, match="does not accept count > 0"):
        validate_and_normalize_inputs(contract, {"conditioning": {"count": 5}})


def test_validate_and_normalize_inputs_conditioning_concept_text():
    """Conditioning kind 'concept_text' accepts concept_text and rejects strategy."""
    contract = InputContract(
        task="text-seg",
        conditioning=ConditioningSpec(kind="concept_text", user_selectable_count=False),
        parameters=[],
    )
    res = validate_and_normalize_inputs(contract, {"conditioning": {"concept_text": "cell"}})
    assert res["conditioning"]["concept_text"] == "cell"
    assert res["conditioning"]["count"] == 0
    assert res["conditioning"]["strategy"] is None

    with pytest.raises(ValueError, match="does not accept a retrieval strategy"):
        validate_and_normalize_inputs(contract, {"conditioning": {"strategy": "global_scene"}})


def test_validate_and_normalize_inputs_conditioning_cardinality_bounds():
    """Cardinality count bounds are enforced for counted conditioning."""
    contract = InputContract(
        task="cross-image-suggestion",
        conditioning=ConditioningSpec(
            kind="reference_images",
            unit="image",
            min_units=1,
            max_units=1,
            requires_complete_annotation=True,
            user_selectable_count=False,
        ),
        parameters=[],
    )
    # Default count filled to 1
    res = validate_and_normalize_inputs(contract, {"conditioning": {"strategy": "global_scene"}})
    assert res["conditioning"]["count"] == 1
    assert res["conditioning"]["strategy"] == "global_scene"

    # Below min
    with pytest.raises(ValueError, match="below min_units"):
        validate_and_normalize_inputs(contract, {"conditioning": {"count": 0, "strategy": "global_scene"}})

    # Above max
    with pytest.raises(ValueError, match="exceeds max_units"):
        validate_and_normalize_inputs(contract, {"conditioning": {"count": 2, "strategy": "global_scene"}})


def test_validate_and_normalize_inputs_query_contour_id():
    """query_contour_id is normalized as integer and rejected where irrelevant."""
    # 1. Accepted on reference_images with region-based strategy
    contract_ref = InputContract(
        task="cross-image-suggestion",
        conditioning=ConditioningSpec(kind="reference_images", unit="image", min_units=1, max_units=5),
        parameters=[],
    )
    res = validate_and_normalize_inputs(
        contract_ref,
        {"conditioning": {"strategy": "concept_region", "query_contour_id": 42}},
    )
    assert res["conditioning"]["query_contour_id"] == 42
    assert res["conditioning"]["strategy"] == "concept_region"

    # 2. Rejected on reference_images with global_scene strategy
    with pytest.raises(ValueError, match="does not accept query_contour_id"):
        validate_and_normalize_inputs(
            contract_ref,
            {"conditioning": {"strategy": "global_scene", "query_contour_id": 42}},
        )

    # 3. Rejected on instances conditioning
    contract_inst = InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(kind="instances", unit="instance", min_units=1, max_units=5),
        parameters=[],
    )
    with pytest.raises(ValueError, match="Conditioning kind 'instances' does not accept query_contour_id"):
        validate_and_normalize_inputs(contract_inst, {"conditioning": {"query_contour_id": 42}})

    # 4. Accepted on embeddings with region_mean
    contract_emb_region = InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(
            kind="embeddings", unit="vector", embedding_kinds=["image_cls", "region_mean"], user_selectable_count=False
        ),
        parameters=[],
    )
    res_emb = validate_and_normalize_inputs(contract_emb_region, {"conditioning": {"query_contour_id": 99}})
    assert res_emb["conditioning"]["query_contour_id"] == 99

    # 5. Rejected on embeddings with only image_cls
    contract_emb_image = InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(kind="embeddings", unit="vector", embedding_kinds=["image_cls"], user_selectable_count=False),
        parameters=[],
    )
    with pytest.raises(ValueError, match="Embedding conditioning without 'region_mean' does not accept query_contour_id"):
        validate_and_normalize_inputs(contract_emb_image, {"conditioning": {"query_contour_id": 99}})

    # 6. Non-integer validation
    with pytest.raises(ValueError, match="query_contour_id must be an integer"):
        validate_and_normalize_inputs(
            contract_ref,
            {"conditioning": {"strategy": "concept_region", "query_contour_id": "invalid"}},
        )

    # 7. Rejected on 'none' kind
    contract_none = InputContract(
        task="instance-segmentation",
        conditioning=ConditioningSpec(kind="none", user_selectable_count=False),
        parameters=[],
    )
    with pytest.raises(ValueError, match="does not accept query_contour_id"):
        validate_and_normalize_inputs(contract_none, {"conditioning": {"query_contour_id": 42}})

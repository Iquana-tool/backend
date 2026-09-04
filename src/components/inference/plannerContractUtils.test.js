import {
  getEffectiveContract,
  getDefaultParameters,
  getDefaultConditioning,
  initStep,
  updateStepParameter,
  updateStepConditioning,
  LEGACY_TASK_DEFAULTS,
} from "./plannerContractUtils";

describe("plannerContractUtils", () => {
  const sampleStrategies = [
    { key: "global_scene", label: "Global Scene", available: true },
    { key: "region_mean", label: "Region Mean", available: true },
  ];

  it("resolves declared contract or falls back to legacy task defaults", () => {
    const modelWithContract = {
      registry_key: "sam3-cross",
      task: "cross-image-suggestion",
      input_contract: {
        schema_version: 1,
        task: "cross-image-suggestion",
        conditioning: { kind: "reference_images", unit: "image", min_units: 1, max_units: 1, user_selectable_count: false },
        parameters: [{ key: "threshold", default_value: 0.5 }],
      },
    };
    expect(getEffectiveContract(modelWithContract)).toBe(modelWithContract.input_contract);

    const legacyCross = { registry_key: "legacy-cross", task: "cross-image-suggestion" };
    expect(getEffectiveContract(legacyCross)).toEqual(LEGACY_TASK_DEFAULTS["cross-image-suggestion"]);
    expect(LEGACY_TASK_DEFAULTS["cross-image-suggestion"].conditioning.kind).toBe("instances");
    expect(LEGACY_TASK_DEFAULTS["cross-image-suggestion"].conditioning.unit).toBe("instance");
    expect(LEGACY_TASK_DEFAULTS["cross-image-suggestion"].conditioning.max_units).toBe(32);
    expect(LEGACY_TASK_DEFAULTS["cross-image-suggestion"].parameters).toEqual([]);
  });

  it("extracts declared default parameters", () => {
    const contract = {
      parameters: [
        { key: "threshold", default_value: 0.5 },
        { key: "mask_threshold", default_value: 0.0 },
        { key: "min_target_frac", default_value: 0.0001 },
      ],
    };
    expect(getDefaultParameters(contract)).toEqual({
      threshold: 0.5,
      mask_threshold: 0.0,
      min_target_frac: 0.0001,
    });
  });

  it("builds conditioning for kind=none with empty payload", () => {
    const contract = { conditioning: { kind: "none", user_selectable_count: false } };
    expect(getDefaultConditioning(contract, sampleStrategies)).toEqual({});
  });

  it("builds conditioning for kind=concept_text using label name", () => {
    const contract = { conditioning: { kind: "concept_text", user_selectable_count: false } };
    const label = { id: 10, name: "cell_membrane" };
    expect(getDefaultConditioning(contract, sampleStrategies, label)).toEqual({
      concept_text: "cell_membrane",
    });
  });

  it("builds conditioning for reference_images with single image and strategy", () => {
    const contract = {
      conditioning: {
        kind: "reference_images",
        unit: "image",
        min_units: 1,
        max_units: 1,
        user_selectable_count: false,
      },
    };
    expect(getDefaultConditioning(contract, sampleStrategies)).toEqual({
      count: 1,
      strategy: "global_scene",
    });
  });

  it("initializes the legacy cross-image fallback with a retrieval strategy", () => {
    const step = initStep(
      { id: 11, name: "cell" },
      { registry_key: "legacy-cross", task: "cross-image-suggestion" },
      sampleStrategies
    );

    expect(step.inputs.conditioning).toEqual({
      count: 5,
      strategy: "global_scene",
    });
    expect(step.retrieval_strategy).toBe("global_scene");
  });

  it("builds conditioning for user-selectable count clamped to bounds", () => {
    const contract = {
      task: "cross-image-suggestion",
      conditioning: {
        kind: "reference_images",
        unit: "image",
        min_units: 2,
        max_units: 4,
        user_selectable_count: true,
      },
    };
    expect(getDefaultConditioning(contract, sampleStrategies)).toEqual({
      count: 4, // 5 clamped to max_units 4
      strategy: "global_scene",
    });
  });

  it("does not inject count or retrieval strategy for within-image instance suggestion with arbitrary exemplars", () => {
    const contract = {
      task: "instance-suggestion",
      conditioning: {
        kind: "instances",
        unit: "instance",
        min_units: 1,
        max_units: null,
        user_selectable_count: false,
      },
    };
    const conditioning = getDefaultConditioning(contract, sampleStrategies);
    expect(conditioning.count).toBeUndefined();
    expect(conditioning.strategy).toBeUndefined();
  });

  it("initializes a complete canonical step with legacy compatibility fields", () => {
    const label = { id: 42, name: "mitochondria" };
    const model = {
      registry_key: "sam3-incontext",
      task: "cross-image-suggestion",
      input_contract: {
        task: "cross-image-suggestion",
        conditioning: {
          kind: "reference_images",
          unit: "image",
          min_units: 1,
          max_units: 1,
          user_selectable_count: false,
        },
        parameters: [
          { key: "threshold", default_value: 0.35 },
          { key: "mask_threshold", default_value: -1.0 },
        ],
      },
    };

    const step = initStep(label, model, sampleStrategies);
    expect(step).toEqual({
      label_id: 42,
      model_registry_key: "sam3-incontext",
      task: "cross-image-suggestion",
      inputs: {
        conditioning: {
          count: 1,
          strategy: "global_scene",
        },
        parameters: {
          threshold: 0.35,
          mask_threshold: -1.0,
        },
      },
      min_confidence: 0,
      retrieval_strategy: "global_scene",
      top_k: 1,
    });
  });

  it("keeps legacy fallback confidence outside the declared parameter map", () => {
    const step = initStep(
      { id: 7, name: "cell" },
      {
        registry_key: "legacy-m2f",
        task: "instance-segmentation",
        provenance: "legacy_default",
        input_contract: LEGACY_TASK_DEFAULTS["instance-segmentation"],
      },
      sampleStrategies
    );

    expect(step.min_confidence).toBe(0);
    expect(step.inputs.parameters).toEqual({});
    expect(step.inputs.parameters).not.toHaveProperty("min_confidence");
  });

  it("updates parameters without mutating gateway min_confidence", () => {
    const initialStep = {
      label_id: 1,
      inputs: { parameters: { threshold: 0.5, mask_threshold: 0.0 } },
      min_confidence: 0.5,
    };

    const updated = updateStepParameter(initialStep, "threshold", 0.8);
    expect(updated.inputs.parameters.threshold).toBe(0.8);
    expect(updated.min_confidence).toBe(0.5);

    const updatedOther = updateStepParameter(updated, "mask_threshold", 1.5);
    expect(updatedOther.inputs.parameters.mask_threshold).toBe(1.5);
    expect(updatedOther.min_confidence).toBe(0.5);
  });

  it("updates conditioning and syncs legacy top_k and retrieval_strategy", () => {
    const initialStep = {
      label_id: 1,
      inputs: { conditioning: { count: 1, strategy: "global_scene" } },
      top_k: 1,
      retrieval_strategy: "global_scene",
    };

    const updatedCount = updateStepConditioning(initialStep, "count", 3);
    expect(updatedCount.inputs.conditioning.count).toBe(3);
    expect(updatedCount.top_k).toBe(3);

    const updatedStrategy = updateStepConditioning(updatedCount, "strategy", "region_mean");
    expect(updatedStrategy.inputs.conditioning.strategy).toBe("region_mean");
    expect(updatedStrategy.retrieval_strategy).toBe("region_mean");
  });
});

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import LabelModelPlanner, { groupLabelsByLevel, modelsForLabel } from "./LabelModelPlanner";

describe("LabelModelPlanner", () => {
  const labelsById = {
    1: { id: 1, name: "cell", parent_id: null },
    2: { id: 2, name: "nucleus", parent_id: 1 },
  };

  const strategies = [
    { key: "global_scene", label: "Global Scene", available: true },
    { key: "region_mean", label: "Region Mean", available: true },
  ];

  const sam3Model = {
    registry_key: "sam3-incontext",
    name: "SAM 3 (In-context)",
    task: "cross-image-suggestion",
    provenance: "declared",
    label_ids: [],
    input_contract: {
      schema_version: 1,
      task: "cross-image-suggestion",
      conditioning: {
        kind: "reference_images",
        unit: "image",
        min_units: 1,
        max_units: 1,
        requires_complete_annotation: true,
        user_selectable_count: false,
      },
      parameters: [
        {
          key: "threshold",
          label: "Detection sensitivity",
          type: "float",
          default_value: 0.3,
          min_value: 0.0,
          max_value: 1.0,
          step: 0.05,
        },
        {
          key: "mask_threshold",
          label: "Mask threshold",
          type: "float",
          default_value: 0.5,
          min_value: 0.0,
          max_value: 1.0,
          step: 0.05,
        },
        {
          key: "min_target_frac",
          label: "Target overlap",
          type: "float",
          default_value: 0.5,
          min_value: 0.0,
          max_value: 1.0,
          step: 0.05,
        },
      ],
      notes: "SAM 3 uses 1 full reference image and prompts iteratively.",
    },
  };

  const mask2formerModel = {
    registry_key: "mask2former-model",
    name: "Mask2Former",
    task: "instance-segmentation",
    provenance: "declared",
    label_ids: [1, 2],
    input_contract: {
      schema_version: 1,
      task: "instance-segmentation",
      conditioning: {
        kind: "none",
        unit: null,
        min_units: 0,
        max_units: null,
        user_selectable_count: false,
      },
      parameters: [
        {
          key: "threshold",
          label: "Score Threshold",
          type: "float",
          default_value: 0.5,
          min_value: 0.0,
          max_value: 1.0,
          step: 0.05,
        },
      ],
    },
  };

  const textModel = {
    registry_key: "clip-seg-model",
    name: "CLIP-Seg Prompted",
    task: "instance-segmentation",
    provenance: "declared",
    label_ids: [],
    input_contract: {
      schema_version: 1,
      task: "instance-segmentation",
      conditioning: {
        kind: "concept_text",
        unit: null,
        min_units: 0,
        max_units: null,
        user_selectable_count: false,
      },
      parameters: [
        {
          key: "threshold",
          label: "Score Threshold",
          type: "float",
          default_value: 0.3,
          min_value: 0.0,
          max_value: 1.0,
          step: 0.05,
        },
      ],
    },
  };

  const models = [sam3Model, mask2formerModel, textModel];

  const legacyModel = {
    registry_key: "legacy-mask2former",
    name: "Legacy Mask2Former",
    task: "instance-segmentation",
    provenance: "legacy_default",
    label_ids: [],
    input_contract: {
      schema_version: 1,
      task: "instance-segmentation",
      conditioning: {
        kind: "none",
        user_selectable_count: false,
      },
      parameters: [],
    },
  };

  const renderPlanner = ({ models: plannerModels, steps, onChange, ...props }) =>
    render(
      <LabelModelPlanner
        labelsById={labelsById}
        models={plannerModels}
        strategies={strategies}
        steps={steps}
        onChange={onChange}
        {...props}
      />
    );

  it("groups labels by hierarchy depth correctly", () => {
    const levels = groupLabelsByLevel(labelsById);
    expect(levels).toHaveLength(2);
    expect(levels[0].level).toBe(0);
    expect(levels[0].labels[0].name).toBe("cell");
    expect(levels[1].level).toBe(1);
    expect(levels[1].labels[0].name).toBe("nucleus");
    expect(levels[1].labels[0].parentName).toBe("cell");
  });

  it("filters models applicable to a label", () => {
    expect(modelsForLabel(models, 1)).toHaveLength(3);
    const specialist = {
      registry_key: "cell-only",
      task: "instance-segmentation",
      label_ids: [1],
    };
    expect(modelsForLabel([specialist], 2)).toHaveLength(0);
    expect(modelsForLabel([specialist], 1)).toHaveLength(1);
  });

  it("renders SAM 3 with strategy selector, declared parameters, notes, and NO count control", () => {
    const onChange = jest.fn();
    const step = {
      label_id: 1,
      model_registry_key: "sam3-incontext",
      task: "cross-image-suggestion",
      inputs: {
        conditioning: { count: 1, strategy: "global_scene" },
        parameters: { threshold: 0.3, mask_threshold: 0.5, min_target_frac: 0.5 },
      },
      min_confidence: 0.3,
      retrieval_strategy: "global_scene",
      top_k: 1,
    };

    renderPlanner({ models, steps: [step], onChange });

    // Strategy selector is present
    expect(screen.getByRole("combobox", { name: /retrieval strategy for cell/i })).toBeInTheDocument();

    // Count input is NOT rendered because user_selectable_count is false
    expect(screen.queryByRole("spinbutton", { name: /images for cell/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("spinbutton", { name: /top-k/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("spinbutton", { name: /min\. confidence for cell/i })).not.toBeInTheDocument();

    // Notes are rendered
    expect(screen.getByText(/SAM 3 uses 1 full reference image/i)).toBeInTheDocument();

    // All 3 declared parameters are rendered with scoped IDs
    const thresholdSlider = screen.getByRole("slider", { name: /detection sensitivity/i });
    expect(thresholdSlider).toBeInTheDocument();
    expect(thresholdSlider.id).toBe("label-1-param-threshold");
    expect(screen.getByRole("slider", { name: /mask threshold/i }).id).toBe("label-1-param-mask_threshold");
    expect(screen.getByRole("slider", { name: /target overlap/i }).id).toBe("label-1-param-min_target_frac");
  });

  it("renders unbounded count input without arbitrary max ceiling", () => {
    const unboundedModel = {
      registry_key: "unbounded-instances",
      name: "Unbounded Instances Model",
      task: "cross-image-suggestion",
      label_ids: [],
      input_contract: {
        schema_version: 1,
        task: "cross-image-suggestion",
        conditioning: {
          kind: "instances",
          unit: "instance",
          min_units: 1,
          max_units: null,
          user_selectable_count: true,
        },
        parameters: [],
      },
    };

    const step = {
      label_id: 1,
      model_registry_key: "unbounded-instances",
      task: "cross-image-suggestion",
      inputs: {
        conditioning: { count: 5 },
        parameters: {},
      },
      top_k: 5,
    };

    renderPlanner({ models: [unboundedModel], steps: [step], onChange: jest.fn() });

    const countInput = screen.getByRole("spinbutton", { name: /instances for cell/i });
    expect(countInput).toBeInTheDocument();
    expect(countInput.id).toBe("label-1-count");
    expect(countInput).not.toHaveAttribute("max");
    expect(screen.queryByRole("combobox", { name: /retrieval strategy for cell/i })).not.toBeInTheDocument();
  });

  it("renders and updates strategy for the legacy cross-image fallback", () => {
    const onChange = jest.fn();
    const legacyCrossImageModel = {
      registry_key: "legacy-cross-image",
      name: "Legacy Cross-Image Suggestion",
      task: "cross-image-suggestion",
      provenance: "legacy_default",
      label_ids: [],
    };
    const step = {
      label_id: 1,
      model_registry_key: "legacy-cross-image",
      task: "cross-image-suggestion",
      inputs: {
        conditioning: { count: 5, strategy: "global_scene" },
        parameters: {},
      },
      retrieval_strategy: "global_scene",
      top_k: 5,
    };

    renderPlanner({ models: [legacyCrossImageModel], steps: [step], onChange });

    const strategySelect = screen.getByRole("combobox", {
      name: /retrieval strategy for cell/i,
    });
    expect(strategySelect).toHaveValue("global_scene");

    fireEvent.change(strategySelect, { target: { value: "region_mean" } });

    expect(onChange).toHaveBeenCalledWith(
      1,
      expect.objectContaining({
        retrieval_strategy: "region_mean",
        inputs: expect.objectContaining({
          conditioning: expect.objectContaining({ strategy: "region_mean" }),
        }),
      })
    );
  });

  it("does not show a strategy selector for ordinary instance conditioning", () => {
    const ordinaryInstanceModel = {
      registry_key: "ordinary-instance-suggestion",
      name: "Ordinary Instance Suggestion",
      task: "instance-suggestion",
      label_ids: [],
      input_contract: {
        schema_version: 1,
        task: "instance-suggestion",
        conditioning: {
          kind: "instances",
          unit: "instance",
          min_units: 1,
          max_units: 4,
          user_selectable_count: true,
        },
        parameters: [],
      },
    };
    const step = {
      label_id: 1,
      model_registry_key: "ordinary-instance-suggestion",
      task: "instance-suggestion",
      inputs: {
        conditioning: { count: 2 },
        parameters: {},
      },
    };

    renderPlanner({
      models: [ordinaryInstanceModel],
      steps: [step],
      onChange: jest.fn(),
      allowedTasks: ["instance-suggestion"],
    });

    expect(screen.getByRole("spinbutton", { name: /instances for cell/i })).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: /retrieval strategy for cell/i })).not.toBeInTheDocument();
  });

  it("renders Mask2Former with threshold and without exemplar controls", () => {
    const onChange = jest.fn();
    const step = {
      label_id: 1,
      model_registry_key: "mask2former-model",
      task: "instance-segmentation",
      inputs: {
        conditioning: {},
        parameters: { threshold: 0.5 },
      },
      min_confidence: 0.5,
      retrieval_strategy: null,
      top_k: 5,
    };

    renderPlanner({ models, steps: [step], onChange });

    // No exemplar/strategy selector
    expect(screen.queryByRole("combobox", { name: /retrieval strategy/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("spinbutton", { name: /images/i })).not.toBeInTheDocument();

    // Threshold parameter is rendered
    expect(screen.getByRole("slider", { name: /score threshold/i })).toBeInTheDocument();
  });

  it("renders the gateway confidence control only for legacy fallback models", () => {
    const onChange = jest.fn();
    const step = {
      label_id: 1,
      model_registry_key: "legacy-mask2former",
      task: "instance-segmentation",
      inputs: {
        conditioning: {},
        parameters: {},
      },
      min_confidence: 0.2,
    };

    renderPlanner({ models: [legacyModel], steps: [step], onChange });

    const confidenceInput = screen.getByRole("spinbutton", {
      name: /min\. confidence for cell/i,
    });
    expect(confidenceInput).toHaveValue(0.2);
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();

    fireEvent.change(confidenceInput, { target: { value: "0.65" } });

    expect(onChange).toHaveBeenCalledWith(
      1,
      expect.objectContaining({
        min_confidence: 0.65,
        inputs: expect.objectContaining({ parameters: {} }),
      })
    );
  });

  it("renders concept_text prompt input for text-conditioned models", () => {
    const onChange = jest.fn();
    const step = {
      label_id: 1,
      model_registry_key: "clip-seg-model",
      task: "instance-segmentation",
      inputs: {
        conditioning: { concept_text: "cell" },
        parameters: { threshold: 0.3 },
      },
      min_confidence: 0.3,
      retrieval_strategy: null,
      top_k: 5,
    };

    renderPlanner({ models, steps: [step], onChange });

    // Text prompt input is rendered
    const textInput = screen.getByRole("textbox", { name: /prompt for cell/i });
    expect(textInput).toBeInTheDocument();
    expect(textInput.value).toBe("cell");

    fireEvent.change(textInput, { target: { value: "fluorescent cell membrane" } });
    expect(onChange).toHaveBeenCalledWith(
      1,
      expect.objectContaining({
        inputs: expect.objectContaining({
          conditioning: expect.objectContaining({
            concept_text: "fluorescent cell membrane",
          }),
        }),
      })
    );
  });

  it("initializes defaults and clears stale inputs when selecting a model", () => {
    const onChange = jest.fn();

    renderPlanner({ models, steps: [], onChange });

    const modelSelect = screen.getByRole("combobox", { name: /model for cell/i });
    fireEvent.change(modelSelect, { target: { value: "cross-image-suggestion::sam3-incontext" } });

    expect(onChange).toHaveBeenCalledWith(
      1,
      expect.objectContaining({
        label_id: 1,
        model_registry_key: "sam3-incontext",
        task: "cross-image-suggestion",
        inputs: {
          conditioning: {
            count: 1,
            strategy: "global_scene",
          },
          parameters: {
            threshold: 0.3,
            mask_threshold: 0.5,
            min_target_frac: 0.5,
          },
        },
      })
    );
  });

  it("handles parameter changes dynamically", () => {
    const onChange = jest.fn();
    const step = {
      label_id: 1,
      model_registry_key: "sam3-incontext",
      task: "cross-image-suggestion",
      inputs: {
        conditioning: { count: 1, strategy: "global_scene" },
        parameters: { threshold: 0.3, mask_threshold: 0.5, min_target_frac: 0.5 },
      },
      min_confidence: 0.3,
      retrieval_strategy: "global_scene",
      top_k: 1,
    };

    renderPlanner({ models, steps: [step], onChange });

    const slider = screen.getByRole("slider", { name: /detection sensitivity/i });
    fireEvent.change(slider, { target: { value: "0.85" } });

    expect(onChange).toHaveBeenCalledWith(
      1,
      expect.objectContaining({
        inputs: expect.objectContaining({
          parameters: expect.objectContaining({
            threshold: 0.85,
          }),
        }),
      })
    );
  });

  it("excludes interactive models from the batch planner dropdown", () => {
    const mixedModels = [
      {
        registry_key: "sam2-prompted",
        name: "SAM 2 Prompted",
        task: "prompted-segmentation",
        label_ids: [],
      },
      {
        registry_key: "sam3-intra",
        name: "SAM 3 Intra",
        task: "instance-suggestion",
        label_ids: [],
      },
      {
        registry_key: "m2f-batch",
        name: "Mask2Former Batch",
        task: "instance-segmentation",
        label_ids: [],
      },
      {
        registry_key: "sam3-cross",
        name: "SAM 3 Cross Exemplar",
        task: "cross-image-suggestion",
        label_ids: [],
      },
    ];

    renderPlanner({ models: mixedModels, steps: [], onChange: jest.fn() });

    const cellSelect = screen.getByLabelText("Model for cell");
    const optionTexts = Array.from(cellSelect.options).map((opt) => opt.textContent);

    expect(optionTexts).toContain("Mask2Former Batch");
    expect(optionTexts.some((t) => t.includes("SAM 3 Cross Exemplar"))).toBe(true);
    expect(optionTexts).not.toContain("SAM 2 Prompted");
    expect(optionTexts).not.toContain("SAM 3 Intra");
  });
});

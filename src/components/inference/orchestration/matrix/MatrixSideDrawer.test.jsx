import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import MatrixSideDrawer from "./MatrixSideDrawer";

describe("MatrixSideDrawer (Design B)", () => {
  const labelsById = {
    1: { id: 1, name: "Coral fragment", color: "#2dd4bf" },
  };

  const catalog = {
    models: [
      {
        registry_key: "coral-ft-v3",
        name: "Coral-FT v3",
        task: "prompted-segmentation",
        label_ids: [1],
        latency_badge: "fast",
        model_size: "38 MB",
        trained_on_dataset: true,
        input_contract: {
          parameters: [
            { key: "threshold", label: "Mask threshold", type: "float", min_value: 0, max_value: 1, step: 0.05, default_value: 0.62 },
            { key: "refine_edges", label: "Refine edges", type: "bool", default_value: true },
            { key: "quality", label: "Quality preset", type: "str", options: ["low", "medium", "high"], default_value: "high" },
          ],
        },
      },
      {
        registry_key: "sam21-small",
        name: "SAM 2.1 Small",
        task: "prompted-segmentation",
        label_ids: [],
        latency_badge: "balanced",
        model_size: "46 MB",
        trained_on_dataset: false,
      },
    ],
    retrieval_strategies: [],
  };

  const createDrawerProps = (overrides = {}) => ({
    isOpen: true,
    onClose: vi.fn(),
    target: { task: "prompted-segmentation", labelId: 1 },
    labelsById,
    catalog,
    draftBindings: [],
    onSaveRoute: vi.fn(),
    ...overrides,
  });

  const renderMatrixSideDrawer = (overrides = {}) =>
    render(<MatrixSideDrawer {...createDrawerProps(overrides)} />);

  it("renders model list in Stage 1 and filters by search, showing trained badge", () => {
    renderMatrixSideDrawer({ onUnbindRoute: vi.fn() });

    expect(screen.getByText("Bind a Model")).toBeInTheDocument();
    expect(screen.getByText("Coral fragment")).toBeInTheDocument();
    expect(screen.getByText("Coral-FT v3")).toBeInTheDocument();
    expect(screen.getByText("SAM 2.1 Small")).toBeInTheDocument();
    expect(screen.getByText("trained here")).toBeInTheDocument();
    expect(screen.getByText(/Leaving this cell unbound falls back to the task default/i)).toBeInTheDocument();

    // Search filter
    const searchInput = screen.getByPlaceholderText(/Search 2 compatible models/i);
    fireEvent.change(searchInput, { target: { value: "coral" } });

    expect(screen.getByText("Coral-FT v3")).toBeInTheDocument();
    expect(screen.queryByText("SAM 2.1 Small")).not.toBeInTheDocument();
  });

  it("transitions to Stage 2 when clicking configure and renders contract hyperparameter inputs", () => {
    const handleSaveRoute = vi.fn();

    renderMatrixSideDrawer({ onSaveRoute: handleSaveRoute, onUnbindRoute: vi.fn() });

    const configureBtns = screen.getAllByRole("button", { name: /configure/i });
    fireEvent.click(configureBtns[0]);

    // Stage 2 headers
    expect(screen.getByText(/models · INPUTS/i)).toBeInTheDocument();
    expect(screen.getByText("Mask threshold")).toBeInTheDocument();
    expect(screen.getByText("Refine edges")).toBeInTheDocument();
    expect(screen.getByText("Quality preset")).toBeInTheDocument();

    // Verify boolean checkbox
    const boolInput = screen.getByRole("checkbox", { name: /Refine edges/i });
    expect(boolInput).toBeChecked();
    fireEvent.click(boolInput);
    expect(boolInput).not.toBeChecked();

    // Verify select dropdown
    const selectDropdown = screen.getByRole("combobox", { name: /Quality preset/i });
    expect(selectDropdown).toHaveValue("high");
    fireEvent.change(selectDropdown, { target: { value: "medium" } });
    expect(selectDropdown).toHaveValue("medium");

    // Apply route
    const applyBtn = screen.getByRole("button", { name: /apply route/i });
    fireEvent.click(applyBtn);

    expect(handleSaveRoute).toHaveBeenCalledWith(
      "prompted-segmentation",
      1,
      expect.objectContaining({
        model_registry_key: "coral-ft-v3",
        task: "prompted-segmentation",
        label_id: 1,
        inputs: expect.objectContaining({
          parameters: expect.objectContaining({
            threshold: 0.62,
            refine_edges: false,
            quality: "medium",
          }),
        }),
      })
    );
  });

  it("renders canonical backend badges and handles Escape key press to close dialog", () => {
    const handleClose = vi.fn();
    const modelWithBadgesCatalog = {
      models: [
        {
          registry_key: "canonical-model",
          name: "Canonical Model",
          task: "prompted-segmentation",
          badges: ["fast inference", "120 MB", "GPU accelerated"],
          trained_on_dataset: true,
        },
      ],
      retrieval_strategies: [],
    };

    renderMatrixSideDrawer({ onClose: handleClose, catalog: modelWithBadgesCatalog });

    // Verify dialog role and aria attributes
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAttribute("aria-labelledby", "matrix-drawer-title");

    // Verify canonical badges
    expect(screen.getByText("fast inference")).toBeInTheDocument();
    expect(screen.getByText("120 MB")).toBeInTheDocument();
    expect(screen.getByText("GPU accelerated")).toBeInTheDocument();

    // Verify Escape key closes dialog
    fireEvent.keyDown(window, { key: "Escape" });
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it("renders the conditioning controls declared by each model contract", () => {
    const conditioningCatalog = {
      models: [
        {
          registry_key: "sam3-instances",
          name: "SAM 3 Instances",
          task: "instance-suggestion",
          input_contract: {
            conditioning: { kind: "instances", user_selectable_count: false, unit: "instance" },
          },
        },
        {
          registry_key: "sam3-cross",
          name: "SAM 3 Cross",
          task: "cross-image-suggestion",
          input_contract: {
            conditioning: { kind: "reference_images", user_selectable_count: true, unit: "image" },
          },
        },
        {
          registry_key: "clip-concept",
          name: "CLIP Concept",
          task: "instance-segmentation",
          input_contract: {
            conditioning: { kind: "concept_text" },
          },
        },
        {
          registry_key: "sam2-prompted",
          name: "SAM 2 Prompted",
          task: "prompted-segmentation",
          input_contract: {
            conditioning: { kind: "none" },
          },
        },
      ],
      retrieval_strategies: [{ key: "global_scene", label: "Global Scene", available: true }],
    };

    // 1. Within-image suggestion (SAM 3 Instances) -> Selected objects in this image, NO strategy selector
    const { rerender } = renderMatrixSideDrawer({
      catalog: conditioningCatalog,
      target: { task: "instance-suggestion", labelId: 1 },
    });

    const configureBtns = screen.getAllByRole("button", { name: /configure/i });
    fireEvent.click(configureBtns[0]);

    expect(screen.getByText("Selected objects in this image")).toBeInTheDocument();
    expect(screen.queryByText("Prompt concept")).not.toBeInTheDocument();
    expect(screen.queryByText("Retrieval strategy")).not.toBeInTheDocument();
    expect(document.getElementById("matrix-drawer-title")).toHaveTextContent("SAM 3 Instances");

    // 2. Cross-image suggestion (SAM 3 Cross) -> Reference examples from other images & strategy selector & count buttons
    rerender(
      <MatrixSideDrawer
        {...createDrawerProps({
          catalog: conditioningCatalog,
          target: { task: "cross-image-suggestion", labelId: 1 },
        })}
      />
    );

    const crossConfigBtns = screen.getAllByRole("button", { name: /configure/i });
    fireEvent.click(crossConfigBtns[0]);

    expect(screen.getByText("Reference examples from other images")).toBeInTheDocument();
    expect(screen.getByText("Retrieval strategy")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "5" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "10" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "20" })).toBeInTheDocument();

    // 3. Concept text is rendered and defaults to the displayed label name.
    rerender(
      <MatrixSideDrawer
        {...createDrawerProps({
          catalog: conditioningCatalog,
          target: { task: "instance-segmentation", labelId: 1 },
        })}
      />
    );

    const instConfigBtns = screen.getAllByRole("button", { name: /configure/i });
    fireEvent.click(instConfigBtns[0]); // clip-concept

    expect(screen.getByText("Prompt concept")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: /prompt concept for coral fragment/i })).toHaveValue("Coral fragment");
    expect(screen.getByText("Conditioning")).toBeInTheDocument();

    // 4. Kind === none (SAM 2 Prompted) -> Conditioning section hidden
    rerender(
      <MatrixSideDrawer
        {...createDrawerProps({
          catalog: conditioningCatalog,
          target: { task: "prompted-segmentation", labelId: 1 },
        })}
      />
    );

    const promptConfigBtns = screen.getAllByRole("button", { name: /configure/i });
    fireEvent.click(promptConfigBtns[0]); // sam2-prompted

    expect(screen.queryByText("Conditioning")).not.toBeInTheDocument();
  });

  it("renders, edits, and persists concept_text with the displayed label name default", () => {
    const handleSaveRoute = vi.fn();
    const conceptModel = {
      registry_key: "clip-concept",
      name: "CLIP Concept",
      task: "instance-segmentation",
      label_ids: [],
      input_contract: {
        conditioning: {
          kind: "concept_text",
          user_selectable_count: false,
        },
        parameters: [
          {
            key: "threshold",
            label: "Score threshold",
            type: "float",
            default_value: 0.3,
            min_value: 0,
            max_value: 1,
            step: 0.05,
          },
        ],
      },
    };

    renderMatrixSideDrawer({
      catalog: { models: [conceptModel], retrieval_strategies: [] },
      target: { task: "instance-segmentation", labelId: 1 },
      draftBindings: [
        {
          task: "instance-segmentation",
          label_id: 1,
          model_registry_key: "clip-concept",
          inputs: {
            conditioning: {},
            parameters: { threshold: 0.55, obsolete_parameter: true },
          },
        },
      ],
      onSaveRoute: handleSaveRoute,
    });

    fireEvent.click(screen.getByRole("button", { name: /configure/i }));

    const conceptInput = screen.getByRole("textbox", {
      name: /prompt concept for coral fragment/i,
    });
    expect(conceptInput).toHaveValue("Coral fragment");

    fireEvent.change(conceptInput, { target: { value: "reef fragment" } });
    fireEvent.click(screen.getByRole("button", { name: /apply route/i }));

    expect(handleSaveRoute).toHaveBeenCalledWith(
      "instance-segmentation",
      1,
      expect.objectContaining({
        inputs: {
          conditioning: { concept_text: "reef fragment" },
          parameters: { threshold: 0.55 },
        },
      })
    );
  });

  it("blocks a stale existing binding until a compatible current model repairs it", () => {
    const handleSaveRoute = vi.fn();
    renderMatrixSideDrawer({
      draftBindings: [
        {
          task: "prompted-segmentation",
          label_id: 1,
          model_registry_key: "removed-from-catalog",
        },
      ],
      catalog: {
        models: [
          {
            registry_key: "removed-from-catalog",
            name: "Wrong task model",
            task: "instance-suggestion",
          },
          ...catalog.models,
        ],
        retrieval_strategies: [],
      },
      onSaveRoute: handleSaveRoute,
    });

    expect(screen.getByRole("alert")).toHaveTextContent(
      /stale or missing.*compatible current model/i
    );
    expect(screen.queryByRole("button", { name: /save route/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: /configure/i })[0]);
    fireEvent.click(screen.getByRole("button", { name: /apply route/i }));

    expect(handleSaveRoute).toHaveBeenCalledWith(
      "prompted-segmentation",
      1,
      expect.objectContaining({ model_registry_key: "coral-ft-v3" })
    );
  });

  it("blocks a class-incompatible existing binding until a compatible model is selected", () => {
    const handleSaveRoute = vi.fn();
    const incompatibleModel = {
      registry_key: "coral-other-label",
      name: "Other-label specialist",
      task: "prompted-segmentation",
      label_ids: [2],
    };
    const compatibleModel = {
      registry_key: "prompted-generic",
      name: "Prompted Generic",
      task: "prompted-segmentation",
      label_ids: [],
    };

    renderMatrixSideDrawer({
      catalog: {
        models: [incompatibleModel, compatibleModel],
        retrieval_strategies: [],
      },
      draftBindings: [
        {
          task: "prompted-segmentation",
          label_id: 1,
          model_registry_key: "coral-other-label",
        },
      ],
      onSaveRoute: handleSaveRoute,
    });

    expect(screen.getByRole("alert")).toHaveTextContent(
      /incompatible with this label.*compatible model/i
    );
    expect(screen.queryByRole("button", { name: /save route/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /configure/i }));
    fireEvent.click(screen.getByRole("button", { name: /apply route/i }));

    expect(handleSaveRoute).toHaveBeenCalledWith(
      "prompted-segmentation",
      1,
      expect.objectContaining({ model_registry_key: "prompted-generic" })
    );
  });

  it("clamps count presets and manual values to declared bounds before applying", () => {
    const handleSaveRoute = vi.fn();
    const boundedModel = {
      registry_key: "bounded-cross-image",
      name: "Bounded Cross Image",
      task: "cross-image-suggestion",
      label_ids: [],
      input_contract: {
        conditioning: {
          kind: "reference_images",
          unit: "image",
          min_units: 8,
          max_units: 12,
          user_selectable_count: true,
        },
        parameters: [],
      },
    };

    renderMatrixSideDrawer({
      target: { task: "cross-image-suggestion", labelId: 1 },
      catalog: {
        models: [boundedModel],
        retrieval_strategies: [
          { key: "global_scene", label: "Global Scene", available: true },
        ],
      },
      onSaveRoute: handleSaveRoute,
    });
    fireEvent.click(screen.getByRole("button", { name: /configure/i }));

    const countInput = screen.getByRole("spinbutton", { name: /exemplars count/i });
    expect(countInput).toHaveAttribute("min", "8");
    expect(countInput).toHaveAttribute("max", "12");
    expect(countInput).toHaveValue(8);

    expect(screen.queryByRole("button", { name: "5" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "20" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "10" }));
    expect(countInput).toHaveValue(10);
    fireEvent.change(countInput, { target: { value: "10" } });
    expect(countInput).toHaveValue(10);
    fireEvent.change(countInput, { target: { value: "999" } });
    expect(countInput).toHaveValue(12);

    fireEvent.click(screen.getByRole("button", { name: /apply route/i }));

    expect(handleSaveRoute).toHaveBeenCalledWith(
      "cross-image-suggestion",
      1,
      expect.objectContaining({
        inputs: expect.objectContaining({
          conditioning: expect.objectContaining({
            count: 12,
            strategy: "global_scene",
          }),
        }),
      })
    );
  });

  it("keeps an explicitly unbounded count uncapped for manual input", () => {
    const handleSaveRoute = vi.fn();
    const unboundedModel = {
      registry_key: "unbounded-instances",
      name: "Unbounded Instances",
      task: "instance-suggestion",
      label_ids: [],
      input_contract: {
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

    renderMatrixSideDrawer({
      target: { task: "instance-suggestion", labelId: 1 },
      catalog: { models: [unboundedModel], retrieval_strategies: [] },
      onSaveRoute: handleSaveRoute,
    });
    fireEvent.click(screen.getByRole("button", { name: /configure/i }));

    const countInput = screen.getByRole("spinbutton", { name: /instances count/i });
    expect(countInput).not.toHaveAttribute("max");
    fireEvent.change(countInput, { target: { value: "1000" } });
    expect(countInput).toHaveValue(1000);
    fireEvent.click(screen.getByRole("button", { name: /apply route/i }));

    expect(handleSaveRoute).toHaveBeenCalledWith(
      "instance-suggestion",
      1,
      expect.objectContaining({
        inputs: { conditioning: { count: 1000 }, parameters: {} },
      })
    );
  });

  it("replaces an unavailable existing retrieval strategy with an active fallback", () => {
    const handleSaveRoute = vi.fn();
    const retrievalModel = {
      registry_key: "cross-image-model",
      name: "Cross Image Model",
      task: "cross-image-suggestion",
      label_ids: [],
      input_contract: {
        conditioning: {
          kind: "reference_images",
          unit: "image",
          min_units: 1,
          max_units: 1,
          user_selectable_count: false,
        },
        parameters: [],
      },
    };

    renderMatrixSideDrawer({
      target: { task: "cross-image-suggestion", labelId: 1 },
      catalog: {
        models: [retrievalModel],
        retrieval_strategies: [
          { key: "old_strategy", label: "Old strategy", available: false },
          { key: "global_scene", label: "Global Scene", available: true },
        ],
      },
      draftBindings: [
        {
          task: "cross-image-suggestion",
          label_id: 1,
          model_registry_key: "cross-image-model",
          inputs: {
            conditioning: { count: 1, strategy: "old_strategy" },
            parameters: {},
          },
        },
      ],
      onSaveRoute: handleSaveRoute,
    });
    fireEvent.click(screen.getByRole("button", { name: /configure/i }));

    const strategySelect = screen.getByRole("combobox", {
      name: /retrieval strategy/i,
    });
    expect(strategySelect).toHaveValue("global_scene");
    fireEvent.click(screen.getByRole("button", { name: /apply route/i }));

    expect(handleSaveRoute).toHaveBeenCalledWith(
      "cross-image-suggestion",
      1,
      expect.objectContaining({
        inputs: expect.objectContaining({
          conditioning: expect.objectContaining({ strategy: "global_scene" }),
        }),
      })
    );
    expect(handleSaveRoute.mock.calls[0][2].inputs.conditioning.strategy).not.toBe(
      "old_strategy"
    );
  });

  it("blocks Apply when a retrieval model has no active strategy", () => {
    const handleSaveRoute = vi.fn();
    const retrievalModel = {
      registry_key: "cross-image-model-no-strategy",
      name: "Cross Image Model",
      task: "cross-image-suggestion",
      label_ids: [],
      input_contract: {
        conditioning: {
          kind: "reference_images",
          unit: "image",
          min_units: 1,
          max_units: 1,
          user_selectable_count: false,
        },
        parameters: [],
      },
    };

    renderMatrixSideDrawer({
      target: { task: "cross-image-suggestion", labelId: 1 },
      catalog: {
        models: [retrievalModel],
        retrieval_strategies: [
          { key: "old_strategy", label: "Old strategy", available: false },
        ],
      },
      draftBindings: [
        {
          task: "cross-image-suggestion",
          label_id: 1,
          model_registry_key: "cross-image-model-no-strategy",
          inputs: {
            conditioning: { count: 1, strategy: "old_strategy" },
            parameters: {},
          },
        },
      ],
      onSaveRoute: handleSaveRoute,
    });
    fireEvent.click(screen.getByRole("button", { name: /configure/i }));

    expect(screen.getByRole("alert")).toHaveTextContent(
      /no available retrieval strategy.*before applying this route/i
    );
    const applyButton = screen.getByRole("button", { name: /apply route/i });
    expect(applyButton).toBeDisabled();
    fireEvent.click(applyButton);
    expect(handleSaveRoute).not.toHaveBeenCalled();
  });

  it("normalizes every declared parameter before saving a route", () => {
    const handleSaveRoute = vi.fn();
    const parameterModel = {
      registry_key: "parameter-normalizer",
      name: "Parameter Normalizer",
      task: "prompted-segmentation",
      label_ids: [],
      input_contract: {
        conditioning: { kind: "none" },
        parameters: [
          {
            key: "int_limit",
            label: "Integer limit",
            type: "int",
            default_value: 4,
            min_value: 2,
            max_value: 8,
          },
          {
            key: "float_limit",
            label: "Float limit",
            type: "float",
            default_value: 0.5,
            min_value: 0.1,
            max_value: 0.9,
          },
          {
            key: "empty_float",
            label: "Empty float",
            type: "float",
            default_value: 0.25,
          },
          {
            key: "bound_fallback",
            label: "Bound fallback",
            type: "float",
            default_value: "not-a-number",
            min_value: 0.3,
            max_value: 1,
          },
          {
            key: "zero_fallback",
            label: "Zero fallback",
            type: "float",
            default_value: "not-a-number",
          },
          {
            key: "enabled",
            label: "Enabled",
            type: "bool",
            default_value: true,
          },
          {
            key: "mode",
            label: "Mode",
            type: "str",
            options: ["safe", "fast"],
            default_value: "invalid-default",
          },
        ],
      },
    };

    renderMatrixSideDrawer({
      catalog: { models: [parameterModel], retrieval_strategies: [] },
      draftBindings: [
        {
          task: "prompted-segmentation",
          label_id: 1,
          model_registry_key: "parameter-normalizer",
          inputs: {
            conditioning: {},
            parameters: {
              int_limit: "999",
              float_limit: "-1",
              empty_float: "",
              bound_fallback: "also-not-a-number",
              zero_fallback: "oops",
              enabled: "false",
              mode: "unknown-mode",
              obsolete_parameter: "discard me",
            },
          },
        },
      ],
      onSaveRoute: handleSaveRoute,
    });

    fireEvent.click(screen.getByRole("button", { name: /configure/i }));
    fireEvent.click(screen.getByRole("button", { name: /apply route/i }));

    expect(handleSaveRoute).toHaveBeenCalledWith(
      "prompted-segmentation",
      1,
      expect.objectContaining({
        inputs: {
          conditioning: {},
          parameters: {
            int_limit: 8,
            float_limit: 0.1,
            empty_float: 0.25,
            bound_fallback: 0.3,
            zero_fallback: 0,
            enabled: false,
            mode: "safe",
          },
        },
      })
    );
    expect(handleSaveRoute.mock.calls[0][2].inputs.parameters).not.toHaveProperty(
      "obsolete_parameter"
    );
  });

  it("describes cross-image instance conditioning as retrieval from other dataset images", () => {
    const crossInstanceModel = {
      registry_key: "cross-instance-model",
      name: "Cross Instance Model",
      task: "cross-image-suggestion",
      label_ids: [],
      input_contract: {
        conditioning: {
          kind: "instances",
          min_units: 1,
          max_units: 5,
          user_selectable_count: true,
          unit: "instance",
        },
        parameters: [],
      },
    };

    renderMatrixSideDrawer({
      target: { task: "cross-image-suggestion", labelId: 1 },
      catalog: {
        models: [crossInstanceModel],
        retrieval_strategies: [
          { key: "global_scene", label: "Global Scene", available: true },
        ],
      },
    });

    fireEvent.click(screen.getByRole("button", { name: /configure/i }));

    expect(screen.getByText("Reference examples from other images")).toBeInTheDocument();
    expect(screen.getByText("Retrieves exemplar annotations from other dataset images.")).toBeInTheDocument();
    expect(screen.queryByText("Selected objects in this image")).not.toBeInTheDocument();
  });

  it("keeps an existing route inspectable but fully read-only when editing is disabled", () => {
    const handleSaveRoute = vi.fn();
    const handleUnbindRoute = vi.fn();
    const handleClose = vi.fn();
    const readOnlyModel = {
      registry_key: "read-only-model",
      name: "Read-only Model",
      task: "prompted-segmentation",
      label_ids: [],
      input_contract: {
        conditioning: { kind: "none" },
        parameters: [
          {
            key: "max_items",
            label: "Max items",
            type: "int",
            default_value: 3,
          },
        ],
      },
    };
    const alternativeModel = {
      ...readOnlyModel,
      registry_key: "alternative-model",
      name: "Alternative Model",
    };

    renderMatrixSideDrawer({
      onClose: handleClose,
      onSaveRoute: handleSaveRoute,
      onUnbindRoute: handleUnbindRoute,
      canEdit: false,
      catalog: {
        models: [readOnlyModel, alternativeModel],
        retrieval_strategies: [],
      },
      draftBindings: [
        {
          task: "prompted-segmentation",
          label_id: 1,
          model_registry_key: "read-only-model",
          inputs: { conditioning: {}, parameters: { max_items: 3 } },
        },
      ],
    });

    const configureButtons = screen.getAllByRole("button", { name: /configure/i });
    expect(configureButtons[0]).not.toBeDisabled();
    expect(configureButtons[1]).toBeDisabled();
    fireEvent.click(screen.getByText("Alternative Model"));
    expect(screen.getByRole("button", { name: /save route/i })).toBeDisabled();

    fireEvent.click(configureButtons[0]);

    expect(screen.getByText("Max items")).toBeInTheDocument();
    const maxItemsInput = screen.getByRole("spinbutton", { name: /max items/i });
    expect(maxItemsInput).toBeDisabled();
    fireEvent.change(maxItemsInput, { target: { value: "99" } });
    expect(maxItemsInput).toHaveValue(3);

    const applyButton = screen.getByRole("button", { name: /apply route/i });
    expect(applyButton).toBeDisabled();
    fireEvent.click(applyButton);
    expect(handleSaveRoute).not.toHaveBeenCalled();
    expect(handleClose).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /models · inputs/i }));
    const unbindButton = screen.getByRole("button", { name: /unbind route/i });
    expect(unbindButton).toBeDisabled();
    fireEvent.click(unbindButton);
    expect(handleUnbindRoute).not.toHaveBeenCalled();
    expect(handleClose).not.toHaveBeenCalled();
  });
});

import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import ModelOrchestrationPanel from "./ModelOrchestrationPanel";

describe("ModelOrchestrationPanel (Design B)", () => {
  const labelsById = {
    1: { id: 1, name: "Coral fragment", parent_id: null, color: "#2dd4bf" },
    2: { id: 2, name: "Coral polyp", parent_id: 1, color: "#c084fc" },
  };

  const models = [
    {
      registry_key: "sam21-small",
      name: "SAM 2.1 Small",
      task: "prompted-segmentation",
      label_ids: [],
      latency_badge: "balanced",
      trained_on_dataset: false,
    },
    {
      registry_key: "coral-ft-v3",
      name: "Coral-FT v3",
      task: "prompted-segmentation",
      label_ids: [1],
      latency_badge: "fast",
      trained_on_dataset: true,
      input_contract: {
        parameters: [
          { key: "threshold", label: "Mask threshold", type: "float", min_value: 0, max_value: 1, default_value: 0.62 },
        ],
      },
    },
    {
      registry_key: "mask-rcnn-coral",
      name: "Mask R-CNN coral",
      task: "instance-segmentation",
      label_ids: [1, 2],
      latency_badge: "accurate",
      trained_on_dataset: true,
    },
  ];

  const strategies = [];

  const renderPanel = ({ onSavePolicy, onDeletePolicy, ...overrides }) =>
    render(
      <ModelOrchestrationPanel
        datasetId={10}
        policy={null}
        labelsById={labelsById}
        catalog={{ models, retrieval_strategies: strategies }}
        onSavePolicy={onSavePolicy}
        onDeletePolicy={onDeletePolicy}
        {...overrides}
      />
    );

  it("renders top summary cards, matrix grid, and save bar", () => {
    renderPanel({ onSavePolicy: vi.fn(), onDeletePolicy: vi.fn() });

    // Top summary
    expect(screen.getByTestId("matrix-top-summary")).toBeInTheDocument();
    expect(screen.getAllByText("Interactive segmentation").length).toBeGreaterThan(0);

    // Matrix grid
    expect(screen.getByTestId("matrix-grid")).toBeInTheDocument();
    expect(screen.getByText("Task default")).toBeInTheDocument();
    expect(screen.getByText("Coral fragment")).toBeInTheDocument();
    expect(screen.getByText("Coral polyp")).toBeInTheDocument();

    // Save bar
    expect(screen.getByTestId("routing-save-bar")).toBeInTheDocument();
  });

  it("removes orphaned label bindings from the draft while keeping task defaults", async () => {
    const taskDefault = {
      task: "prompted-segmentation",
      label_id: null,
      model_registry_key: "sam21-small",
    };
    const orphanedBinding = {
      task: "prompted-segmentation",
      label_id: 999,
      model_registry_key: "sam21-small",
    };
    const onSavePolicy = vi.fn().mockResolvedValue({});

    renderPanel({
      onSavePolicy,
      onDeletePolicy: vi.fn(),
      policy: { bindings: [taskDefault, orphanedBinding] },
    });

    await waitFor(() => {
      expect(screen.getByText(/1 unsaved change/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/Unbound Label #999/)).toBeInTheDocument();
    expect(
      screen.getByTestId("matrix-cell-prompted-segmentation-default")
    ).toHaveTextContent("SAM 2.1 Small");

    fireEvent.click(screen.getByRole("button", { name: /reset changes/i }));
    expect(screen.getByText(/1 unsaved change/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /save routing policy/i }));

    await waitFor(() => {
      expect(onSavePolicy).toHaveBeenCalledWith([taskDefault]);
    });
  });

  it("opens slide-over drawer when a cell is clicked and binds a model", async () => {
    const handleSavePolicy = vi.fn().mockResolvedValue({});

    renderPanel({ onSavePolicy: handleSavePolicy, onDeletePolicy: vi.fn() });

    // Click cell for Coral fragment under interactive segmentation
    const cell = screen.getByTestId("matrix-cell-prompted-segmentation-1");
    fireEvent.click(cell);

    // Drawer opens
    expect(screen.getByTestId("matrix-side-drawer")).toBeInTheDocument();
    expect(screen.getByText("Bind a Model")).toBeInTheDocument();

    // Select Coral-FT v3
    const modelCard = screen.getByText("Coral-FT v3");
    fireEvent.click(modelCard);

    // Click save route
    const saveRouteBtn = screen.getByRole("button", { name: /save route/i });
    fireEvent.click(saveRouteBtn);

    // Drawer closes and cell shows Coral-FT v3
    expect(screen.queryByTestId("matrix-side-drawer")).not.toBeInTheDocument();
    expect(screen.getByText("Coral-FT v3")).toBeInTheDocument();

    // Save bar shows unsaved change
    const savePolicyBtn = screen.getByRole("button", { name: /save routing policy/i });
    expect(savePolicyBtn).not.toBeDisabled();

    fireEvent.click(savePolicyBtn);

    await waitFor(() => {
      expect(handleSavePolicy).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            task: "prompted-segmentation",
            label_id: 1,
            model_registry_key: "coral-ft-v3",
          }),
        ])
      );
    });
  });

  it("opens the existing route for read-only inspection without enabling drawer edits", async () => {
    renderPanel({
      onSavePolicy: vi.fn(),
      onDeletePolicy: vi.fn(),
      canEdit: false,
      policy: {
        bindings: [
          {
            task: "prompted-segmentation",
            label_id: 1,
            model_registry_key: "coral-ft-v3",
            inputs: { parameters: { threshold: 0.62 } },
          },
        ],
      },
    });

    const cell = screen.getByTestId("matrix-cell-prompted-segmentation-1");
    expect(cell).not.toBeDisabled();
    fireEvent.click(cell);

    const drawer = screen.getByTestId("matrix-side-drawer");
    expect(drawer).toBeInTheDocument();
    expect(within(drawer).getByText("Coral-FT v3")).toBeInTheDocument();
    expect(within(drawer).getByRole("button", { name: /save route/i })).toBeDisabled();
    await waitFor(() => {
      const configureButtons = within(drawer).getAllByRole("button", {
        name: /^configure$/i,
      });
      expect(configureButtons.some((button) => !button.disabled)).toBe(true);
    });

    const configureButtons = within(drawer).getAllByRole("button", {
      name: /^configure$/i,
    });
    const selectedConfigureButton = configureButtons.find(
      (button) => !button.disabled
    );
    expect(selectedConfigureButton).toBeDefined();
    configureButtons
      .filter((button) => button !== selectedConfigureButton)
      .forEach((button) => expect(button).toBeDisabled());
    expect(within(drawer).getByRole("button", { name: /unbind route/i })).toBeDisabled();

    // The existing route may be opened for inspection, but cannot be applied.
    fireEvent.click(selectedConfigureButton);
    expect(within(drawer).getByRole("button", { name: /apply route/i })).toBeDisabled();
  });

  it("preserves mutation feedback across same-dataset policy sync and clears it on dataset change", async () => {
    const taskDefault = {
      task: "prompted-segmentation",
      label_id: null,
      model_registry_key: "sam21-small",
    };
    const orphanedBinding = {
      task: "prompted-segmentation",
      label_id: 999,
      model_registry_key: "sam21-small",
    };
    const onSavePolicy = vi.fn().mockResolvedValue({});
    const panel = renderPanel({
      onSavePolicy,
      onDeletePolicy: vi.fn(),
      policy: { bindings: [taskDefault, orphanedBinding] },
    });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /save routing policy/i })).not.toBeDisabled();
    });
    fireEvent.click(screen.getByRole("button", { name: /save routing policy/i }));

    await waitFor(() => {
      expect(screen.getByText("Routing policy saved successfully.")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("matrix-cell-prompted-segmentation-default"));
    fireEvent.click(screen.getByRole("button", { name: /unbind route/i }));

    await waitFor(() => {
      expect(
        screen.queryByText("Routing policy saved successfully.")
      ).not.toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /save routing policy/i }));
    await waitFor(() => {
      expect(screen.getByText("Routing policy saved successfully.")).toBeInTheDocument();
    });

    panel.rerender(
      <ModelOrchestrationPanel
        datasetId={10}
        policy={{ bindings: [taskDefault] }}
        labelsById={labelsById}
        catalog={{ models, retrieval_strategies: strategies }}
        onSavePolicy={onSavePolicy}
        onDeletePolicy={vi.fn()}
      />
    );

    await waitFor(() => {
      expect(screen.getByText("Routing policy saved successfully.")).toBeInTheDocument();
    });

    panel.rerender(
      <ModelOrchestrationPanel
        datasetId={11}
        policy={null}
        labelsById={labelsById}
        catalog={{ models, retrieval_strategies: strategies }}
        onSavePolicy={onSavePolicy}
        onDeletePolicy={vi.fn()}
      />
    );

    await waitFor(() => {
      expect(
        screen.queryByText("Routing policy saved successfully.")
      ).not.toBeInTheDocument();
    });
  });

  it.each([
    ["saving", { isSaving: true }],
    ["deleting", { isDeleting: true }],
  ])("does not open a new drawer while %s", (_busyState, busyProps) => {
    renderPanel({
      onSavePolicy: vi.fn(),
      onDeletePolicy: vi.fn(),
      ...busyProps,
    });

    const cell = screen.getByTestId("matrix-cell-prompted-segmentation-1");
    expect(cell).toBeDisabled();
    fireEvent.click(cell);

    expect(screen.queryByTestId("matrix-side-drawer")).not.toBeInTheDocument();
  });
});

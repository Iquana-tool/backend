import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import MatrixGrid from "./MatrixGrid";

describe("MatrixGrid (Design B)", () => {
  const labelsById = {
    1: { id: 1, name: "Coral fragment", parent_id: null, color: "#2dd4bf", instance_count: 412 },
    2: { id: 2, name: "Coral polyp", parent_id: 1, color: "#c084fc", instance_count: 85 },
  };

  const catalog = {
    models: [
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
      },
      {
        registry_key: "mask-rcnn-coral",
        name: "Mask R-CNN coral",
        task: "instance-segmentation",
        label_ids: [1, 2],
        latency_badge: "accurate",
        trained_on_dataset: true,
      },
      {
        registry_key: "specific-default-model",
        name: "Polyp Specialist",
        task: "instance-suggestion",
        label_ids: [2],
        trained_on_dataset: false,
      },
    ],
    retrieval_strategies: [],
  };

  const draftBindings = [
    { task: "prompted-segmentation", label_id: null, model_registry_key: "sam21-small" },
    { task: "prompted-segmentation", label_id: 1, model_registry_key: "coral-ft-v3" },
    { task: "instance-suggestion", label_id: null, model_registry_key: "specific-default-model" },
    { task: "instance-segmentation", label_id: null, model_registry_key: "mask-rcnn-coral" },
  ];

  const renderMatrixGrid = (overrides = {}) =>
    render(
      <MatrixGrid
        labelsById={labelsById}
        catalog={catalog}
        draftBindings={draftBindings}
        onSelectCell={vi.fn()}
        {...overrides}
      />
    );

  it("renders task default row and hierarchical label rows for all 4 canonical tasks", () => {
    renderMatrixGrid();

    // 4 canonical task headers without extra subtitles
    expect(screen.getByText("Prompted segmentation")).toBeInTheDocument();
    expect(screen.getByText("Within-image suggestion")).toBeInTheDocument();
    expect(screen.getByText("Instance segmentation")).toBeInTheDocument();
    expect(screen.getByText("Cross-image suggestion")).toBeInTheDocument();

    // Task default row header and inherited cells
    expect(screen.getAllByText("Task default").length).toBeGreaterThan(1);
    expect(screen.getByText("Default where compatible")).toBeInTheDocument();

    // Labels
    expect(screen.getByText("Coral fragment")).toBeInTheDocument();
    expect(screen.getByText("Coral polyp")).toBeInTheDocument();

    // Model cells - explicit override
    expect(screen.getByText("Coral-FT v3")).toBeInTheDocument();

    // Model cells - inherited with Task default sublabel
    expect(screen.getByText("Inherits SAM 2.1 Small")).toBeInTheDocument();
    expect(screen.getByText("Inherits Polyp Specialist")).toBeInTheDocument();
    expect(screen.getAllByText("Inherits Mask R-CNN coral")).toHaveLength(2);
    expect(screen.getAllByText("Task default").length).toBeGreaterThan(1);

    // Model cells - incompatible task default
    expect(screen.getByText("Needs compatible model")).toBeInTheDocument();
    expect(screen.getByText("Default incompatible")).toBeInTheDocument();

    // Model cells - unbound / not configured
    expect(screen.getByText("No task default")).toBeInTheDocument();
    expect(screen.getAllByText("Not configured")).toHaveLength(2);

    // Legend
    expect(screen.getByText("configured route")).toBeInTheDocument();
    expect(screen.getByText("inherited")).toBeInTheDocument();
    expect(screen.getByText("needs attention")).toBeInTheDocument();
    expect(screen.getByText("trained on this dataset")).toBeInTheDocument();
  });

  it("renders continuous tree guides for parent and child rows", () => {
    renderMatrixGrid();

    const parentRow = screen.getByTestId("matrix-label-row-1");
    const childRow = screen.getByTestId("matrix-label-row-2");

    expect(parentRow).toHaveAttribute("data-hierarchy-depth", "0");
    expect(childRow).toHaveAttribute("data-hierarchy-depth", "1");
    expect(parentRow.querySelectorAll('[data-hierarchy-guide="vertical"]')).toHaveLength(1);
    expect(childRow.querySelectorAll('[data-hierarchy-guide="vertical"]')).toHaveLength(1);
    expect(childRow.querySelectorAll('[data-hierarchy-guide="horizontal"]')).toHaveLength(1);
  });

  it("calls onSelectCell with correct task and labelId when a cell is clicked", () => {
    const handleSelectCell = vi.fn();

    renderMatrixGrid({ onSelectCell: handleSelectCell });

    const cell = screen.getByTestId("matrix-cell-prompted-segmentation-1");
    fireEvent.click(cell);

    expect(handleSelectCell).toHaveBeenCalledWith("prompted-segmentation", 1);
  });

  it("keeps read-only cells available for route inspection", () => {
    const handleSelectCell = vi.fn();

    renderMatrixGrid({ canEdit: false, onSelectCell: handleSelectCell });

    const cell = screen.getByTestId("matrix-cell-prompted-segmentation-1");
    expect(cell).not.toBeDisabled();
    fireEvent.click(cell);

    expect(handleSelectCell).toHaveBeenCalledWith("prompted-segmentation", 1);
  });

  it("does not open a drawer when drawer selection is temporarily blocked", () => {
    const handleSelectCell = vi.fn();

    renderMatrixGrid({ canOpenDrawer: false, onSelectCell: handleSelectCell });

    const cell = screen.getByTestId("matrix-cell-prompted-segmentation-1");
    expect(cell).toBeDisabled();
    fireEvent.click(cell);

    expect(handleSelectCell).not.toHaveBeenCalled();
  });

  it("calls onSelectCell for within-image suggestion cells", () => {
    const handleSelectCell = vi.fn();

    renderMatrixGrid({ onSelectCell: handleSelectCell });

    const cell = screen.getByTestId("matrix-cell-instance-suggestion-1");
    fireEvent.click(cell);

    expect(handleSelectCell).toHaveBeenCalledWith("instance-suggestion", 1);
  });

  it("calls onSelectCell for task default cell when clicked", () => {
    const handleSelectCell = vi.fn();

    renderMatrixGrid({ onSelectCell: handleSelectCell });

    const cell = screen.getByTestId("matrix-cell-instance-segmentation-default");
    fireEvent.click(cell);

    expect(handleSelectCell).toHaveBeenCalledWith("instance-segmentation", null);
  });

  it("provides task-specific accessibility help text for interactive vs batch unconfigured cells", () => {
    renderMatrixGrid({ draftBindings: [] });

    // Interactive unconfigured cell (prompted seg)
    const interactiveCell = screen.getByTestId("matrix-cell-prompted-segmentation-1");
    expect(interactiveCell).toHaveAttribute(
      "title",
      "Not configured. Uses task default when available, or falls back to personal favorite/model fallback."
    );

    // Batch unconfigured cell (instance segmentation)
    const batchCell = screen.getByTestId("matrix-cell-instance-segmentation-1");
    expect(batchCell).toHaveAttribute(
      "title",
      "Not configured. Uses task default when available; omitted from batch operations if no default exists."
    );
  });

  it("renders canonical backend badges in configured model cell subtitle", () => {
    const badgeCatalog = {
      models: [
        {
          registry_key: "badge-model",
          name: "Badge Model",
          task: "prompted-segmentation",
          badges: ["realtime", "15 MB"],
          label_ids: [1],
        },
      ],
      retrieval_strategies: [],
    };

    renderMatrixGrid({
      catalog: badgeCatalog,
      draftBindings: [{ task: "prompted-segmentation", label_id: 1, model_registry_key: "badge-model" }],
    });

    expect(screen.getByText("Badge Model")).toBeInTheDocument();
    expect(screen.getByText(/realtime · 15 MB · 1 class/i)).toBeInTheDocument();
  });

  it("preserves explicit label colors and canonical palette mapping for label rows", () => {
    const customLabelsById = {
      1: { id: 1, name: "Teal Explicit", color: "#2dd4bf" },
      2: { id: 2, name: "Purple Implicit" },
    };

    renderMatrixGrid({ labelsById: customLabelsById, draftBindings: [] });

    const row1 = screen.getByTestId("matrix-label-row-1");
    const dot1 = row1.querySelector('span[style*="background-color"]');
    expect(dot1).toHaveStyle({ backgroundColor: "rgb(45, 212, 191)" }); // #2dd4bf in rgb

    const row2 = screen.getByTestId("matrix-label-row-2");
    const dot2 = row2.querySelector('span[style*="background-color"]');
    expect(dot2).toHaveStyle({ backgroundColor: "rgb(249, 115, 22)" }); // #f97316 (orange-500, CLASS_PALETTE[1])
  });
});

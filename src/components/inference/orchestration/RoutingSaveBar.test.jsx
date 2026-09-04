import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import RoutingSaveBar from "./RoutingSaveBar";

const renderSaveBar = (props) => render(<RoutingSaveBar {...props} />);

describe("RoutingSaveBar", () => {
  it("renders clean state when there are no unsaved changes", () => {
    renderSaveBar({ hasUnsavedChanges: false, hasSavedPolicy: true, canEdit: true });

    expect(screen.getByText("All routing changes saved")).toBeInTheDocument();
    expect(screen.getByText("Saved")).toBeInTheDocument();
  });

  it("renders empty saved policy text when hasSavedPolicy is false", () => {
    renderSaveBar({ hasUnsavedChanges: false, hasSavedPolicy: false, canEdit: true });

    expect(screen.getByText("No custom routing policy saved")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /clear policy/i })).not.toBeInTheDocument();
  });

  it("renders dirty state with count and preview summary", () => {
    const onSave = vi.fn();
    const onReset = vi.fn();

    renderSaveBar({
      hasUnsavedChanges: true,
      changeSummary: {
        totalCount: 2,
        items: ["Bound SAM 2 to cell", "Updated parameters for nucleus"],
        summaryText: "Bound SAM 2 to cell, Updated parameters for nucleus",
      },
      hasSavedPolicy: true,
      onSave,
      onReset,
      canEdit: true,
    });

    expect(screen.getByText(/2 unsaved changes/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Bound SAM 2 to cell, Updated parameters for nucleus/i)
    ).toBeInTheDocument();

    const saveBtn = screen.getByRole("button", { name: /save routing policy/i });
    expect(saveBtn).not.toBeDisabled();
    fireEvent.click(saveBtn);
    expect(onSave).toHaveBeenCalledTimes(1);

    const resetBtn = screen.getByRole("button", { name: /reset changes/i });
    fireEvent.click(resetBtn);
    expect(onReset).toHaveBeenCalledTimes(1);
  });

  it("shows confirmation and calls onClear when confirming clear policy even on empty saved policy", () => {
    const onClear = vi.fn();

    renderSaveBar({ hasUnsavedChanges: false, hasSavedPolicy: true, onClear, canEdit: true });

    const clearBtn = screen.getByRole("button", { name: /clear policy/i });
    fireEvent.click(clearBtn);

    expect(screen.getByText("Clear all custom routes?")).toBeInTheDocument();

    const confirmBtn = screen.getByRole("button", { name: /yes, clear/i });
    fireEvent.click(confirmBtn);

    expect(onClear).toHaveBeenCalledTimes(1);
  });

  it("cancels clear policy confirmation without calling onClear", () => {
    const onClear = vi.fn();

    renderSaveBar({ hasUnsavedChanges: false, hasSavedPolicy: true, onClear, canEdit: true });

    fireEvent.click(screen.getByRole("button", { name: /clear policy/i }));
    expect(screen.getByText("Clear all custom routes?")).toBeInTheDocument();

    const cancelBtn = screen.getByRole("button", { name: /cancel/i });
    fireEvent.click(cancelBtn);

    expect(onClear).not.toHaveBeenCalled();
    expect(screen.queryByText("Clear all custom routes?")).not.toBeInTheDocument();
  });

  it("renders status message when provided", () => {
    renderSaveBar({
      hasUnsavedChanges: false,
      statusMessage: { type: "success", text: "Policy successfully updated." },
      canEdit: true,
    });

    expect(screen.getByText("Policy successfully updated.")).toBeInTheDocument();
  });

  it("disables actions during saving or deleting", () => {
    renderSaveBar({
      hasUnsavedChanges: true,
      changeSummary: { totalCount: 1, items: [], summaryText: "One change" },
      hasSavedPolicy: true,
      isSaving: true,
      canEdit: true,
    });

    const saveBtn = screen.getByRole("button", { name: /saving/i });
    expect(saveBtn).toBeDisabled();

    const resetBtn = screen.getByRole("button", { name: /reset changes/i });
    expect(resetBtn).toBeDisabled();
  });

  it("does not render mutation actions in read-only mode", () => {
    renderSaveBar({ hasUnsavedChanges: false, hasSavedPolicy: true, canEdit: false });

    expect(screen.queryByRole("button", { name: /clear policy/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /save routing policy/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /saved/i })).not.toBeInTheDocument();
    expect(screen.getByText("Read-only mode")).toBeInTheDocument();
  });
});

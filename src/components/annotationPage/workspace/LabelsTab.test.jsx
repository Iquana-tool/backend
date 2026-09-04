import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import LabelsTab from "./LabelsTab";
import * as DatasetContext from "../../../contexts/DatasetContext";
import * as ToastContext from "../../../contexts/ToastContext";
import * as annotationSelectors from "../../../stores/selectors/annotationSelectors";
import * as labelsApi from "../../../api/labels";

vi.mock("../../../api/labels", () => ({
    createLabel: vi.fn(),
    fetchLabels: vi.fn(),
}));

describe("LabelsTab taxonomy rendering and interactions", () => {
    const setActiveLabelId = vi.fn();
    const toggleVisibility = vi.fn();
    const addToast = vi.fn();

    const mockLabels = [
        { id: 1, name: "Coral", color: "#FF0000", parent_id: null },
        { id: 2, name: "Bleached", color: "#00FF00", parent_id: null },
    ];

    beforeEach(() => {
        vi.clearAllMocks();

        vi.spyOn(DatasetContext, "useDataset").mockReturnValue({
            currentDataset: { id: 10, name: "Test Dataset" },
        });
        vi.spyOn(ToastContext, "useToast").mockReturnValue({ addToast });

        vi.spyOn(annotationSelectors, "useDatasetLabels").mockReturnValue(mockLabels);
        vi.spyOn(annotationSelectors, "useSetDatasetLabels").mockReturnValue(vi.fn());
        vi.spyOn(annotationSelectors, "useObjectsList").mockReturnValue([
            { id: 100, labelId: 1 },
            { id: 101, labelId: 1 },
        ]);
        vi.spyOn(annotationSelectors, "useObjectsVisibility").mockReturnValue({ labels: { 1: true, 2: false } });
        vi.spyOn(annotationSelectors, "useToggleVisibility").mockReturnValue(toggleVisibility);
        vi.spyOn(annotationSelectors, "useActiveLabelId").mockReturnValue(1);
        vi.spyOn(annotationSelectors, "useSetActiveLabelId").mockReturnValue(setActiveLabelId);
        vi.spyOn(annotationSelectors, "useLabelColorOverrides").mockReturnValue({});
        vi.spyOn(annotationSelectors, "useSetLabelColorOverride").mockReturnValue(vi.fn());
    });

    it("renders label rows with counts and active badge", () => {
        render(<LabelsTab />);

        expect(screen.getAllByText("Coral").length).toBeGreaterThanOrEqual(1);
        expect(screen.getByText("Bleached")).toBeInTheDocument();
        expect(screen.getByText("ACTIVE")).toBeInTheDocument();
        expect(screen.getByText("2")).toBeInTheDocument(); // count for Coral
    });

    it("clicking a label arms it as active label", () => {
        render(<LabelsTab />);

        fireEvent.click(screen.getByText("Bleached"));
        expect(setActiveLabelId).toHaveBeenCalledWith(2);
    });

    it("clicking visibility button toggles class visibility", () => {
        render(<LabelsTab />);

        const hideButton = screen.getByRole("button", { name: "Hide Coral" });
        fireEvent.click(hideButton);
        expect(toggleVisibility).toHaveBeenCalledWith(1);
    });

    it("creates a new label and displays success toast", async () => {
        labelsApi.createLabel.mockResolvedValue({ id: 3, name: "Algae" });
        labelsApi.fetchLabels.mockResolvedValue({
            labels: {
                id_to_label_object: {
                    1: { id: 1, name: "Coral" },
                    2: { id: 2, name: "Bleached" },
                    3: { id: 3, name: "Algae" },
                },
            },
        });

        render(<LabelsTab />);

        const newLabelBtn = screen.getByRole("button", { name: "New label" });
        fireEvent.click(newLabelBtn);

        const input = screen.getByPlaceholderText("Label name");
        expect(input).toBeInTheDocument();

        fireEvent.change(input, { target: { value: "Algae" } });
        expect(input.value).toBe("Algae");

        const addBtn = screen.getByRole("button", { name: "Add" });
        fireEvent.click(addBtn);

        await waitFor(() => {
            expect(labelsApi.createLabel).toHaveBeenCalledWith(
                { name: "Algae", parent_id: null },
                10
            );
            expect(addToast).toHaveBeenCalledWith({
                type: "success",
                message: "Label “Algae” created.",
            });
        });
    });
});

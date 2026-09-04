import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { vi, describe, it, expect } from "vitest";
import InferenceProgressPanel from "./InferenceProgressPanel";

vi.mock("../../api/inference", () => ({
    getInferenceJobItems: vi.fn().mockResolvedValue([]),
}));

describe("InferenceProgressPanel run history actions", () => {
    const baseJob = {
        id: 101,
        dataset_id: 1,
        name: "test_run",
        total_units: 10,
        done_units: 10,
        failed_units: 0,
        contours_created: 5,
        status: "succeeded",
        plan_steps: [{ label_id: 1, model_registry_key: "m2f" }],
        options: {},
    };

    it("keeps cancellation disabled while a run is cancelling", () => {
        const onCancel = vi.fn();
        const job = { ...baseJob, status: "cancelling" };
        render(
            <InferenceProgressPanel
                job={job}
                onCancel={onCancel}
                isCancelling={false}
            />
        );

        const stopButton = screen.getByRole("button", { name: /stopping after this image/i });
        expect(stopButton).toBeInTheDocument();
        expect(stopButton).toBeDisabled();
        fireEvent.click(stopButton);
        expect(onCancel).not.toHaveBeenCalled();
    });

    it("renders 'Remove from history' for non-running statuses and hides it while running", () => {
        const allowedStatuses = ["pending", "cancelling", "succeeded", "partial", "failed", "cancelled"];
        allowedStatuses.forEach((status) => {
            const onDelete = vi.fn();
            const job = { ...baseJob, status };
            const { unmount } = render(
                <InferenceProgressPanel
                    job={job}
                    onDelete={onDelete}
                />
            );

            const deleteButton = screen.getByRole("button", { name: /remove from history/i });
            expect(deleteButton).toBeInTheDocument();
            fireEvent.click(deleteButton);
            expect(onDelete).toHaveBeenCalled();
            unmount();
        });

        // Running status blocks deletion
        const { unmount } = render(
            <InferenceProgressPanel
                job={{ ...baseJob, status: "running" }}
                onDelete={vi.fn()}
            />
        );
        expect(screen.queryByRole("button", { name: /remove from history/i })).not.toBeInTheDocument();
        unmount();
    });
});

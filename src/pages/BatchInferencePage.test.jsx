import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach } from "vitest";
import BatchInferencePage from "./BatchInferencePage";
import { fetchLabels } from "../api/labels";
import {
    getInferenceModelCatalog,
    getInferenceScopeCounts,
    getInferenceJobs,
    startInferenceJob,
    cancelInferenceJob,
    streamInferenceJob,
    previewInferenceReplace,
    getInferenceRoutingPolicy,
    deleteInferenceJob,
} from "../api/inference";

const { mockRoute } = vi.hoisted(() => ({
    mockRoute: { datasetId: "101" },
}));

vi.mock("react-router-dom", () => ({
    useParams: () => ({ datasetId: mockRoute.datasetId }),
    useNavigate: () => vi.fn(),
    Link: ({ to, children, ...props }) => (
        <a href={to} {...props}>
            {children}
        </a>
    ),
}));

vi.mock("../contexts/DatasetContext", () => ({
    useDataset: () => ({ currentDataset: { id: mockRoute.datasetId, name: "Test Dataset" } }),
}));

vi.mock("../hooks/usePermissions", () => ({
    usePermissions: () => ({ can: () => true }),
}));

vi.mock("../components/datasets/gallery/DatasetManagementLayout", () => ({
    default: ({ children }) => <div data-testid="layout">{children}</div>,
}));

vi.mock("../components/inference/LabelModelPlanner", () => ({
    default: ({ labelsById, models, steps, onChange }) => (
        <div data-testid="planner">
            <span>Planner Component ({steps?.length || 0} steps)</span>
            <span data-testid="planner-label-count">{Object.keys(labelsById || {}).length}</span>
            <span data-testid="planner-model-count">{models?.length || 0}</span>
            <button
                onClick={() =>
                    onChange?.(1, {
                        label_id: 1,
                        model_registry_key: "m2f",
                        task: "instance-segmentation",
                        inputs: null,
                    })
                }
                data-testid="add-step-btn"
            >
                Add Step
            </button>
        </div>
    ),
}));

vi.mock("../components/inference/InferenceProgressPanel", () => ({
    STATUS_STYLE: {
        pending: "bg-well text-t2",
        running: "bg-acS text-ac",
        cancelling: "bg-warnBg text-warn",
        succeeded: "bg-okBg text-ok",
        partial: "bg-warnBg text-warn",
        failed: "bg-errBg text-err",
        cancelled: "bg-well text-t3",
    },
    TERMINAL_JOB_STATUSES: new Set(["succeeded", "partial", "failed", "cancelled"]),
    default: ({ job, onCancel, isCancelling, onDelete }) => (
        <div data-testid="progress-panel">
            <span data-testid="progress-job-id">{job?.id}</span>
            <span data-testid="progress-is-cancelling">{isCancelling ? "Cancelling" : "Not Cancelling"}</span>
            <button onClick={onCancel} disabled={isCancelling}>Cancel Run</button>
            <button onClick={onDelete} data-testid="delete-job-btn">Delete Run</button>
        </div>
    ),
}));

vi.mock("../api/labels", () => ({
    fetchLabels: vi.fn(),
}));

vi.mock("../api/inference", () => ({
    getInferenceModelCatalog: vi.fn(),
    getInferenceScopeCounts: vi.fn(),
    getInferenceJobs: vi.fn(),
    startInferenceJob: vi.fn(),
    cancelInferenceJob: vi.fn(),
    deleteInferenceJob: vi.fn(),
    streamInferenceJob: vi.fn(),
    previewInferenceReplace: vi.fn(),
    getInferenceRoutingPolicy: vi.fn(),
}));

describe("BatchInferencePage execution workflow and policy integration", () => {
    const mockAbort = vi.fn();

    beforeEach(() => {
        vi.clearAllMocks();
        mockRoute.datasetId = "101";

        fetchLabels.mockResolvedValue({
            labels: {
                id_to_label_object: {
                    1: { id: 1, name: "Cell", color: "#ff0000" },
                    2: { id: 2, name: "Nucleus", color: "#00ff00" },
                },
            },
        });
        getInferenceModelCatalog.mockResolvedValue({
            models: [
                { registry_key: "m2f", name: "Mask2Former", task: "instance-segmentation", label_ids: [] },
                { registry_key: "sam3-cross", name: "SAM 3 Cross", task: "cross-image-suggestion", label_ids: [] },
            ],
            retrieval_strategies: [],
        });
        getInferenceScopeCounts.mockResolvedValue({
            total: 10,
            not_started: 5,
            unreviewed: 3,
        });
        getInferenceRoutingPolicy.mockResolvedValue(null);
        getInferenceJobs.mockResolvedValue([]);
        previewInferenceReplace.mockResolvedValue({
            contours: 0,
            images: 0,
            reviewed_contours: 0,
            root_contours: 0,
            protected_contours: 0,
        });
        streamInferenceJob.mockReturnValue({ abort: mockAbort });
    });

    it("resets active run view, stream, and jobs when switching datasets", async () => {
        const activeJobA = {
            id: 999,
            dataset_id: 101,
            status: "running",
            created_at: new Date().toISOString(),
            steps: [{ label_id: 1, model_registry_key: "m2f", task: "instance-segmentation" }],
            image_count: 10,
            completed_images: 2,
        };
        getInferenceJobs.mockResolvedValueOnce([activeJobA]);

        const { rerender } = render(<BatchInferencePage />);

        await waitFor(() => {
            expect(screen.getByTestId("progress-panel")).toBeInTheDocument();
            expect(screen.getByTestId("progress-job-id")).toHaveTextContent("999");
            expect(streamInferenceJob).toHaveBeenCalledWith(999, expect.any(Function), expect.any(Function));
        });

        // Switch to Dataset 202 which has no jobs and no active run
        mockRoute.datasetId = "202";
        getInferenceJobs.mockResolvedValueOnce([]);
        getInferenceRoutingPolicy.mockResolvedValueOnce(null);

        rerender(<BatchInferencePage />);

        expect(mockAbort).toHaveBeenCalled();

        await waitFor(() => {
            expect(screen.queryByTestId("progress-panel")).not.toBeInTheDocument();
            expect(screen.queryByTestId("orchestration-panel")).not.toBeInTheDocument();
        });
    });

    it("does not render policy editor controls and provides link to Model Orchestration page", async () => {
        render(<BatchInferencePage />);

        await waitFor(() => {
            expect(screen.getByTestId("planner")).toBeInTheDocument();
        });

        expect(screen.queryByTestId("orchestration-panel")).not.toBeInTheDocument();
        expect(screen.queryByText(/Dataset Model Routing/i)).not.toBeInTheDocument();

        const editLink = screen.getByRole("link", { name: /edit model routing/i });
        expect(editLink).toBeInTheDocument();
        expect(editLink).toHaveAttribute("href", "/dataset/101/model-orchestration");
    });

    it("loads saved routing policy and automatically pre-fills batch plan when single batch task exists", async () => {
        const singleTaskPolicy = {
            dataset_id: 101,
            bindings: [
                {
                    task: "instance-segmentation",
                    label_id: 1,
                    model_registry_key: "m2f",
                    inputs: null,
                },
            ],
            updated_by: "tester",
            created_at: "2026-08-24T10:00:00Z",
            updated_at: "2026-08-24T12:00:00Z",
        };
        getInferenceRoutingPolicy.mockResolvedValueOnce(singleTaskPolicy);

        render(<BatchInferencePage />);

        await waitFor(() => {
            expect(screen.getByTestId("planner")).toHaveTextContent("Planner Component (1 steps)");
        });
    });

    it("keeps required setup and manual planning available when routing policy loading fails", async () => {
        getInferenceRoutingPolicy.mockRejectedValueOnce(new Error("Policy service unavailable"));

        render(<BatchInferencePage />);

        await waitFor(() => {
            expect(screen.getByTestId("planner")).toBeInTheDocument();
            expect(screen.getByTestId("planner-label-count")).toHaveTextContent("2");
            expect(screen.getByTestId("planner-model-count")).toHaveTextContent("2");
            expect(screen.getByText(/Every image/i)).toHaveTextContent("10");
            expect(screen.getByText(/Not annotated yet/i)).toHaveTextContent("5");
        });

        expect(screen.queryByText("Policy service unavailable")).not.toBeInTheDocument();

        fireEvent.click(screen.getByTestId("add-step-btn"));

        expect(screen.getByTestId("planner")).toHaveTextContent("Planner Component (1 steps)");
        expect(screen.getByRole("button", { name: /start inference/i })).not.toBeDisabled();
    });

    it("renders compact chooser buttons when dual batch tasks exist and applies selected routes", async () => {
        const dualTaskPolicy = {
            dataset_id: 101,
            bindings: [
                {
                    task: "instance-segmentation",
                    label_id: 1,
                    model_registry_key: "m2f",
                    inputs: null,
                },
                {
                    task: "cross-image-suggestion",
                    label_id: 2,
                    model_registry_key: "sam3-cross",
                    inputs: null,
                },
            ],
            updated_by: "tester",
        };
        getInferenceRoutingPolicy.mockResolvedValueOnce(dualTaskPolicy);

        render(<BatchInferencePage />);

        await waitFor(() => {
            expect(screen.getByRole("button", { name: /apply all routes/i })).toBeInTheDocument();
            expect(screen.getByRole("button", { name: /instance seg only/i })).toBeInTheDocument();
            expect(screen.getByRole("button", { name: /cross-image only/i })).toBeInTheDocument();
        });

        // Click Apply All Routes
        const applyAllBtn = screen.getByRole("button", { name: /apply all routes/i });
        fireEvent.click(applyAllBtn);
        expect(screen.getByTestId("planner")).toHaveTextContent("Planner Component (2 steps)");
        expect(applyAllBtn).toHaveAttribute("aria-pressed", "true");

        // Click Instance Seg Only
        const instanceOnlyBtn = screen.getByRole("button", { name: /instance seg only/i });
        fireEvent.click(instanceOnlyBtn);
        expect(screen.getByTestId("planner")).toHaveTextContent("Planner Component (1 steps)");
        expect(instanceOnlyBtn).toHaveAttribute("aria-pressed", "true");
        expect(applyAllBtn).toHaveAttribute("aria-pressed", "false");

        // Click Cross-Image Only
        const crossOnlyBtn = screen.getByRole("button", { name: /cross-image only/i });
        fireEvent.click(crossOnlyBtn);
        expect(screen.getByTestId("planner")).toHaveTextContent("Planner Component (1 steps)");
        expect(crossOnlyBtn).toHaveAttribute("aria-pressed", "true");
        expect(instanceOnlyBtn).toHaveAttribute("aria-pressed", "false");
    });

    it("starts a batch inference run and transitions to run mode", async () => {
        const createdJob = {
            id: 888,
            dataset_id: 101,
            status: "running",
            created_at: new Date().toISOString(),
            steps: [{ label_id: 1, model_registry_key: "m2f", task: "instance-segmentation" }],
            image_count: 10,
            completed_images: 0,
        };
        startInferenceJob.mockResolvedValueOnce(createdJob);

        render(<BatchInferencePage />);

        await waitFor(() => {
            expect(screen.getByTestId("planner")).toBeInTheDocument();
        });

        // Add step to planner
        fireEvent.click(screen.getByTestId("add-step-btn"));

        const startBtn = screen.getByRole("button", { name: /start inference/i });
        expect(startBtn).not.toBeDisabled();

        fireEvent.click(startBtn);

        await waitFor(() => {
            expect(startInferenceJob).toHaveBeenCalledWith(
                expect.objectContaining({
                    dataset_id: 101,
                    steps: [
                        {
                            label_id: 1,
                            model_registry_key: "m2f",
                            task: "instance-segmentation",
                            inputs: null,
                        },
                    ],
                })
            );
            expect(screen.getByTestId("progress-panel")).toBeInTheDocument();
            expect(screen.getByTestId("progress-job-id")).toHaveTextContent("888");
        });
    });
});

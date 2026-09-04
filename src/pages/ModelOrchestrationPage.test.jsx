import React from "react";
import { render, screen, waitFor, fireEvent, act } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach } from "vitest";
import ModelOrchestrationPage from "./ModelOrchestrationPage";
import { fetchLabels } from "../api/labels";
import {
    getInferenceModelCatalog,
    getInferenceRoutingPolicy,
    updateInferenceRoutingPolicy,
    deleteInferenceRoutingPolicy,
} from "../api/inference";

const { mockRoute, mockPermissions, mockDatasetContext } = vi.hoisted(() => ({
    mockRoute: { datasetId: "101" },
    mockPermissions: { can: vi.fn(() => true) },
    mockDatasetContext: { loading: false },
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
    useDataset: () => ({
        currentDataset: { id: mockRoute.datasetId, name: "Test Dataset" },
        loading: mockDatasetContext.loading,
    }),
}));

vi.mock("../hooks/usePermissions", () => ({
    usePermissions: () => ({ can: mockPermissions.can }),
}));

vi.mock("../components/datasets/gallery/DatasetManagementLayout", () => ({
    default: ({ children }) => <div data-testid="layout">{children}</div>,
}));

vi.mock("../components/inference/ModelOrchestrationPanel", () => ({
    default: ({ datasetId, policy, labelsById, catalog, onSavePolicy, onDeletePolicy, isSaving, isDeleting, canEdit }) => (
        <div data-testid="orchestration-panel">
            <span data-testid="panel-dataset-id">{datasetId}</span>
            <span data-testid="panel-can-edit">{canEdit ? "editable" : "read-only"}</span>
            <span data-testid="panel-policy-state">{policy ? `Saved: ${policy.bindings?.length || 0} bindings` : "No Policy"}</span>
            <button
                onClick={() =>
                    onSavePolicy?.([
                        {
                            task: "instance-segmentation",
                            label_id: 1,
                            model_registry_key: "m2f",
                            inputs: null,
                        },
                    ])
                }
                data-testid="save-btn"
                disabled={!canEdit || isSaving}
            >
                Save
            </button>
            <button
                onClick={() => onDeletePolicy?.()}
                data-testid="delete-btn"
                disabled={!canEdit || isDeleting}
            >
                Delete
            </button>
        </div>
    ),
}));

vi.mock("../api/labels", () => ({
    fetchLabels: vi.fn(),
}));

vi.mock("../api/inference", () => ({
    getInferenceModelCatalog: vi.fn(),
    getInferenceRoutingPolicy: vi.fn(),
    updateInferenceRoutingPolicy: vi.fn(),
    deleteInferenceRoutingPolicy: vi.fn(),
}));

describe("ModelOrchestrationPage", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockRoute.datasetId = "101";
        mockPermissions.can.mockImplementation(() => true);
        mockDatasetContext.loading = false;

        fetchLabels.mockResolvedValue({
            success: true,
            labels: {
                id_to_label_object: {
                    1: { id: 1, name: "cell", parent_id: null },
                },
            },
        });

        getInferenceModelCatalog.mockResolvedValue({
            models: [
                { registry_key: "m2f", name: "Mask2Former", task: "instance-segmentation" },
            ],
            retrieval_strategies: [],
        });

        getInferenceRoutingPolicy.mockResolvedValue({
            dataset_id: 101,
            bindings: [
                { task: "instance-segmentation", label_id: 1, model_registry_key: "m2f", inputs: null },
            ],
            updated_by: "test-user",
            created_at: "2026-08-24T12:00:00Z",
            updated_at: "2026-08-24T12:00:00Z",
        });
    });

    it("loads labels, catalog, and policy and renders ModelOrchestrationPanel", async () => {
        render(<ModelOrchestrationPage />);

        expect(screen.getByText(/Loading dataset models, routing policy, and labels…/i)).toBeInTheDocument();

        await waitFor(() => {
            expect(screen.getByTestId("orchestration-panel")).toBeInTheDocument();
        });

        expect(screen.getByText(/Model Orchestration/i)).toBeInTheDocument();
        expect(screen.getByTestId("panel-dataset-id")).toHaveTextContent("101");
        expect(screen.getByTestId("panel-can-edit")).toHaveTextContent("editable");
        expect(screen.getByTestId("panel-policy-state")).toHaveTextContent("Saved: 1 bindings");
    });

    it("reports active route coverage without counting bindings for deleted labels", async () => {
        getInferenceRoutingPolicy.mockResolvedValueOnce({
            dataset_id: 101,
            bindings: [
                { task: "instance-segmentation", label_id: 1, model_registry_key: "m2f", inputs: null },
                { task: "instance-segmentation", label_id: 999, model_registry_key: "m2f", inputs: null },
            ],
        });

        render(<ModelOrchestrationPage />);

        await waitFor(() => {
            expect(screen.getByTestId("orchestration-panel")).toBeInTheDocument();
        });

        expect(screen.getByText("1 / 8")).toBeInTheDocument();
        expect(screen.queryByText("2 / 8")).not.toBeInTheDocument();
    });

    it("handles saving policy and updates state with canonical response", async () => {
        const canonicalSavedPolicy = {
            dataset_id: 101,
            bindings: [
                { task: "instance-segmentation", label_id: 1, model_registry_key: "m2f", inputs: null },
            ],
            updated_by: "current-user",
            updated_at: "2026-08-24T14:00:00Z",
        };
        updateInferenceRoutingPolicy.mockResolvedValueOnce(canonicalSavedPolicy);

        render(<ModelOrchestrationPage />);

        await waitFor(() => {
            expect(screen.getByTestId("save-btn")).toBeInTheDocument();
        });

        await act(async () => {
            fireEvent.click(screen.getByTestId("save-btn"));
        });

        expect(updateInferenceRoutingPolicy).toHaveBeenCalledWith({
            dataset_id: 101,
            bindings: [
                {
                    task: "instance-segmentation",
                    label_id: 1,
                    model_registry_key: "m2f",
                    inputs: null,
                },
            ],
        });

        expect(screen.getByTestId("panel-policy-state")).toHaveTextContent("Saved: 1 bindings");
    });

    it("handles clearing policy and calls deleteInferenceRoutingPolicy", async () => {
        deleteInferenceRoutingPolicy.mockResolvedValueOnce({ success: true });

        render(<ModelOrchestrationPage />);

        await waitFor(() => {
            expect(screen.getByTestId("delete-btn")).toBeInTheDocument();
        });

        await act(async () => {
            fireEvent.click(screen.getByTestId("delete-btn"));
        });

        expect(deleteInferenceRoutingPolicy).toHaveBeenCalledWith("101");
        expect(screen.getByTestId("panel-policy-state")).toHaveTextContent("No Policy");
    });

    it("passes read-only mode when user has AI_INTERACTIVE but not AI_BATCH_INFER", async () => {
        mockPermissions.can.mockImplementation((perm) => perm === "ai.interactive");

        render(<ModelOrchestrationPage />);

        await waitFor(() => {
            expect(screen.getByTestId("orchestration-panel")).toBeInTheDocument();
        });

        expect(screen.getByTestId("panel-can-edit")).toHaveTextContent("read-only");
    });

    it("reactively loads data when permissions resolve from unavailable to available without dataset ID change", async () => {
        mockPermissions.can.mockReturnValue(false);

        const { rerender } = render(<ModelOrchestrationPage />);

        expect(screen.getByText(/Access Restricted/i)).toBeInTheDocument();
        expect(screen.queryByTestId("orchestration-panel")).not.toBeInTheDocument();

        // Permissions resolve as available for current dataset
        mockPermissions.can.mockImplementation((perm) => perm === "ai.interactive");

        rerender(<ModelOrchestrationPage />);

        await waitFor(() => {
            expect(screen.getByTestId("orchestration-panel")).toBeInTheDocument();
        });

        expect(screen.getByTestId("panel-can-edit")).toHaveTextContent("read-only");
        expect(screen.getByTestId("panel-policy-state")).toHaveTextContent("Saved: 1 bindings");
    });

    it("displays access restricted banner when user has no AI permissions", async () => {
        mockPermissions.can.mockReturnValue(false);

        render(<ModelOrchestrationPage />);

        expect(screen.getByText(/Access Restricted/i)).toBeInTheDocument();
        expect(screen.queryByTestId("orchestration-panel")).not.toBeInTheDocument();
    });

    it("displays error banner when loading policy fails", async () => {
        getInferenceRoutingPolicy.mockRejectedValueOnce(new Error("Network connection failed"));

        render(<ModelOrchestrationPage />);

        await waitFor(() => {
            expect(screen.getByText(/Network connection failed/i)).toBeInTheDocument();
        });
    });

    it("discards stale responses when datasetId changes", async () => {
        let resolveFirstPolicy;
        getInferenceRoutingPolicy.mockImplementationOnce(() => new Promise((resolve) => {
            resolveFirstPolicy = resolve;
        }));

        const { rerender } = render(<ModelOrchestrationPage />);

        // Switch route to dataset 102
        mockRoute.datasetId = "102";
        getInferenceRoutingPolicy.mockResolvedValueOnce({
            dataset_id: 102,
            bindings: [],
        });

        rerender(<ModelOrchestrationPage />);

        await waitFor(() => {
            expect(screen.getByTestId("panel-dataset-id")).toHaveTextContent("102");
        });

        // Resolve first promise after switch
        if (resolveFirstPolicy) {
            resolveFirstPolicy({
                dataset_id: 101,
                bindings: [{ task: "instance-segmentation", label_id: 1, model_registry_key: "m2f" }],
            });
        }

        // Must still show dataset 102 state
        expect(screen.getByTestId("panel-dataset-id")).toHaveTextContent("102");
        expect(screen.getByTestId("panel-policy-state")).toHaveTextContent("Saved: 0 bindings");
    });
});

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
    getInferenceRoutingPolicy,
    streamInferenceJob,
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

describe("BatchInferencePage real integration with policy and planner", () => {
    const catalogModels = [
        {
            registry_key: "m2f-generic",
            name: "Mask2Former Generic",
            task: "instance-segmentation",
            label_ids: [],
            provenance: "legacy_default",
        },
        {
            registry_key: "m2f-cell-specialist",
            name: "Mask2Former Cell Only",
            task: "instance-segmentation",
            label_ids: [1],
            provenance: "declared",
        },
        {
            registry_key: "sam3-cross",
            name: "SAM 3 Cross Exemplar",
            task: "cross-image-suggestion",
            label_ids: [],
            provenance: "declared",
        },
        {
            registry_key: "sam3-cross-nucleus",
            name: "SAM 3 Cross Nucleus Only",
            task: "cross-image-suggestion",
            label_ids: [2],
            provenance: "declared",
        },
        {
            registry_key: "sam2-prompted",
            name: "SAM 2 Interactive Point",
            task: "prompted-segmentation",
            label_ids: [],
            provenance: "declared",
        },
    ];

    const labelsById = {
        1: { id: 1, name: "cell", parent_id: null },
        2: { id: 2, name: "nucleus", parent_id: 1 },
    };

    beforeEach(() => {
        vi.clearAllMocks();
        mockRoute.datasetId = "101";

        fetchLabels.mockResolvedValue({
            labels: { id_to_label_object: labelsById },
        });
        getInferenceModelCatalog.mockResolvedValue({
            models: catalogModels,
            retrieval_strategies: [{ key: "global_scene", label: "Global Scene", available: true }],
        });
        getInferenceScopeCounts.mockResolvedValue({
            total: 20,
            not_started: 15,
            unreviewed: 5,
        });
        getInferenceJobs.mockResolvedValue([]);
        streamInferenceJob.mockReturnValue({ abort: vi.fn() });
    });

    it("pre-fills batch plan from policy with class-aware task defaults when applying routes", async () => {
        // Policy has a class-aware task default (sam3-cross-nucleus which only predicts label 2)
        const policy = {
            dataset_id: 101,
            bindings: [
                {
                    task: "cross-image-suggestion",
                    label_id: null,
                    model_registry_key: "sam3-cross-nucleus", // predicts label 2 only
                },
                {
                    task: "instance-segmentation",
                    label_id: 1,
                    model_registry_key: "m2f-cell-specialist", // explicit override for label 1
                },
            ],
            updated_by: "tester",
        };
        getInferenceRoutingPolicy.mockResolvedValueOnce(policy);

        render(<BatchInferencePage />);

        await waitFor(() => {
            expect(screen.getByRole("button", { name: /apply all routes/i })).toBeInTheDocument();
        });

        // Click "Apply All Routes"
        const applyAllBtn = screen.getByRole("button", { name: /apply all routes/i });
        fireEvent.click(applyAllBtn);

        // Label 1 should have m2f-cell-specialist bound
        const cellSelect = screen.getByLabelText("Model for cell");
        expect(cellSelect.value).toBe("instance-segmentation::m2f-cell-specialist");

        // Label 2 should inherit the compatible default sam3-cross-nucleus
        const nucleusSelect = screen.getByLabelText("Model for nucleus");
        expect(nucleusSelect.value).toBe("cross-image-suggestion::sam3-cross-nucleus");
    });

    it("automatically pre-fills batch plan on load when batch task is unambiguous", async () => {
        const policy = {
            dataset_id: 101,
            bindings: [
                {
                    task: "instance-segmentation",
                    label_id: 1,
                    model_registry_key: "m2f-cell-specialist",
                },
            ],
            updated_by: "tester",
        };
        getInferenceRoutingPolicy.mockResolvedValueOnce(policy);

        render(<BatchInferencePage />);

        await waitFor(() => {
            expect(
                screen.getByRole("heading", {
                    name: /which model annotates which label/i,
                })
            ).toBeInTheDocument();
        });

        // Automatically pre-filled on load without clicking Apply
        const cellSelect = screen.getByLabelText("Model for cell");
        expect(cellSelect.value).toBe("instance-segmentation::m2f-cell-specialist");
    });

    it("allows applying task-specific routes when both batch tasks are configured for one label", async () => {
        // Label 1 has bindings for both instance-segmentation and cross-image-suggestion
        const policy = {
            dataset_id: 101,
            bindings: [
                {
                    task: "instance-segmentation",
                    label_id: 1,
                    model_registry_key: "m2f-cell-specialist",
                },
                {
                    task: "cross-image-suggestion",
                    label_id: 1,
                    model_registry_key: "sam3-cross",
                },
            ],
            updated_by: "tester",
        };
        getInferenceRoutingPolicy.mockResolvedValueOnce(policy);

        render(<BatchInferencePage />);

        await waitFor(() => {
            expect(
                screen.getByRole("heading", {
                    name: /which model annotates which label/i,
                })
            ).toBeInTheDocument();
        });

        // Click "Cross-Image Only" apply button
        const crossOnlyBtn = screen.getByRole("button", { name: /cross-image only/i });
        fireEvent.click(crossOnlyBtn);

        const cellSelect = screen.getByLabelText("Model for cell");
        expect(cellSelect.value).toBe("cross-image-suggestion::sam3-cross");

        // Click "Instance Seg Only" apply button
        const instOnlyBtn = screen.getByRole("button", { name: /instance seg only/i });
        fireEvent.click(instOnlyBtn);

        expect(cellSelect.value).toBe("instance-segmentation::m2f-cell-specialist");
    });

    it("safely excludes stale models from prefilling the batch plan", async () => {
        const policyWithStale = {
            dataset_id: 101,
            bindings: [
                {
                    task: "instance-segmentation",
                    label_id: 1,
                    model_registry_key: "non-existent-retired-model",
                },
            ],
            updated_by: "tester",
        };
        getInferenceRoutingPolicy.mockResolvedValueOnce(policyWithStale);

        render(<BatchInferencePage />);

        await waitFor(() => {
            expect(
                screen.getByRole("heading", {
                    name: /which model annotates which label/i,
                })
            ).toBeInTheDocument();
        });

        // Stale binding is not applied as a valid executable step
        const cellSelect = screen.getByLabelText("Model for cell");
        expect(cellSelect.value).toBe(""); // Skipped / unconfigured in planner
    });
});

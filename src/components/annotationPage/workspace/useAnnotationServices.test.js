import { renderHook, act, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import useAnnotationServices from "./useAnnotationServices";
import { getInferenceRoutingPolicy } from "../../../api/inference";

const { mockState, mockRunInstance } = vi.hoisted(() => ({
    mockRunInstance: vi.fn(),
    mockState: {
        currentDataset: { id: "101" },
        activeLabelId: 1,
        favorites: {},
        // Production slice models: only id, name, description, tags, etc.
        availablePromptedModels: [
            { id: "sam2-generic", name: "SAM 2 Generic", description: "Base", label_ids: [], tags: {} },
            { id: "sam2-cell", name: "SAM 2 Cell Specialist", description: "Cell", label_ids: [1], tags: {} },
        ],
        availableSuggestionModels: [
            { id: "sam3-suggestion-generic", name: "SAM 3 Intra", description: "Intra", label_ids: [], tags: {} },
        ],
        availableInstanceModels: [
            { id: "m2f-generic", name: "Mask2Former Generic", description: "M2F Base", label_ids: [], tags: {} },
            { id: "m2f-routed-default", name: "M2F Routed Default", description: "M2F Routed", label_ids: [], tags: {} },
        ],
        promptedModel: null,
        suggestionModel: null,
        instanceModel: null,
    },
}));

vi.mock("../../../contexts/DatasetContext", () => ({
    useDataset: () => ({ currentDataset: mockState.currentDataset }),
}));

vi.mock("../../../api/inference", () => ({
    getInferenceRoutingPolicy: vi.fn(),
}));

const mockSetPromptedModel = vi.fn((m) => { mockState.promptedModel = m; });
const mockSetSuggestionModel = vi.fn((m) => { mockState.suggestionModel = m; });
const mockSetInstanceModel = vi.fn((m) => { mockState.instanceModel = m; });

vi.mock("../../../stores/selectors/annotationSelectors", () => ({
    useAvailablePromptedModels: () => mockState.availablePromptedModels,
    useAvailableSuggestionModels: () => mockState.availableSuggestionModels,
    useAvailableInstanceModels: () => mockState.availableInstanceModels,
    useFetchAvailablePromptedModels: () => vi.fn(),
    useFetchAvailableSuggestionModels: () => vi.fn(),
    useFetchAvailableInstanceModels: () => vi.fn(),
    useIsLoadingPromptedModels: () => false,
    useIsLoadingSuggestionModels: () => false,
    useIsLoadingInstanceModels: () => false,
    useIsRunningSuggestion: () => false,
    useIsRunningInstance: () => false,
    usePromptedModel: () => mockState.promptedModel,
    useSuggestionModel: () => mockState.suggestionModel,
    useInstanceModel: () => mockState.instanceModel,
    useSetPromptedModel: () => mockSetPromptedModel,
    useSetSuggestionModel: () => mockSetSuggestionModel,
    useSetInstanceModel: () => mockSetInstanceModel,
    useInstanceRunRequested: () => false,
    useSetInstanceRunRequested: () => vi.fn(),
    useSetInstanceWarningModalOpen: () => vi.fn(),
    useActiveLabelId: () => mockState.activeLabelId,
    useModelFavorites: () => mockState.favorites,
}));

vi.mock("../../../services/annotationSession", () => ({
    default: {
        selectPromptedModel: vi.fn(),
        selectSuggestionModel: vi.fn(),
        selectInstanceModel: vi.fn(),
    },
}));

vi.mock("../../../hooks/useModelSwitchPreloader", () => ({
    default: vi.fn(),
}));

vi.mock("../../../hooks/useInstanceSegmentation", () => ({
    useInstanceSegmentation: () => ({ runInstance: mockRunInstance }),
}));

describe("useAnnotationServices with dataset routing policy", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockState.currentDataset = { id: "101" };
        mockState.promptedModel = null;
        mockState.suggestionModel = null;
        mockState.instanceModel = null;
        mockState.activeLabelId = 1;
        mockState.favorites = {};
    });

    it("applies label override for prompted segmentation and task default for instance segmentation on production model shapes", async () => {
        getInferenceRoutingPolicy.mockResolvedValueOnce({
            dataset_id: 101,
            bindings: [
                {
                    task: "prompted-segmentation",
                    label_id: 1,
                    model_registry_key: "sam2-cell",
                },
                {
                    task: "instance-segmentation",
                    label_id: null,
                    model_registry_key: "m2f-routed-default",
                },
            ],
        });

        renderHook(() => useAnnotationServices());

        await waitFor(() => {
            expect(mockSetPromptedModel).toHaveBeenCalledWith("sam2-cell");
            expect(mockSetInstanceModel).toHaveBeenCalledWith("m2f-routed-default");
        });
    });

    it("falls back to first available model when no dataset route is configured", async () => {
        getInferenceRoutingPolicy.mockResolvedValueOnce(null);

        renderHook(() => useAnnotationServices());

        await waitFor(() => {
            expect(mockSetPromptedModel).toHaveBeenCalledWith("sam2-generic");
            expect(mockSetInstanceModel).toHaveBeenCalledWith("m2f-generic");
        });
    });

    it("does not apply class-aware task default to incompatible active labels", async () => {
        // activeLabelId is 2 (unsupported by sam2-cell which only predicts 1)
        mockState.activeLabelId = 2;
        mockState.availablePromptedModels = [
            { id: "sam2-generic", name: "SAM 2 Generic", label_ids: [] },
            { id: "sam2-cell", name: "SAM 2 Cell Specialist", label_ids: [1] },
        ];

        getInferenceRoutingPolicy.mockResolvedValueOnce({
            dataset_id: 101,
            bindings: [
                {
                    task: "prompted-segmentation",
                    label_id: null,
                    model_registry_key: "sam2-cell", // class-aware default for label 1 only
                },
            ],
        });

        renderHook(() => useAnnotationServices());

        await waitFor(() => {
            // A configured route is authoritative. An invalid class-aware
            // default is disabled instead of silently choosing another model.
            expect(mockSetPromptedModel).toHaveBeenCalledWith(null);
        });
    });

    it("restores personal favorite or first compatible model when switching to an unrouted label without leaking previous specialist", async () => {
        mockState.activeLabelId = 1;
        mockState.promptedModel = "sam2-cell"; // previously armed label 1 had specialist
        mockState.favorites = { "prompted-segmentation": "sam2-generic" };

        getInferenceRoutingPolicy.mockResolvedValueOnce({
            dataset_id: 101,
            bindings: [
                {
                    task: "prompted-segmentation",
                    label_id: 1,
                    model_registry_key: "sam2-cell",
                },
            ],
        });

        const { rerender } = renderHook(() => useAnnotationServices());

        await waitFor(() => {
            expect(mockSetPromptedModel).toHaveBeenCalledWith("sam2-cell");
        });

        // Switch to label 2 (has no override)
        mockState.activeLabelId = 2;
        rerender();

        await waitFor(() => {
            // Restores favorite (sam2-generic) instead of leaking sam2-cell
            expect(mockSetPromptedModel).toHaveBeenCalledWith("sam2-generic");
        });
    });

    it("tracks policyError and restores favorite/first-model fallbacks", async () => {
        const error = new Error("500 Internal Server Error");
        getInferenceRoutingPolicy.mockRejectedValueOnce(error);

        const { result, rerender } = renderHook(() => useAnnotationServices());

        await waitFor(() => {
            expect(result.current.policy).toBeNull();
            expect(result.current.policyError).toBe("500 Internal Server Error");
            expect(mockSetPromptedModel).toHaveBeenCalledWith("sam2-generic");
            expect(mockSetSuggestionModel).toHaveBeenCalledWith("sam3-suggestion-generic");
            expect(mockSetInstanceModel).toHaveBeenCalledWith("m2f-generic");
        });

        rerender();
        const instanceService = result.current.services.find((service) => service.key === "instance");
        expect(instanceService.onRun).toEqual(expect.any(Function));
    });

    it("does not overwrite a manual model selection after policy prefill", async () => {
        getInferenceRoutingPolicy.mockResolvedValueOnce({
            dataset_id: 101,
            bindings: [
                {
                    task: "prompted-segmentation",
                    label_id: 1,
                    model_registry_key: "sam2-cell",
                },
            ],
        });

        const { rerender } = renderHook(() => useAnnotationServices());
        await waitFor(() => {
            expect(mockSetPromptedModel).toHaveBeenCalledWith("sam2-cell");
        });
        const callsAfterPrefill = mockSetPromptedModel.mock.calls.length;

        mockState.promptedModel = "sam2-generic";
        rerender();

        expect(mockSetPromptedModel).toHaveBeenCalledTimes(callsAfterPrefill);
    });

    it("passes instance route inputs when the selected model matches the resolved route", async () => {
        const inputs = { parameters: { mask_threshold: 0.62 } };
        getInferenceRoutingPolicy.mockResolvedValueOnce({
            dataset_id: 101,
            bindings: [
                {
                    task: "instance-segmentation",
                    label_id: null,
                    model_registry_key: "m2f-routed-default",
                    inputs,
                },
            ],
        });

        const { result, rerender } = renderHook(() => useAnnotationServices());

        await waitFor(() => {
            expect(mockSetInstanceModel).toHaveBeenCalledWith("m2f-routed-default");
        });

        mockState.instanceModel = "m2f-routed-default";
        rerender();

        act(() => {
            result.current.confirmInstanceRun();
        });

        expect(mockRunInstance).toHaveBeenCalledWith("patch", inputs);
    });

    it("does not pass route inputs after manually selecting a different instance model", async () => {
        const inputs = { parameters: { mask_threshold: 0.62 } };
        getInferenceRoutingPolicy.mockResolvedValueOnce({
            dataset_id: 101,
            bindings: [
                {
                    task: "instance-segmentation",
                    label_id: null,
                    model_registry_key: "m2f-routed-default",
                    inputs,
                },
            ],
        });

        const { result, rerender } = renderHook(() => useAnnotationServices());

        await waitFor(() => {
            expect(mockSetInstanceModel).toHaveBeenCalledWith("m2f-routed-default");
        });

        mockState.instanceModel = "m2f-generic";
        rerender();

        act(() => {
            result.current.confirmInstanceRun();
        });

        expect(mockRunInstance).toHaveBeenCalledWith("patch");
        expect(mockRunInstance).not.toHaveBeenCalledWith("patch", inputs);
    });
});

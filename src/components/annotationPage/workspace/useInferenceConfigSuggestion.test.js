import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useInferenceConfigSuggestion } from "./useInferenceConfigSuggestion";
import * as inferenceApi from "../../../api/inference";
import * as contoursApi from "../../../api/contours";
import * as DatasetContext from "../../../contexts/DatasetContext";
import * as ToastContext from "../../../contexts/ToastContext";
import * as usePermissionsModule from "../../../hooks/usePermissions";
import * as annotationSelectors from "../../../stores/selectors/annotationSelectors";

vi.mock("../../../api/inference");
vi.mock("../../../api/contours");

describe("useInferenceConfigSuggestion", () => {
    const addToast = vi.fn();
    const setObjectsFromHierarchy = vi.fn();

    beforeEach(() => {
        vi.clearAllMocks();

        vi.spyOn(DatasetContext, "useDataset").mockReturnValue({
            currentDataset: { id: 10, name: "Test Dataset" },
        });
        vi.spyOn(ToastContext, "useToast").mockReturnValue({ addToast });
        vi.spyOn(usePermissionsModule, "usePermissions").mockReturnValue({
            can: vi.fn().mockReturnValue(true),
        });
        vi.spyOn(annotationSelectors, "useCurrentImageId").mockReturnValue(100);
        vi.spyOn(annotationSelectors, "useCurrentMaskId").mockReturnValue(50);
        vi.spyOn(annotationSelectors, "useDatasetLabelsMap").mockReturnValue(new Map([[1, "cell"]]));
        vi.spyOn(annotationSelectors, "useSetObjectsFromHierarchy").mockReturnValue(setObjectsFromHierarchy);

        inferenceApi.getInferenceModelCatalog.mockResolvedValue({
            models: [
                { registry_key: "m2f", task: "instance-segmentation", label_ids: [] },
                { registry_key: "sam3-cross", task: "cross-image-suggestion", label_ids: [] },
                { registry_key: "sam3-cross-default", task: "cross-image-suggestion", label_ids: [] },
            ],
        });
        inferenceApi.getInferenceRoutingPolicy.mockResolvedValue({
            dataset_id: 10,
            bindings: [{ task: "cross-image-suggestion", label_id: 1, model_registry_key: "sam3-cross" }],
        });
    });

    it("indexes configured steps and identifies configured labels", async () => {
        const { result } = renderHook(() => useInferenceConfigSuggestion());

        await waitFor(() => {
            expect(result.current.isConfigured(1)).toBe(true);
        });

        expect(result.current.isConfigured(2)).toBe(false);
        expect(result.current.isRunning(1)).toBe(false);
    });

    it("executes suggestion, refreshes contour hierarchy, and displays toast", async () => {
        inferenceApi.suggestModelRoutingStep.mockResolvedValue({
            dataset_id: 10,
            image_id: 100,
            label_id: 1,
            contours_created: 3,
            contours_suppressed: 1,
            contour_ids: [101, 102, 103],
        });

        const mockResponse = {
            success: true,
            message: "Contours hierarchy retrieved.",
            contours: { root_contours: [{ id: 101, label_id: 1 }] },
        };
        contoursApi.getContourHierarchy.mockResolvedValue(mockResponse);

        const { result } = renderHook(() => useInferenceConfigSuggestion());

        await waitFor(() => {
            expect(result.current.isConfigured(1)).toBe(true);
        });

        await act(async () => {
            await result.current.suggestLabel(1);
        });

        expect(inferenceApi.suggestModelRoutingStep).toHaveBeenCalledWith(
            expect.objectContaining({
                datasetId: 10,
                imageId: 100,
                maskId: 50,
                labelId: 1,
            })
        );
        expect(contoursApi.getContourHierarchy).toHaveBeenCalledWith(50);
        expect(setObjectsFromHierarchy).toHaveBeenCalledWith(
            { root_contours: [{ id: 101, label_id: 1 }] },
            expect.any(Map)
        );
        expect(addToast).toHaveBeenCalledWith({
            type: "success",
            message: "Suggested 3 objects (1 duplicate suppressed).",
        });
    });

    it("discards hierarchy refresh if user navigates to another image while suggestion is in-flight", async () => {
        let resolveInference;
        inferenceApi.suggestModelRoutingStep.mockImplementation(
            () =>
                new Promise((resolve) => {
                    resolveInference = resolve;
                })
        );

        let currentImageId = 100;
        vi.spyOn(annotationSelectors, "useCurrentImageId").mockImplementation(() => currentImageId);

        const { result, rerender } = renderHook(() => useInferenceConfigSuggestion());

        await waitFor(() => {
            expect(result.current.isConfigured(1)).toBe(true);
        });

        // Trigger suggestion on image 100
        let suggestPromise;
        act(() => {
            suggestPromise = result.current.suggestLabel(1);
        });

        // User switches to image 200 while request is in-flight
        currentImageId = 200;
        rerender();

        // Complete the in-flight inference request
        await act(async () => {
            resolveInference({
                dataset_id: 10,
                image_id: 100,
                label_id: 1,
                contours_created: 3,
                contours_suppressed: 0,
            });
            await suggestPromise;
        });

        expect(contoursApi.getContourHierarchy).not.toHaveBeenCalled();
        expect(setObjectsFromHierarchy).not.toHaveBeenCalled();
    });

    it("clears stale configuration and active running state immediately on dataset change", async () => {
        let currentDataset = { id: 10, name: "Dataset 10" };
        vi.spyOn(DatasetContext, "useDataset").mockImplementation(() => ({ currentDataset }));

        let resolveDataset20;
        inferenceApi.getInferenceRoutingPolicy.mockImplementation((dsId) => {
            if (dsId === 10) {
                return Promise.resolve({
                    dataset_id: 10,
                    bindings: [{ task: "cross-image-suggestion", label_id: 1, model_registry_key: "sam3-cross" }],
                });
            }
            return new Promise((resolve) => {
                resolveDataset20 = resolve;
            });
        });

        const { result, rerender } = renderHook(() => useInferenceConfigSuggestion());

        await waitFor(() => {
            expect(result.current.isConfigured(1)).toBe(true);
            expect(result.current.config?.dataset_id).toBe(10);
        });

        // Switch to Dataset 20 while API is in-flight
        currentDataset = { id: 20, name: "Dataset 20" };
        rerender();

        // Stale config and running state must be cleared immediately
        expect(result.current.config).toBeNull();
        expect(result.current.isConfigured(1)).toBe(false);
        expect(result.current.isAnyRunning).toBe(false);
        expect(result.current.isLoadingConfig).toBe(true);

        // Resolve new dataset config
        await act(async () => {
            resolveDataset20({
                dataset_id: 20,
                bindings: [{ task: "cross-image-suggestion", label_id: 2, model_registry_key: "sam3-cross" }],
            });
        });

        await waitFor(() => {
            expect(result.current.isLoadingConfig).toBe(false);
            expect(result.current.config?.dataset_id).toBe(20);
            expect(result.current.isConfigured(2)).toBe(true);
            expect(result.current.isConfigured(1)).toBe(false);
        });
    });

    it("suppresses error toast if user switches masks while suggestion is in-flight", async () => {
        let rejectInference;
        inferenceApi.suggestModelRoutingStep.mockImplementation(
            () =>
                new Promise((_, reject) => {
                    rejectInference = reject;
                })
        );

        let currentMaskId = 50;
        vi.spyOn(annotationSelectors, "useCurrentMaskId").mockImplementation(() => currentMaskId);

        const { result, rerender } = renderHook(() => useInferenceConfigSuggestion());

        await waitFor(() => {
            expect(result.current.isConfigured(1)).toBe(true);
        });

        let suggestPromise;
        act(() => {
            suggestPromise = result.current.suggestLabel(1);
        });

        // User switches mask to 60 while inference is in-flight
        currentMaskId = 60;
        rerender();

        // Inference fails
        await act(async () => {
            rejectInference(new Error("Model connection timeout on mask 50"));
            await suggestPromise;
        });

        // Stale mask error toast is suppressed
        expect(addToast).not.toHaveBeenCalled();
    });

    it("displays error toast if user remains on the same image and mask", async () => {
        inferenceApi.suggestModelRoutingStep.mockRejectedValue(
            new Error("Inference engine error")
        );

        const { result } = renderHook(() => useInferenceConfigSuggestion());

        await waitFor(() => {
            expect(result.current.isConfigured(1)).toBe(true);
        });

        await act(async () => {
            await result.current.suggestLabel(1);
        });

        expect(addToast).toHaveBeenCalledWith({
            type: "error",
            message: "Suggestion failed: Inference engine error",
        });
    });

    it("fences stale suggestion finally cleanup against newer requests across dataset changes", async () => {
        let currentDataset = { id: 10, name: "Dataset 10" };
        vi.spyOn(DatasetContext, "useDataset").mockImplementation(() => ({ currentDataset }));

        let resolveInferenceDs10;
        let resolveInferenceDs20;
        inferenceApi.suggestModelRoutingStep.mockImplementation(({ datasetId }) => {
            if (datasetId === 10) {
                return new Promise((resolve) => {
                    resolveInferenceDs10 = resolve;
                });
            }
            return new Promise((resolve) => {
                resolveInferenceDs20 = resolve;
            });
        });

        inferenceApi.getInferenceRoutingPolicy.mockImplementation((dsId) => {
            if (dsId === 10) {
                return Promise.resolve({
                    dataset_id: 10,
                    bindings: [{ task: "cross-image-suggestion", label_id: 1, model_registry_key: "sam3-cross" }],
                });
            }
            return Promise.resolve({
                dataset_id: 20,
                bindings: [{ task: "cross-image-suggestion", label_id: 2, model_registry_key: "sam3-cross" }],
            });
        });

        const { result, rerender } = renderHook(() => useInferenceConfigSuggestion());

        await waitFor(() => {
            expect(result.current.isConfigured(1)).toBe(true);
        });

        // 1. Start suggestion on Dataset 10 (label 1)
        let promise1;
        act(() => {
            promise1 = result.current.suggestLabel(1);
        });
        expect(result.current.isRunning(1)).toBe(true);

        // 2. User switches to Dataset 20 while suggestion 1 is in-flight
        currentDataset = { id: 20, name: "Dataset 20" };
        rerender();

        await waitFor(() => {
            expect(result.current.isConfigured(2)).toBe(true);
            expect(result.current.isAnyRunning).toBe(false);
        });

        // 3. Start suggestion on Dataset 20 (label 2)
        let promise2;
        act(() => {
            promise2 = result.current.suggestLabel(2);
        });
        expect(result.current.isRunning(2)).toBe(true);

        // 4. Stale request from Dataset 10 finishes
        await act(async () => {
            resolveInferenceDs10({
                dataset_id: 10,
                image_id: 100,
                label_id: 1,
                contours_created: 1,
            });
            await promise1;
        });

        // Stale completion must NOT have cleared the active running state of Dataset 20's suggestion
        expect(result.current.isRunning(2)).toBe(true);
        expect(result.current.isAnyRunning).toBe(true);

        // 5. Complete Dataset 20 suggestion
        await act(async () => {
            resolveInferenceDs20({
                dataset_id: 20,
                image_id: 100,
                label_id: 2,
                contours_created: 2,
            });
            await promise2;
        });

        // Now Dataset 20's running state is cleanly reset
        expect(result.current.isRunning(2)).toBe(false);
        expect(result.current.isAnyRunning).toBe(false);
    });

    it("discards hierarchy refresh and toast if component unmounts while suggestion is in-flight", async () => {
        let resolveInference;
        inferenceApi.suggestModelRoutingStep.mockImplementation(
            () =>
                new Promise((resolve) => {
                    resolveInference = resolve;
                })
        );

        const { result, unmount } = renderHook(() => useInferenceConfigSuggestion());

        await waitFor(() => {
            expect(result.current.isConfigured(1)).toBe(true);
        });

        let suggestPromise;
        act(() => {
            suggestPromise = result.current.suggestLabel(1);
        });

        // Component unmounts while inference is in-flight
        unmount();

        // In-flight inference resolves after unmount
        await act(async () => {
            resolveInference({
                dataset_id: 10,
                image_id: 100,
                label_id: 1,
                contours_created: 5,
            });
            await suggestPromise;
        });

        // Hierarchy refresh and toasts must NOT have been triggered after unmount
        expect(contoursApi.getContourHierarchy).not.toHaveBeenCalled();
        expect(setObjectsFromHierarchy).not.toHaveBeenCalled();
        expect(addToast).not.toHaveBeenCalled();
    });

    it("separates suggestion creation success from hierarchy refresh failure", async () => {
        inferenceApi.suggestModelRoutingStep.mockResolvedValue({
            dataset_id: 10,
            image_id: 100,
            label_id: 1,
            contours_created: 4,
            contours_suppressed: 0,
        });
        contoursApi.getContourHierarchy.mockRejectedValue(
            new Error("Network timeout loading hierarchy")
        );

        const { result } = renderHook(() => useInferenceConfigSuggestion());

        await waitFor(() => {
            expect(result.current.isConfigured(1)).toBe(true);
        });

        await act(async () => {
            await result.current.suggestLabel(1);
        });

        // 1. Success toast is displayed for the created annotations
        expect(addToast).toHaveBeenCalledWith({
            type: "success",
            message: "Suggested 4 objects.",
        });

        // 2. Warning toast is displayed for the canvas refresh failure, NOT a generic "Suggestion failed" error
        expect(addToast).toHaveBeenCalledWith({
            type: "warning",
            message: "Suggested annotations were saved, but canvas refresh failed. Reload the image to view them.",
        });
        expect(addToast).not.toHaveBeenCalledWith(
            expect.objectContaining({
                type: "error",
                message: expect.stringMatching(/suggestion failed/i),
            })
        );
    });

    it("exposes configError when configuration fetch fails", async () => {
        inferenceApi.getInferenceRoutingPolicy.mockRejectedValue(
            new Error("Internal Server Error (500)")
        );

        const { result } = renderHook(() => useInferenceConfigSuggestion());

        await waitFor(() => {
            expect(result.current.isLoadingConfig).toBe(false);
        });

        expect(result.current.config).toBeNull();
        expect(result.current.configError).toBe("Internal Server Error (500)");
        expect(result.current.isConfigured(1)).toBe(false);
    });

    it("resolves configured labels from canonical task-aware bindings", async () => {
        inferenceApi.getInferenceRoutingPolicy.mockResolvedValue({
            dataset_id: 10,
            bindings: [
                {
                    task: "cross-image-suggestion",
                    label_id: 1,
                    model_registry_key: "sam3-cross",
                },
                {
                    task: "instance-segmentation",
                    label_id: 2,
                    model_registry_key: "m2f",
                },
            ],
        });

        const { result } = renderHook(() => useInferenceConfigSuggestion());

        await waitFor(() => {
            expect(result.current.isConfigured(1)).toBe(true);
        });

        // Label 2 is only bound to instance-segmentation (not interactive cross-image-suggestion)
        expect(result.current.isConfigured(2)).toBe(false);
    });

    it("resolves task-default cross-image binding for unassigned dataset labels", async () => {
        inferenceApi.getInferenceRoutingPolicy.mockResolvedValue({
            dataset_id: 10,
            bindings: [
                {
                    task: "cross-image-suggestion",
                    label_id: null,
                    model_registry_key: "sam3-cross-default",
                },
            ],
        });

        const { result } = renderHook(() => useInferenceConfigSuggestion());

        await waitFor(() => {
            // Label 1 is in datasetLabelsMap, so it inherits the default cross-image binding
            expect(result.current.isConfigured(1)).toBe(true);
        });
    });

    it("does not apply class-aware cross-image task default to unsupported labels", async () => {
        vi.spyOn(annotationSelectors, "useDatasetLabelsMap").mockReturnValue(
            new Map([
                [1, "cell"],
                [2, "nucleus"],
            ])
        );
        inferenceApi.getInferenceModelCatalog.mockResolvedValueOnce({
            models: [
                {
                    registry_key: "sam3-cross-nucleus-only",
                    task: "cross-image-suggestion",
                    label_ids: [2], // predicts label 2 only
                },
            ],
        });
        inferenceApi.getInferenceRoutingPolicy.mockResolvedValue({
            dataset_id: 10,
            bindings: [
                {
                    task: "cross-image-suggestion",
                    label_id: null,
                    model_registry_key: "sam3-cross-nucleus-only",
                },
            ],
        });

        const { result } = renderHook(() => useInferenceConfigSuggestion());

        await waitFor(() => {
            // Label 2 (predicted by model) is configured
            expect(result.current.isConfigured(2)).toBe(true);
        });

        // Label 1 (NOT predicted by this class-aware model) is NOT configured
        expect(result.current.isConfigured(1)).toBe(false);
    });
});

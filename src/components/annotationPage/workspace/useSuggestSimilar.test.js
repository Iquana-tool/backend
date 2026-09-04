import { renderHook, act, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import useSuggestSimilar from "./useSuggestSimilar";

const mockState = {
  objectsList: [],
  selectedIds: [],
  suggestionModel: "sam3-default",
  availableModels: [
    { id: "sam3-default", name: "SAM 3 Default", task: "instance-suggestion", label_ids: [] },
    { id: "polyp-specialist", name: "Polyp Specialist", task: "instance-suggestion", label_ids: [2] },
  ],
  wsReady: true,
  isRunning: false,
  currentDataset: { id: "101" },
  favorites: {},
};

const mockRunSuggestion = vi.fn();

vi.mock("../../../stores/selectors/annotationSelectors", () => ({
  useObjectsList: () => mockState.objectsList,
  useSelectedObjects: () => mockState.selectedIds,
  useSuggestionModel: () => mockState.suggestionModel,
  useAvailableSuggestionModels: () => mockState.availableModels,
  useModelFavorites: () => mockState.favorites,
  useWebSocketIsReady: () => mockState.wsReady,
  useIsRunningSuggestion: () => mockState.isRunning,
}));

vi.mock("../../../hooks/useSuggestionSegmentation", () => ({
  useSuggestionSegmentation: () => ({
    runSuggestion: mockRunSuggestion,
    isRunning: mockState.isRunning,
  }),
}));

vi.mock("../../../contexts/DatasetContext", () => ({
  useDataset: () => ({ currentDataset: mockState.currentDataset }),
}));

vi.mock("../../../contexts/ToastContext", () => ({
  useToast: () => ({ addToast: vi.fn() }),
}));

vi.mock("../../../api/inference", () => ({
  getInferenceRoutingPolicy: vi.fn(),
}));

import { getInferenceRoutingPolicy } from "../../../api/inference";

describe("useSuggestSimilar dynamic exemplar routing", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState.objectsList = [
      { id: 1, label: "polyp", labelId: 2, contour_id: 101 },
      { id: 2, label: "polyp", labelId: 2, contour_id: 102 },
      { id: 3, label: "coral", labelId: 1, contour_id: 103 },
      { id: 4, label: "", labelId: null, contour_id: 104 },
    ];
    mockState.selectedIds = [1, 2];
    mockState.suggestionModel = "sam3-default";
    mockState.wsReady = true;
    mockState.isRunning = false;
    mockState.favorites = {};
  });

  it("resolves label-specific model when selected exemplars share a label", async () => {
    getInferenceRoutingPolicy.mockResolvedValueOnce({
      dataset_id: 101,
      bindings: [
        {
          task: "instance-suggestion",
          label_id: 2,
          model_registry_key: "polyp-specialist",
        },
      ],
    });

    const { result } = renderHook(() => useSuggestSimilar());

    await waitFor(() => {
      expect(result.current.eligible).toBe(true);
      expect(result.current.resolvedModelId).toBe("polyp-specialist");
    });

    await act(async () => {
      await result.current.run();
    });

    expect(mockRunSuggestion).toHaveBeenCalledWith([101, 102], 2, "polyp-specialist", null);
  });

  it("forwards saved route parameters and conditioning inputs to suggestion runner", async () => {
    getInferenceRoutingPolicy.mockResolvedValueOnce({
      dataset_id: 101,
      bindings: [
        {
          task: "instance-suggestion",
          label_id: 2,
          model_registry_key: "polyp-specialist",
          inputs: {
            parameters: {
              mask_threshold: 0.8,
              min_mask_area: 120,
            },
            conditioning: {
              count: 3,
            },
          },
        },
      ],
    });

    const { result } = renderHook(() => useSuggestSimilar());

    await waitFor(() => {
      expect(result.current.eligible).toBe(true);
      expect(result.current.resolvedModelId).toBe("polyp-specialist");
    });

    await act(async () => {
      await result.current.run();
    });

    expect(mockRunSuggestion).toHaveBeenCalledWith(
      [101, 102],
      2,
      "polyp-specialist",
      {
        parameters: {
          mask_threshold: 0.8,
          min_mask_area: 120,
        },
        conditioning: {
          count: 3,
        },
      }
    );
  });

  it("uses task default routing when selected exemplars are unlabelled", async () => {
    mockState.selectedIds = [4];

    getInferenceRoutingPolicy.mockResolvedValueOnce({
      dataset_id: 101,
      bindings: [
        {
          task: "instance-suggestion",
          label_id: 2,
          model_registry_key: "polyp-specialist",
        },
        {
          task: "instance-suggestion",
          label_id: null,
          model_registry_key: "sam3-default",
        },
      ],
    });

    const { result } = renderHook(() => useSuggestSimilar());

    await waitFor(() => {
      expect(result.current.eligible).toBe(true);
      expect(result.current.resolvedModelId).toBe("sam3-default");
    });

    await act(async () => {
      await result.current.run();
    });

    expect(mockRunSuggestion).toHaveBeenCalledWith(104, null, "sam3-default", null);
  });

  it("stays disabled while the dataset policy is loading", async () => {
    let resolvePolicy;
    const pendingPolicy = new Promise((resolve) => {
      resolvePolicy = resolve;
    });
    getInferenceRoutingPolicy.mockReturnValueOnce(pendingPolicy);

    const { result } = renderHook(() => useSuggestSimilar());

    expect(result.current.policyLoading).toBe(true);
    expect(result.current.eligible).toBe(false);

    resolvePolicy(null);
    await waitFor(() => expect(result.current.policyLoading).toBe(false));
    expect(result.current.eligible).toBe(true);
  });

  it("exposes policy failures while preserving the normal model fallback", async () => {
    getInferenceRoutingPolicy.mockRejectedValueOnce(new Error("policy unavailable"));

    const { result } = renderHook(() => useSuggestSimilar());

    await waitFor(() => expect(result.current.policyError).toBe("policy unavailable"));
    expect(result.current.eligible).toBe(true);
    expect(result.current.resolvedModelId).toBe("sam3-default");
    expect(result.current.reason).toBeNull();
  });

  it("clears the old policy while switching datasets", async () => {
    getInferenceRoutingPolicy.mockResolvedValueOnce({
      dataset_id: 101,
      bindings: [
        {
          task: "instance-suggestion",
          label_id: 2,
          model_registry_key: "polyp-specialist",
        },
      ],
    });

    const { result, rerender } = renderHook(() => useSuggestSimilar());
    await waitFor(() => expect(result.current.resolvedModelId).toBe("polyp-specialist"));

    let resolveNewPolicy;
    const pendingNewPolicy = new Promise((resolve) => {
      resolveNewPolicy = resolve;
    });
    getInferenceRoutingPolicy.mockReturnValueOnce(pendingNewPolicy);
    mockState.currentDataset = { id: "202" };
    rerender();

    expect(result.current.policy).toBeNull();
    expect(result.current.policyLoading).toBe(true);
    expect(result.current.eligible).toBe(false);

    resolveNewPolicy(null);
    await waitFor(() => expect(result.current.policyLoading).toBe(false));
    expect(result.current.resolvedModelId).toBe("sam3-default");
  });

  it("uses the personal favorite after a routed label becomes unrouted", async () => {
    mockState.favorites = { "instance-suggestion": "sam3-default" };
    getInferenceRoutingPolicy.mockResolvedValueOnce({
      dataset_id: 101,
      bindings: [
        {
          task: "instance-suggestion",
          label_id: 2,
          model_registry_key: "polyp-specialist",
        },
      ],
    });

    const { result, rerender } = renderHook(() => useSuggestSimilar());
    await waitFor(() => expect(result.current.resolvedModelId).toBe("polyp-specialist"));

    mockState.selectedIds = [3];
    rerender();

    await waitFor(() => expect(result.current.resolvedModelId).toBe("sam3-default"));
    expect(result.current.eligible).toBe(true);
  });

  it("preserves a compatible manual selection when no route exists", async () => {
    mockState.suggestionModel = "polyp-specialist";
    getInferenceRoutingPolicy.mockResolvedValueOnce(null);

    const { result } = renderHook(() => useSuggestSimilar());

    await waitFor(() => expect(result.current.resolvedModelId).toBe("polyp-specialist"));
  });

  it("does not fall back when an explicit route is stale", async () => {
    getInferenceRoutingPolicy.mockResolvedValueOnce({
      dataset_id: 101,
      bindings: [
        {
          task: "instance-suggestion",
          label_id: 2,
          model_registry_key: "retired-specialist",
        },
      ],
    });

    const { result } = renderHook(() => useSuggestSimilar());

    await waitFor(() => expect(result.current.reason).toContain("no longer available"));
    expect(result.current.resolvedModelId).toBeNull();
    expect(result.current.eligible).toBe(false);
  });
});

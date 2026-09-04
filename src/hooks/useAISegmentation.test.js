import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import useAISegmentation from './useAISegmentation';
import annotationSession from '../services/annotationSession';
import { getInferenceRoutingPolicy } from '../api/inference';

const hookState = vi.hoisted(() => ({
  currentDataset: { id: 42 },
  routeDatasetId: '42',
  currentImage: { dataset_id: 42 },
  imageObject: { width: 100, height: 100 },
  prompts: [{ type: 'point', coords: { x: 25, y: 50 }, label: 'positive' }],
  promptedModelId: 'routed-model',
  activeLabelId: null,
  availablePromptedModels: [],
}));

const actions = vi.hoisted(() => ({
  consumePrompts: vi.fn(),
  setIsSubmitting: vi.fn(),
  addObject: vi.fn(),
  updateObject: vi.fn(),
  exitRefinementMode: vi.fn(),
  setCurrentTool: vi.fn(),
  setPromptedModel: vi.fn(),
  syncEditModeDraftFromRefinement: vi.fn(),
}));

vi.mock('react-router-dom', () => ({
  useParams: () => ({ datasetId: hookState.routeDatasetId }),
}));

vi.mock('../contexts/DatasetContext', () => ({
  useDataset: () => ({ currentDataset: hookState.currentDataset }),
}));

vi.mock('../stores/selectors/annotationSelectors', () => ({
  useAIPrompts: () => hookState.prompts,
  usePromptedModel: () => hookState.promptedModelId,
  useCurrentImage: () => hookState.currentImage,
  useConsumePrompts: () => actions.consumePrompts,
  useSetIsSubmittingAI: () => actions.setIsSubmitting,
  useImageObject: () => hookState.imageObject,
  useAddObject: () => actions.addObject,
  useUpdateObject: () => actions.updateObject,
  useObjectsList: () => [],
  useRefinementModeActive: () => false,
  useRefinementModeObjectId: () => null,
  useExitRefinementMode: () => actions.exitRefinementMode,
  useSetCurrentTool: () => actions.setCurrentTool,
  useSetPromptedModel: () => actions.setPromptedModel,
  useSyncEditModeDraftFromRefinement: () => actions.syncEditModeDraftFromRefinement,
  useAvailablePromptedModels: () => hookState.availablePromptedModels,
  useActiveLabelId: () => hookState.activeLabelId,
}));

vi.mock('../services/annotationSession', () => ({
  default: {
    isReady: vi.fn(),
    isServiceAvailable: vi.fn(),
    runSegmentation: vi.fn(),
    unselectRefinementObject: vi.fn(),
  },
}));

vi.mock('../api/inference', () => ({
  getInferenceRoutingPolicy: vi.fn(),
}));

const makeModel = (id) => ({
  id,
  registry_key: id,
  task: 'prompted-segmentation',
  label_ids: [],
});

const makePolicy = (modelKey = 'routed-model', inputs = { parameters: { mask_threshold: 0.62 } }) => ({
  dataset_id: 42,
  bindings: [
    {
      task: 'prompted-segmentation',
      label_id: null,
      model_registry_key: modelKey,
      inputs,
    },
  ],
});

const settlePolicy = async () => {
  await act(async () => {
    await Promise.resolve();
  });
};

describe('useAISegmentation routing inputs', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    hookState.currentDataset = { id: 42 };
    hookState.routeDatasetId = '42';
    hookState.currentImage = { dataset_id: 42 };
    hookState.imageObject = { width: 100, height: 100 };
    hookState.prompts = [{ type: 'point', coords: { x: 25, y: 50 }, label: 'positive' }];
    hookState.promptedModelId = 'routed-model';
    hookState.activeLabelId = null;
    hookState.availablePromptedModels = [makeModel('routed-model'), makeModel('manual-model')];

    getInferenceRoutingPolicy.mockResolvedValue(null);
    annotationSession.isReady.mockReturnValue(true);
    annotationSession.isServiceAvailable.mockReturnValue(true);
    annotationSession.runSegmentation.mockResolvedValue({ success: true });
  });

  it('sends saved inputs when the selected model matches the resolved binding', async () => {
    const inputs = { parameters: { mask_threshold: 0.62 } };
    getInferenceRoutingPolicy.mockResolvedValue(makePolicy('routed-model', inputs));
    const { result } = renderHook(() => useAISegmentation());

    await settlePolicy();
    await act(async () => {
      await result.current.runSegmentation();
    });

    expect(annotationSession.runSegmentation).toHaveBeenCalledWith(
      'routed-model',
      expect.objectContaining({
        point_prompts: [{ x: 0.25, y: 0.5, label: true }],
      }),
      inputs
    );
  });

  it('does not mistake a click event for explicit routing inputs', async () => {
    const inputs = { parameters: { mask_threshold: 0.62 } };
    getInferenceRoutingPolicy.mockResolvedValue(makePolicy('routed-model', inputs));
    const { result } = renderHook(() => useAISegmentation());

    await settlePolicy();
    await act(async () => {
      await result.current.runSegmentation({
        type: 'click',
        preventDefault: vi.fn(),
        nativeEvent: {},
      });
    });

    expect(annotationSession.runSegmentation).toHaveBeenCalledWith(
      'routed-model',
      expect.any(Object),
      inputs
    );
  });

  it('still accepts an explicit routing-input envelope', async () => {
    const inputs = { parameters: { mask_threshold: 0.8 } };
    const { result } = renderHook(() => useAISegmentation());

    await act(async () => {
      await result.current.runSegmentation(inputs);
    });

    expect(annotationSession.runSegmentation).toHaveBeenCalledWith(
      'routed-model',
      expect.any(Object),
      inputs
    );
  });

  it('does not send the binding inputs after a manual model switch', async () => {
    const inputs = { parameters: { mask_threshold: 0.62 } };
    getInferenceRoutingPolicy.mockResolvedValue(makePolicy('routed-model', inputs));
    hookState.promptedModelId = 'manual-model';
    const { result } = renderHook(() => useAISegmentation());

    await settlePolicy();
    await act(async () => {
      await result.current.runSegmentation();
    });

    expect(annotationSession.runSegmentation).toHaveBeenCalledWith(
      'manual-model',
      expect.any(Object),
      null
    );
  });

  it('blocks a policy-prefilled model while the local policy is still loading', async () => {
    let resolvePolicy;
    const policyPromise = new Promise((resolve) => {
      resolvePolicy = resolve;
    });
    getInferenceRoutingPolicy.mockReturnValue(policyPromise);
    const { result } = renderHook(() => useAISegmentation());

    let runResult;
    await act(async () => {
      runResult = await result.current.runSegmentation();
    });

    expect(runResult).toEqual({
      success: false,
      error: 'Routing policy is still loading. Please try again in a moment.',
    });
    expect(annotationSession.runSegmentation).not.toHaveBeenCalled();

    await act(async () => {
      resolvePolicy(makePolicy('routed-model'));
      await policyPromise;
    });
    await act(async () => {
      await result.current.runSegmentation();
    });

    expect(annotationSession.runSegmentation).toHaveBeenCalledWith(
      'routed-model',
      expect.any(Object),
      { parameters: { mask_threshold: 0.62 } }
    );
  });

  it('preserves manual fallback execution when the loaded policy has no binding', async () => {
    hookState.promptedModelId = 'manual-model';
    getInferenceRoutingPolicy.mockResolvedValue({ dataset_id: 42, bindings: [] });
    const { result } = renderHook(() => useAISegmentation());

    await settlePolicy();
    await act(async () => {
      await result.current.runSegmentation();
    });

    expect(annotationSession.runSegmentation).toHaveBeenCalledWith(
      'manual-model',
      expect.any(Object),
      null
    );
  });

  it('preserves manual fallback execution when policy loading fails', async () => {
    hookState.promptedModelId = 'manual-model';
    getInferenceRoutingPolicy.mockRejectedValue(new Error('policy unavailable'));
    const { result } = renderHook(() => useAISegmentation());

    await settlePolicy();
    await act(async () => {
      await result.current.runSegmentation();
    });

    expect(annotationSession.runSegmentation).toHaveBeenCalledWith(
      'manual-model',
      expect.any(Object),
      null
    );
  });
});

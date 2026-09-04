import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useSuggestionSegmentation } from './useSuggestionSegmentation';
import annotationSession from '../services/annotationSession';

const addToast = vi.hoisted(() => vi.fn());
const setIsRunning = vi.hoisted(() => vi.fn());

vi.mock('../stores/selectors/annotationSelectors', () => ({
  useAvailableSuggestionModels: () => [
    { id: 'personal-default', name: 'Personal Default' },
    { id: 'routed-specialist', name: 'Routed Specialist' },
  ],
  useSuggestionModel: () => 'personal-default',
  useIsRunningSuggestion: () => false,
  useSetIsRunningSuggestion: () => setIsRunning,
}));

vi.mock('../contexts/ToastContext', () => ({
  useToast: () => ({ addToast }),
}));

vi.mock('../services/annotationSession', () => ({
  default: {
    isServiceAvailable: vi.fn(),
    runSuggestion: vi.fn(),
  },
}));

describe('useSuggestionSegmentation model override', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    annotationSession.isServiceAvailable.mockReturnValue(true);
    annotationSession.runSuggestion.mockResolvedValue({
      success: true,
      data: { added_count: 0 },
    });
  });

  it('executes and reports zero results using the routed override model', async () => {
    const { result } = renderHook(() => useSuggestionSegmentation());

    await act(async () => {
      await result.current.runSuggestion([11, 12], 2, 'routed-specialist');
    });

    expect(annotationSession.runSuggestion).toHaveBeenCalledWith(
      [11, 12],
      'routed-specialist',
      2,
      null
    );
    expect(addToast).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'info',
        message: expect.stringContaining('Routed Specialist'),
      })
    );
  });

  it('forwards inputs to annotationSession.runSuggestion', async () => {
    const { result } = renderHook(() => useSuggestionSegmentation());
    const inputs = { parameters: { mask_threshold: 0.75 } };

    await act(async () => {
      await result.current.runSuggestion([11], 2, 'routed-specialist', inputs);
    });

    expect(annotationSession.runSuggestion).toHaveBeenCalledWith(
      [11],
      'routed-specialist',
      2,
      inputs
    );
  });
});

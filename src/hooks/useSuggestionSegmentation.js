import { useCallback } from 'react';
import annotationSession from '../services/annotationSession';
import {
  useAvailableSuggestionModels,
  useSuggestionModel,
  useIsRunningSuggestion,
  useSetIsRunningSuggestion,
} from "../stores/selectors/annotationSelectors";
import { useToast } from '../contexts/ToastContext';

/**
 * Hook for running suggestion segmentation to find similar instances
 * 
 * @param {Function} onSuccess - Optional callback on success
 * @param {Function} onError - Optional callback on error
 * @returns {Object} - { runSuggestion, isRunning }
 * 
 * runSuggestion accepts either a single contour ID or an array of contour IDs as seeds
 */
export function useSuggestionSegmentation(onSuccess, onError) {
  const isRunning = useIsRunningSuggestion();
  const setIsRunning = useSetIsRunningSuggestion();
  const suggestionModelId = useSuggestionModel(); // This is a string ID, not an object
  const availableModels = useAvailableSuggestionModels();
  const { addToast } = useToast();

  const runSuggestion = useCallback(async (contourIdOrIds, labelId, modelIdOverride = null, inputs = null) => {
    // Note: Model status is handled by the backend, no need to update here

    // Normalize to array
    const contourIds = Array.isArray(contourIdOrIds) ? contourIdOrIds : [contourIdOrIds];

    if (!contourIds || contourIds.length === 0) {
      const error = new Error('Contour ID(s) required');
      if (onError) {
        onError(error);
      }
      return;
    }

    const effectiveModelId = modelIdOverride || suggestionModelId;

    if (!effectiveModelId) {
      const error = new Error('Please select a suggestion model first');
      if (onError) {
        onError(error);
      } else {
        alert('Please select a suggestion model first');
      }
      return;
    }

    // Check if suggestion service is available
    if (!annotationSession.isServiceAvailable('suggestion_segmentation')) {
      const error = new Error('Suggestion segmentation service is not available');
      if (onError) {
        onError(error);
      } else {
        alert('Suggestion segmentation service is not available. Please check your connection.');
      }
      return;
    }

    setIsRunning(true);

    try {
      // Call WebSocket method - objects will be added automatically via OBJECT_ADDED messages
      // effectiveModelId is already the string identifier we need
      const response = await annotationSession.runSuggestion(
        contourIds,         // Array of seed contour IDs (can be single or multiple)
        effectiveModelId,   // Model identifier (string)
        labelId,            // Label ID
        inputs              // Saved inputs from route policy (conditioning and parameters)
      );

      // Note: Model status is handled by the backend, no need to update here
      if (!response.success) {
        throw new Error(response.message || 'Suggestion failed');
      }

      // Let the user know when the model returned no new instances. The found
      // objects (if any) arrive separately as OBJECT_ADDED messages, but the
      // ack reports how many were found.
      if (response?.data?.added_count === 0) {
        const model = availableModels.find(
          (m) => (m?.id || m?.registry_key || m?.identifier) === effectiveModelId
        );
        const modelName = model?.name || effectiveModelId;
        addToast({
          type: 'info',
          message: `${modelName} did not find any new instances. Try another model or add more exemplars!`,
          duration: 6000,
        });
      }
    } catch (error) {
      console.error('Suggestion segmentation error:', error);

      // Call error callback if provided
      if (onError) {
        onError(error);
      } else {
        alert(`Failed to suggest similar instances: ${error.message || 'Unknown error'}`);
      }
    } finally {
      setIsRunning(false);
    }
  }, [suggestionModelId, availableModels, addToast, onSuccess, onError]);

  return { runSuggestion, isRunning };
}

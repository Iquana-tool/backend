import { useCallback } from 'react';
import annotationSession from '../services/annotationSession';
import {useInstanceModel, useIsRunningInstance, useSetIsRunningInstance} from "../stores/selectors/annotationSelectors";

/**
 * Hook for running instance segmentation inference
 *
 * @param {Function} onSuccess - Optional callback on successful completion
 * @param {Function} onError - Optional callback on error
 * @returns {Object} - { runInstance, isRunning }
 */
export function useInstanceSegmentation(onSuccess, onError) {
  const isRunning = useIsRunningInstance();
  const setIsRunning = useSetIsRunningInstance();
  const instanceModelId = useInstanceModel(); // This is a string ID, not an object

  const runInstance = useCallback(async (writeMode = 'patch', inputs = null) => {
    if (!instanceModelId) {
      const error = new Error('Please select an instance segmentation model first');
      if (onError) {
        onError(error);
      } else {
        alert('Please select an instance segmentation model first');
      }
      return;
    }

    // Check if instance service is available
    if (!annotationSession.isServiceAvailable('instance_segmentation')) {
      const error = new Error('Instance segmentation service is not available');
      if (onError) {
        onError(error);
      } else {
        alert('Instance segmentation service is not available. Please check your connection.');
      }
      return;
    }

    setIsRunning(true);

    try {
      // Call WebSocket method - objects will be added automatically via OBJECT_ADDED messages
      // instanceModelId is already the string identifier we need
      const response = await annotationSession.runInstance(
        instanceModelId, // Model identifier (string)
        writeMode,
        inputs
      );

      if (!response.success) {
        throw new Error(response.message || 'Instance segmentation failed');
      }

      // Call success callback if provided
      if (onSuccess) {
        onSuccess(response);
      }
    } catch (error) {
      console.error('Instance segmentation error:', error);

      // Call error callback if provided
      if (onError) {
        onError(error);
      } else {
        alert(`Failed to run instance segmentation: ${error.message || 'Unknown error'}`);
      }
    } finally {
      setIsRunning(false);
    }
  }, [instanceModelId, onSuccess, onError, setIsRunning]);

  return { runInstance, isRunning };
}

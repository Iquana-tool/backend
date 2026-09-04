import { useEffect, useCallback, useMemo } from 'react';
import {
  useCurrentTool,
  useAIPrompts,
  usePromptedModel,
  useIsSubmitting,
  useSelectedObjects,
  useObjectsList,
  useRemoveObject,
  useRemoveLastPrompt,
  useClearSelection,
  useSetInstanceRunRequested,
  useInstanceWarningModalOpen,
  useRefinementModeActive,
  useWorkspaceMode,
} from '../stores/selectors/annotationSelectors';
import useAISegmentation from './useAISegmentation';
import useSuggestSimilar from '../components/annotationPage/workspace/useSuggestSimilar';
import { deleteObject } from '../utils/objectOperations';

/**
 * Action keyboard shortcuts for the annotation page.
 *
 * Tool selection, panel toggles, zoom and image navigation live in
 * useWorkspaceShortcuts; this hook owns the keys that trigger annotation work.
 *
 * - Enter: Run primary action (AI segmentation when in AI tool with prompts).
 *   Not in review mode — there the primary action is approving the instance
 *   under review, which the action bar owns along with the review cursor.
 * - 1: Run Prompted Segmentation
 * - 2: Run Instance Suggestion (suggestion) with selected objects as seeds
 * - 3: Open Instance Segmentation (warning modal)
 * - Delete/Backspace: In refinement mode with prompts, remove last prompt; otherwise reject selected objects, or remove last prompt when in AI tool with no selection
 */
export default function useAnnotationKeyboardShortcuts() {
  const currentTool = useCurrentTool();
  const prompts = useAIPrompts();
  const promptedModel = usePromptedModel();
  const isSubmitting = useIsSubmitting();
  const selectedIds = useSelectedObjects(); // store holds selected object IDs
  const objectsList = useObjectsList();
  const selectedObjects = useMemo(
    () => objectsList.filter((obj) => selectedIds.includes(obj.id)),
    [objectsList, selectedIds]
  );
  const removeObject = useRemoveObject();
  const removeLastPrompt = useRemoveLastPrompt();
  const clearSelection = useClearSelection();
  const setInstanceRunRequested = useSetInstanceRunRequested();
  const instanceWarningModalOpen = useInstanceWarningModalOpen();
  const refinementModeActive = useRefinementModeActive();
  const mode = useWorkspaceMode();

  const { runSegmentation } = useAISegmentation();
  const suggestSimilar = useSuggestSimilar();
  const runInstanceRequest = setInstanceRunRequested;

  const canRunPrompted =
    currentTool === 'ai_annotation' &&
    promptedModel &&
    !isSubmitting &&
    prompts.length > 0 &&
    !instanceWarningModalOpen;

  const handleRejectSelected = useCallback(async () => {
    if (selectedObjects.length === 0) return;
    for (const obj of selectedObjects) {
      try {
        await deleteObject(obj, removeObject);
      } catch (err) {
        console.error('Reject object failed:', err);
        alert(`Failed to reject object: ${err.message || 'Unknown error'}`);
        break;
      }
    }
    clearSelection();
  }, [selectedObjects, removeObject, clearSelection]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (
        e.target.tagName === 'INPUT' ||
        e.target.tagName === 'TEXTAREA' ||
        e.target.isContentEditable
      ) {
        return;
      }

      const isModifier = e.ctrlKey || e.metaKey || e.altKey;

      switch (e.key) {
        case 'Enter': {
          // Shift+Enter belongs to the action bar's "Add this object"; running
          // the model here as well would do both at once.
          if (canRunPrompted && mode !== 'review' && !e.shiftKey) {
            e.preventDefault();
            runSegmentation();
          }
          break;
        }
        case '1': {
          if (!isModifier && canRunPrompted) {
            e.preventDefault();
            runSegmentation();
          }
          break;
        }
        case '2': {
          if (isModifier) break;
          if (!suggestSimilar.eligible) break;
          e.preventDefault();
          suggestSimilar.run();
          break;
        }
        case '3': {
          if (!isModifier) {
            e.preventDefault();
            runInstanceRequest(true);
          }
          break;
        }
        case 'Delete':
        case 'Backspace': {
          // In refinement mode with prompts: erase last prompt (don't reject the contour being refined)
          if (
            currentTool === 'ai_annotation' &&
            refinementModeActive &&
            prompts.length > 0
          ) {
            e.preventDefault();
            removeLastPrompt();
          } else if (selectedObjects.length > 0) {
            e.preventDefault();
            handleRejectSelected();
          } else if (currentTool === 'ai_annotation') {
            e.preventDefault();
            removeLastPrompt();
          }
          break;
        }
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    canRunPrompted,
    mode,
    runSegmentation,
    selectedObjects,
    suggestSimilar,
    runInstanceRequest,
    handleRejectSelected,
    currentTool,
    refinementModeActive,
    prompts.length,
    removeLastPrompt,
  ]);
}

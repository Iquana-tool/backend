import { useCallback, useMemo } from 'react';
import {
  useObjectsList,
  useSelectedObjects,
  useSuggestionModel,
  useAvailableSuggestionModels,
  useModelFavorites,
  useWebSocketIsReady,
  useIsRunningSuggestion,
} from '../../../stores/selectors/annotationSelectors';
import { useSuggestionSegmentation } from '../../../hooks/useSuggestionSegmentation';
import { useDataset } from '../../../contexts/DatasetContext';
import { useToast } from '../../../contexts/ToastContext';
import { hasValidLabel } from '../../../stores/utils/labelValidation';
import { useAnnotationRoutingPolicy } from '../../../contexts/AnnotationRoutingPolicyContext';
import {
  isModelCompatibleWithLabel,
  matchesModelKey,
  resolveRoutingBinding,
} from '../../../utils/inferenceRouting';

/**
 * Stable identity for an object's class. Unlabelled objects collapse to one
 * shared key so a purely unlabelled selection still counts as homogeneous.
 */
const getClassKey = (object) => {
  if (!hasValidLabel(object.label)) return '__unlabelled__';
  if (object.labelId != null) return `id:${object.labelId}`;
  return `name:${String(object.label).trim()}`;
};

const getModelKey = (m) => m?.id || m?.registry_key || m?.identifier || null;

/**
 * "Suggest similar instances" — an action on the current selection rather than
 * a tool, as the redesign specifies.
 *
 * Eligibility is unchanged from the old floating button: the selection must be
 * non-empty, all of one class (or all unlabelled), and every object must expose
 * a contour id to seed from.
 */
export default function useSuggestSimilar() {
  const objectsList = useObjectsList();
  const selectedIds = useSelectedObjects();
  const suggestionModel = useSuggestionModel();
  const availableModels = useAvailableSuggestionModels();
  const favorites = useModelFavorites();
  const wsReady = useWebSocketIsReady();
  const isRunning = useIsRunningSuggestion();
  const { currentDataset } = useDataset();
  const datasetId = currentDataset?.id;
  const { addToast } = useToast();

  const {
    policy,
    policyReady,
    policyResolved,
    policyLoading,
    policyError,
  } = useAnnotationRoutingPolicy(datasetId);

  const { runSuggestion } = useSuggestionSegmentation(null, (error) =>
    addToast({
      type: 'error',
      message: `Failed to suggest similar instances: ${error.message || 'Unknown error'}`,
    })
  );

  const targets = useMemo(
    () => objectsList.filter((object) => selectedIds.includes(object.id)),
    [objectsList, selectedIds]
  );

  const classKeys = useMemo(() => new Set(targets.map(getClassKey)), [targets]);

  const contourIds = useMemo(
    () => targets.map((object) => object.contour_id).filter((id) => id != null),
    [targets]
  );

  const isHomogeneous = targets.length > 0 && classKeys.size === 1;
  const hasSeeds = contourIds.length === targets.length && contourIds.length > 0;

  // Determine exemplar shared label if any
  const sharedLabelId = useMemo(() => {
    if (!isHomogeneous || targets.length === 0) return null;
    const first = targets[0];
    return hasValidLabel(first.label) && first.labelId != null ? Number(first.labelId) : null;
  }, [isHomogeneous, targets]);

  // Resolve the selected exemplar label. A configured binding is authoritative;
  // only the absence of a binding permits the personal favorite/first fallback.
  const routing = useMemo(() => {
    if (!policyResolved || availableModels.length === 0) {
      return { modelId: null, inputs: null, error: null };
    }

    const resolved = policyReady
      ? resolveRoutingBinding(
          policy,
          'instance-suggestion',
          sharedLabelId,
          availableModels
        )
      : null;

    if (resolved?.binding) {
      if (resolved.isCompatible && !resolved.isStale && resolved.model) {
        return {
          modelId: getModelKey(resolved.model),
          inputs: resolved.binding.inputs || null,
          error: null,
        };
      }
      return {
        modelId: null,
        inputs: null,
        error: resolved.isStale
          ? 'The configured suggestion model is no longer available'
          : 'The configured suggestion model does not support the selected label',
      };
    }

    const favoriteKey = favorites?.['instance-suggestion'];
    const manualSelection = suggestionModel
      ? availableModels.find((model) =>
          matchesModelKey(model, 'instance-suggestion', suggestionModel)
        )
      : null;
    const favorite = favoriteKey
      ? availableModels.find((model) => matchesModelKey(model, 'instance-suggestion', favoriteKey))
      : null;
    const compatibleModels = availableModels.filter((model) =>
      isModelCompatibleWithLabel(model, sharedLabelId)
    );
    const fallback =
      (manualSelection && isModelCompatibleWithLabel(manualSelection, sharedLabelId)
        ? manualSelection
        : null) ||
      (favorite && isModelCompatibleWithLabel(favorite, sharedLabelId) ? favorite : null) ||
      compatibleModels[0];

    return { modelId: getModelKey(fallback), inputs: null, error: null };
  }, [policy, policyReady, policyResolved, sharedLabelId, availableModels, favorites, suggestionModel]);

  const resolvedModelId = routing.modelId;
  const resolvedInputs = routing.inputs;
  const routingError = routing.error;

  const eligible =
    isHomogeneous &&
    hasSeeds &&
    policyResolved &&
    !policyLoading &&
    !routingError &&
    !!resolvedModelId &&
    wsReady &&
    !isRunning;

  const reason = !isHomogeneous
    ? 'Select samples of the same class (or all unlabelled)'
    : !hasSeeds
      ? 'Selected objects are missing contour data'
      : !policyResolved && policyLoading
        ? 'Loading model routing policy'
        : routingError
            ? routingError
            : !resolvedModelId
              ? 'Select an Instance Suggestion model first'
              : !wsReady
                ? 'Connection not ready'
                : null;

  const run = useCallback(async () => {
    if (!eligible || !resolvedModelId) return;
    await runSuggestion(
      contourIds.length === 1 ? contourIds[0] : contourIds,
      sharedLabelId,
      resolvedModelId,
      resolvedInputs
    );
  }, [eligible, contourIds, sharedLabelId, resolvedModelId, resolvedInputs, runSuggestion]);

  return {
    eligible,
    reason,
    isRunning,
    run,
    seedCount: contourIds.length,
    resolvedModelId,
    policy,
    policyLoading,
    policyError,
  };
}

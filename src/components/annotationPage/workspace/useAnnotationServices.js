import { useEffect, useMemo, useState } from 'react';
import {
  useAvailablePromptedModels,
  useAvailableSuggestionModels,
  useAvailableInstanceModels,
  useFetchAvailablePromptedModels,
  useFetchAvailableSuggestionModels,
  useFetchAvailableInstanceModels,
  useIsLoadingPromptedModels,
  useIsLoadingSuggestionModels,
  useIsLoadingInstanceModels,
  useIsRunningSuggestion,
  useIsRunningInstance,
  usePromptedModel,
  useSuggestionModel,
  useInstanceModel,
  useSetPromptedModel,
  useSetSuggestionModel,
  useSetInstanceModel,
  useInstanceRunRequested,
  useSetInstanceRunRequested,
  useSetInstanceWarningModalOpen,
  useActiveLabelId,
  useModelFavorites,
} from '../../../stores/selectors/annotationSelectors';
import { useDataset } from '../../../contexts/DatasetContext';
import { useAnnotationRoutingPolicy } from '../../../contexts/AnnotationRoutingPolicyContext';
import {
  resolveRoutingBinding,
  isModelCompatibleWithLabel,
  matchesModelKey,
} from '../../../utils/inferenceRouting';
import annotationSession from '../../../services/annotationSession';
import useModelSwitchPreloader from '../../../hooks/useModelSwitchPreloader';
import { useInstanceSegmentation } from '../../../hooks/useInstanceSegmentation';

const getModelKey = (model) => model?.id || model?.registry_key || model?.identifier || null;

const getFallbackModelId = (models, task, favoriteKey, labelId = null) => {
  const compatibleModels = (models || []).filter((model) =>
    isModelCompatibleWithLabel(model, labelId)
  );
  const favorite = favoriteKey
    ? compatibleModels.find((model) => matchesModelKey(model, task, favoriteKey))
    : null;
  return getModelKey(favorite || compatibleModels[0]);
};

const isUsableBinding = (resolved) =>
  Boolean(resolved?.binding && resolved.model && resolved.isCompatible && !resolved.isStale);

const getMatchingBindingInputs = (resolved, task, selectedModel) => {
  if (!isUsableBinding(resolved) || !matchesModelKey(resolved.model, task, selectedModel)) {
    return null;
  }

  return resolved.binding?.inputs ?? null;
};

/**
 * Model lists, selections, dataset policy defaults, preloading and the instance-segmentation
 * write-mode flow for the three annotation services.
 */
export default function useAnnotationServices() {
  const { currentDataset } = useDataset();
  const datasetId = currentDataset?.id;
  const activeLabelId = useActiveLabelId();
  const favorites = useModelFavorites();
  const {
    policy,
    policyReady,
    policyResolved,
    policyLoading,
    policyError,
  } = useAnnotationRoutingPolicy(datasetId);

  const fetchPromptedModels = useFetchAvailablePromptedModels();
  const fetchSuggestionModels = useFetchAvailableSuggestionModels();
  const fetchInstanceModels = useFetchAvailableInstanceModels();

  const availablePromptedModels = useAvailablePromptedModels();
  const availableSuggestionModels = useAvailableSuggestionModels();
  const availableInstanceModels = useAvailableInstanceModels();

  const promptedModel = usePromptedModel();
  const suggestionModel = useSuggestionModel();
  const instanceModel = useInstanceModel();
  const setPromptedModel = useSetPromptedModel();
  const setSuggestionModel = useSetSuggestionModel();
  const setInstanceModel = useSetInstanceModel();

  const isLoadingPrompted = useIsLoadingPromptedModels();
  const isLoadingSuggestion = useIsLoadingSuggestionModels();
  const isLoadingInstance = useIsLoadingInstanceModels();
  const isRunningSuggestion = useIsRunningSuggestion();
  const isRunningInstance = useIsRunningInstance();

  const instanceRunRequested = useInstanceRunRequested();
  const setInstanceRunRequested = useSetInstanceRunRequested();
  const setInstanceWarningModalOpen = useSetInstanceWarningModalOpen();
  const { runInstance } = useInstanceSegmentation();

  const [showInstanceWarning, setShowInstanceWarning] = useState(false);

  const promptedRouting = useMemo(
    () =>
      policyReady
        ? resolveRoutingBinding(
            policy,
            'prompted-segmentation',
            activeLabelId,
            availablePromptedModels
          )
        : null,
    [policy, policyReady, activeLabelId, availablePromptedModels]
  );
  const suggestionRouting = useMemo(
    () =>
      policyReady
        ? resolveRoutingBinding(policy, 'instance-suggestion', null, availableSuggestionModels)
        : null,
    [policy, policyReady, availableSuggestionModels]
  );
  const instanceRouting = useMemo(
    () =>
      policyReady
        ? resolveRoutingBinding(policy, 'instance-segmentation', null, availableInstanceModels)
        : null,
    [policy, policyReady, availableInstanceModels]
  );
  const instanceBindingInvalid =
    policyReady && Boolean(instanceRouting?.binding) && !isUsableBinding(instanceRouting);
  const instanceInputs = getMatchingBindingInputs(
    instanceRouting,
    'instance-segmentation',
    instanceModel
  );
  const canRunInstance =
    policyResolved &&
    !instanceBindingInvalid &&
    !isRunningInstance &&
    Boolean(
      instanceModel &&
        availableInstanceModels.some((model) =>
          matchesModelKey(model, 'instance-segmentation', instanceModel)
        )
    );

  useEffect(() => {
    if (instanceRunRequested) {
      if (canRunInstance) setShowInstanceWarning(true);
      setInstanceRunRequested(false);
    }
  }, [instanceRunRequested, canRunInstance, setInstanceRunRequested]);

  // Mirrored to the store so keyboard shortcuts don't steal Enter while the
  // modal has focus.
  useEffect(() => {
    setInstanceWarningModalOpen(showInstanceWarning);
  }, [showInstanceWarning, setInstanceWarningModalOpen]);

  // Fetch model lists on mount if empty
  useEffect(() => {
    if (availablePromptedModels.length === 0) fetchPromptedModels();
    if (availableSuggestionModels.length === 0) fetchSuggestionModels();
    if (availableInstanceModels.length === 0) fetchInstanceModels();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Clear model selections immediately on dataset changes. The shared page-level
  // policy loader then resolves either configured routes or the normal fallbacks.
  useEffect(() => {
    setPromptedModel(null);
    setSuggestionModel(null);
    setInstanceModel(null);
  }, [datasetId, setPromptedModel, setSuggestionModel, setInstanceModel]);

  // Prompted Segmentation model resolution (active label override -> task default -> favorite/first compatible)
  useEffect(() => {
    if (!policyResolved || availablePromptedModels.length === 0) return;
    if (promptedRouting?.binding) {
      if (isUsableBinding(promptedRouting)) {
        const modelId = getModelKey(promptedRouting.model);
        if (modelId) setPromptedModel(modelId);
      } else {
        setPromptedModel(null);
      }
      return;
    }

    const modelId = getFallbackModelId(
      availablePromptedModels,
      'prompted-segmentation',
      favorites?.['prompted-segmentation'],
      activeLabelId
    );
    if (modelId) setPromptedModel(modelId);
  }, [
    policyResolved,
    promptedRouting,
    activeLabelId,
    availablePromptedModels,
    favorites,
    setPromptedModel,
  ]);

  // Instance Suggestion model resolution (task default -> favorite/first; label overrides resolve dynamically on exemplar selection)
  useEffect(() => {
    if (!policyResolved || availableSuggestionModels.length === 0) return;
    if (suggestionRouting?.binding) {
      if (isUsableBinding(suggestionRouting)) {
        const modelId = getModelKey(suggestionRouting.model);
        if (modelId) setSuggestionModel(modelId);
      } else {
        setSuggestionModel(null);
      }
      return;
    }

    const modelId = getFallbackModelId(
      availableSuggestionModels,
      'instance-suggestion',
      favorites?.['instance-suggestion']
    );
    if (modelId) setSuggestionModel(modelId);
  }, [
    policyResolved,
    suggestionRouting,
    availableSuggestionModels,
    favorites,
    setSuggestionModel,
  ]);

  // Whole-image Instance Segmentation model resolution (task default -> favorite/first)
  useEffect(() => {
    if (!policyResolved || availableInstanceModels.length === 0) return;
    if (instanceRouting?.binding) {
      if (isUsableBinding(instanceRouting)) {
        const modelId = getModelKey(instanceRouting.model);
        if (modelId) setInstanceModel(modelId);
      } else {
        setInstanceModel(null);
      }
      return;
    }

    const modelId = getFallbackModelId(
      availableInstanceModels,
      'instance-segmentation',
      favorites?.['instance-segmentation']
    );
    if (modelId) setInstanceModel(modelId);
  }, [
    policyResolved,
    instanceRouting,
    availableInstanceModels,
    favorites,
    setInstanceModel,
  ]);

  useModelSwitchPreloader(
    promptedModel,
    annotationSession.selectPromptedModel.bind(annotationSession),
    'prompted'
  );
  useModelSwitchPreloader(
    suggestionModel,
    annotationSession.selectSuggestionModel.bind(annotationSession),
    'suggestion'
  );
  useModelSwitchPreloader(
    instanceModel,
    annotationSession.selectInstanceModel.bind(annotationSession),
    'instance'
  );

  const closeInstanceWarning = () => {
    setShowInstanceWarning(false);
    setInstanceWarningModalOpen(false);
  };

  const confirmInstanceRun = (writeMode = 'patch') => {
    if (!canRunInstance) return;
    setShowInstanceWarning(false);
    setInstanceRunRequested(false);
    setInstanceWarningModalOpen(false);
    if (instanceInputs == null) {
      runInstance(writeMode);
    } else {
      runInstance(writeMode, instanceInputs);
    }
  };

  const services = [
    {
      key: 'prompted',
      task: 'prompted-segmentation',
      name: 'Prompted Segmentation',
      models: availablePromptedModels,
      isLoading: isLoadingPrompted,
      selectedModel: promptedModel,
      setSelectedModel: setPromptedModel,
      isRunning: false,
    },
    {
      key: 'instance',
      task: 'instance-segmentation',
      name: 'Instance Segmentation',
      models: availableInstanceModels,
      isLoading: isLoadingInstance,
      selectedModel: instanceModel,
      setSelectedModel: setInstanceModel,
      isRunning: isRunningInstance,
      onRun: canRunInstance ? () => setShowInstanceWarning(true) : undefined,
    },
    {
      key: 'suggestion',
      task: 'instance-suggestion',
      name: 'Within-Image Suggestion',
      models: availableSuggestionModels,
      isLoading: isLoadingSuggestion,
      selectedModel: suggestionModel,
      setSelectedModel: setSuggestionModel,
      isRunning: isRunningSuggestion,
      usageHint:
        'Shift-click objects on canvas to select exemplars, then right-click any exemplar and choose “Suggest Similar Instances”.',
    },
  ];

  return {
    services,
    policy,
    policyLoading,
    policyError,
    showInstanceWarning,
    closeInstanceWarning,
    confirmInstanceRun,
  };
}

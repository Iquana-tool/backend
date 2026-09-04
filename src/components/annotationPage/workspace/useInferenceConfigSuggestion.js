import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
    getInferenceModelCatalog,
    suggestModelRoutingStep,
} from "../../../api/inference";
import { getContourHierarchy } from "../../../api/contours";
import { useDataset } from "../../../contexts/DatasetContext";
import { useAnnotationRoutingPolicy } from "../../../contexts/AnnotationRoutingPolicyContext";
import { useToast } from "../../../contexts/ToastContext";
import { usePermissions } from "../../../hooks/usePermissions";
import { Permission } from "../../../utils/permissions";
import { resolveRoutingBinding } from "../../../utils/inferenceRouting";
import {
    useCurrentImageId,
    useCurrentMaskId,
    useDatasetLabelsMap,
    useSetObjectsFromHierarchy,
} from "../../../stores/selectors/annotationSelectors";

/**
 * Hook for executing single-image per-label AI suggestions based on the dataset's
 * active model routing policy.
 */
export function useInferenceConfigSuggestion() {
    const { currentDataset } = useDataset();
    const datasetId = currentDataset?.id;
    const {
        policy: config,
        policyReady,
        policyLoading,
        policyError,
    } = useAnnotationRoutingPolicy(datasetId);
    const imageId = useCurrentImageId();
    const currentMaskId = useCurrentMaskId();
    const datasetLabelsMap = useDatasetLabelsMap();
    const setObjectsFromHierarchy = useSetObjectsFromHierarchy();
    const { addToast } = useToast();
    const { can } = usePermissions(datasetId);

    const imageIdRef = useRef(imageId);
    imageIdRef.current = imageId;
    const currentMaskIdRef = useRef(currentMaskId);
    currentMaskIdRef.current = currentMaskId;
    const suggestionRequestTokenRef = useRef(0);

    const [catalogModels, setCatalogModels] = useState([]);
    const [isLoadingCatalog, setIsLoadingCatalog] = useState(false);
    const [catalogError, setCatalogError] = useState(null);
    const [activeRunningLabelId, setActiveRunningLabelId] = useState(null);
    const isLoadingConfig = policyLoading || isLoadingCatalog;
    const configError = policyError || catalogError;

    const canSuggest = Boolean(
        can(Permission.AI_INTERACTIVE) && can(Permission.ANNOTATION_CREATE)
    );

    // The policy is shared across the annotation workspace; this hook only
    // owns the catalog needed to validate cross-image routes.
    useEffect(() => {
        // Immediately invalidate in-flight suggestions and reset dataset-scoped catalog state.
        suggestionRequestTokenRef.current += 1;
        setCatalogModels([]);
        setCatalogError(null);
        setActiveRunningLabelId(null);

        if (!datasetId) {
            setIsLoadingCatalog(false);
            return undefined;
        }

        let isCancelled = false;
        setIsLoadingCatalog(true);

        getInferenceModelCatalog(datasetId)
            .then((catRes) => {
                if (!isCancelled) {
                    setCatalogModels(catRes?.models || []);
                    setCatalogError(null);
                }
            })
            .catch((err) => {
                if (!isCancelled) {
                    setCatalogModels([]);
                    setCatalogError(err?.message || "Failed to load the model catalog.");
                }
            })
            .finally(() => {
                if (!isCancelled) {
                    setIsLoadingCatalog(false);
                }
            });

        return () => {
            isCancelled = true;
            suggestionRequestTokenRef.current += 1;
        };
    }, [datasetId]);

    // Invalidate any in-flight suggestion requests on unmount
    useEffect(() => {
        return () => {
            suggestionRequestTokenRef.current += 1;
        };
    }, []);

    const configuredBindingsByLabelId = useMemo(() => {
        const map = new Map();
        if (!policyReady || !config) return map;

        if (Array.isArray(config.bindings)) {
            const labelIdsSet = new Set();
            if (datasetLabelsMap) {
                const mapKeys =
                    datasetLabelsMap instanceof Map
                        ? Array.from(datasetLabelsMap.keys())
                        : Object.keys(datasetLabelsMap);
                mapKeys.forEach((k) => labelIdsSet.add(Number(k)));
            }
            config.bindings.forEach((b) => {
                if (b.label_id != null) labelIdsSet.add(Number(b.label_id));
            });

            labelIdsSet.forEach((lid) => {
                const resolved = resolveRoutingBinding(
                    config,
                    "cross-image-suggestion",
                    Number(lid),
                    catalogModels
                );
                if (resolved && resolved.binding && resolved.isCompatible && !resolved.isStale) {
                    map.set(String(lid), {
                        ...resolved.binding,
                        model: resolved.model || null,
                    });
                }
            });
            return map;
        }

        return map;
    }, [config, policyReady, datasetLabelsMap, catalogModels]);

    const isConfigured = useCallback(
        (labelId) => {
            if (!canSuggest || labelId == null) return false;
            return configuredBindingsByLabelId.has(String(labelId));
        },
        [canSuggest, configuredBindingsByLabelId]
    );

    const getResolvedBinding = useCallback(
        (labelId) => {
            if (labelId == null) return null;
            const step = configuredBindingsByLabelId.get(String(labelId));
            if (!step) return null;
            return {
                binding: step,
                model: step.model || null,
            };
        },
        [configuredBindingsByLabelId]
    );

    const isRunning = useCallback(
        (labelId) => {
            return String(activeRunningLabelId) === String(labelId);
        },
        [activeRunningLabelId]
    );

    const isAnyRunning = activeRunningLabelId !== null;

    const suggestLabel = useCallback(
        async (labelId) => {
            if (!datasetId || !imageId || !currentMaskId || labelId == null) return;
            if (isAnyRunning) return;

            const step = configuredBindingsByLabelId.get(String(labelId));
            if (!step) return;

            const targetImageId = imageId;
            const targetMaskId = currentMaskId;
            const requestToken = ++suggestionRequestTokenRef.current;

            setActiveRunningLabelId(labelId);

            let result;
            try {
                result = await suggestModelRoutingStep({
                    datasetId,
                    imageId: targetImageId,
                    maskId: targetMaskId,
                    labelId: Number(labelId),
                    task: "cross-image-suggestion",
                });
            } catch (err) {
                // Only toast mutation errors if this request is still the active token and user is on target image and mask
                if (
                    requestToken === suggestionRequestTokenRef.current &&
                    imageIdRef.current === targetImageId &&
                    currentMaskIdRef.current === targetMaskId
                ) {
                    addToast({
                        type: "error",
                        message: `Suggestion failed: ${err.message || "Unknown error"}`,
                    });
                }
                return;
            } finally {
                // Only clear the active running state if this request is still the active token
                if (suggestionRequestTokenRef.current === requestToken) {
                    setActiveRunningLabelId(null);
                }
            }

            // Mutation succeeded! Guard against stale image/mask changes
            if (
                requestToken !== suggestionRequestTokenRef.current ||
                imageIdRef.current !== targetImageId ||
                currentMaskIdRef.current !== targetMaskId
            ) {
                return;
            }

            const created = result.contours_created ?? 0;
            const suppressed = result.contours_suppressed ?? 0;

            if (created > 0) {
                const suppressionText =
                    suppressed > 0
                        ? ` (${suppressed} duplicate${suppressed === 1 ? "" : "s"} suppressed)`
                        : "";
                addToast({
                    type: "success",
                    message: `Suggested ${created} object${created === 1 ? "" : "s"}${suppressionText}.`,
                });
            } else {
                addToast({
                    type: "info",
                    message: "Model finished with 0 candidate objects on this image.",
                });
            }

            // Reload contour hierarchy to update canvas objects (separate try/catch so refresh errors don't report mutation failure)
            try {
                const rawHierarchy = await getContourHierarchy(targetMaskId);
                const hierarchy = rawHierarchy?.contours || rawHierarchy;

                if (
                    requestToken !== suggestionRequestTokenRef.current ||
                    imageIdRef.current !== targetImageId ||
                    currentMaskIdRef.current !== targetMaskId
                ) {
                    return;
                }

                setObjectsFromHierarchy(hierarchy, datasetLabelsMap);
            } catch (refreshErr) {
                console.warn("Could not refresh canvas hierarchy after suggestion:", refreshErr);
                if (
                    requestToken === suggestionRequestTokenRef.current &&
                    imageIdRef.current === targetImageId &&
                    currentMaskIdRef.current === targetMaskId
                ) {
                    addToast({
                        type: "warning",
                        message: "Suggested annotations were saved, but canvas refresh failed. Reload the image to view them.",
                    });
                }
            }
        },
        [
            datasetId,
            imageId,
            currentMaskId,
            isAnyRunning,
            configuredBindingsByLabelId,
            setObjectsFromHierarchy,
            datasetLabelsMap,
            addToast,
        ]
    );

    return {
        config,
        catalogModels,
        isLoadingConfig,
        configError,
        isConfigured,
        getResolvedBinding,
        isRunning,
        isAnyRunning,
        suggestLabel,
    };
}

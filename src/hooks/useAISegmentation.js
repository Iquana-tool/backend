import { useState, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import annotationSession from '../services/annotationSession';
import { pixelToNormalized } from '../utils/coordinateUtils';
import { useDataset } from '../contexts/DatasetContext';
import {
  useAIPrompts,
  usePromptedModel,
  useCurrentImage,
  useConsumePrompts,
  useSetIsSubmittingAI,
  useImageObject,
  useAddObject,
  useUpdateObject,
  useObjectsList,
  useRefinementModeActive,
  useRefinementModeObjectId,
  useExitRefinementMode,
  useSetCurrentTool,
  useSetPromptedModel,
  useSyncEditModeDraftFromRefinement,
  useAvailablePromptedModels,
  useActiveLabelId,
} from '../stores/selectors/annotationSelectors';
import { useAnnotationRoutingPolicy } from '../contexts/AnnotationRoutingPolicyContext';
import { matchesModelKey, resolveRoutingBinding } from '../utils/inferenceRouting';

const PROMPTED_SEGMENTATION_TASK = 'prompted-segmentation';
const ROUTING_POLICY_LOADING_ERROR = 'Routing policy is still loading. Please try again in a moment.';
const isRoutingInputs = (value) =>
  value != null &&
  typeof value === 'object' &&
  !Array.isArray(value) &&
  (Object.prototype.hasOwnProperty.call(value, 'conditioning') ||
    Object.prototype.hasOwnProperty.call(value, 'parameters'));

/**
 * Normalize contour data from the backend.
 * Pydantic V2 serializes a Contour model as an array of [key, value] pairs when the
 * ServerMessage.data field is typed Union[dict, list, None] and the model fails dict
 * coercion. Convert it back to a plain object when detected.
 */
const normalizeContourData = (data) => {
  if (!Array.isArray(data)) return data;
  const looksLikeEntries = data.length > 0 && Array.isArray(data[0]) && data[0].length === 2 && typeof data[0][0] === 'string';
  return looksLikeEntries ? Object.fromEntries(data) : data;
};

/**
 * Custom hook to handle AI segmentation via WebSocket
 * Converts prompts to API format and processes the response
 */
const useAISegmentation = () => {
  const [error, setError] = useState(null);
  const { datasetId: routeDatasetId } = useParams();
  const { currentDataset } = useDataset();

  // Store state
  const prompts = useAIPrompts();
  const promptedModelId = usePromptedModel(); // This is a string ID, not an object
  const setPromptedModel = useSetPromptedModel();
  const currentImage = useCurrentImage();
  const imageObject = useImageObject();
  const objectsList = useObjectsList();
  const availablePromptedModels = useAvailablePromptedModels();
  const activeLabelId = useActiveLabelId();
  const datasetId =
    currentImage?.dataset_id ??
    (routeDatasetId ? parseInt(routeDatasetId, 10) : null) ??
    currentDataset?.id ??
    null;
  const { policy, policyLoading } = useAnnotationRoutingPolicy(datasetId);

  // Store actions
  // Prompts that produced an object are spent, not discarded — see the slice.
  const consumePrompts = useConsumePrompts();
  const setIsSubmitting = useSetIsSubmittingAI();
  const addObject = useAddObject();
  const updateObject = useUpdateObject();
  const refinementModeActive = useRefinementModeActive();
  const refinementModeObjectId = useRefinementModeObjectId();
  const exitRefinementMode = useExitRefinementMode();
  const setCurrentTool = useSetCurrentTool();
  const syncEditModeDraftFromRefinement = useSyncEditModeDraftFromRefinement();

  /**
   * Transform API response to mask format expected by SegmentationOverlay
   * Handles object_added, object_modified, and success message formats with contour data.
   * Contour is valid if it has path OR (x and y arrays with points); overlay can build path from x,y.
   */
  const transformResponseToMask = useCallback((response) => {
    let contour = null;

    // Normalize data: Pydantic V2 may serialize a Contour as an array of [key, value] pairs
    const rawData = response?.data;
    const normalizedData = normalizeContourData(rawData);

    // Ignore hierarchy payload (backend sometimes sends full hierarchy in object_added)
    if (normalizedData && Array.isArray(normalizedData.root_contours)) {
      return null;
    }

    // Handle: object_added, object_modified, or success message with contour data
    if (response && (response.type === 'object_added' || response.type === 'object_modified' || response.type === 'success') && normalizedData) {
      contour = normalizedData;
    }
    // Handle direct contour data
    else if (response && (response.path || (response.x && response.y))) {
      contour = response;
    }

    if (!contour) {
      return null;
    }
    const path = contour.path ?? contour.svg_path ?? contour.path_d;
    const hasPath = !!path;
    const x = contour.x ?? contour.X;
    const y = contour.y ?? contour.Y;
    const hasCoords = Array.isArray(x) && Array.isArray(y) && (x.length > 0 || y.length > 0);
    if (!hasPath && !hasCoords) {
      return null;
    }

    const mask = {
      id: contour.id ?? contour.contour_id ?? Date.now(),
      path: path || null,
      pixelCount: contour.quantification?.area ?? contour.pixel_count ?? contour.quantification?.pixel_count ?? 0,
      label: contour.label || 'AI Generated',
      confidence: contour.confidence,
      x: x || [],
      y: y || [],
    };

    return mask;
  }, []);

  /**
   * Run AI segmentation via WebSocket
   */
  const runSegmentation = useCallback(async (explicitInputs = null) => {
    const requestedInputs = isRoutingInputs(explicitInputs) ? explicitInputs : null;
    const hasExplicitInputs = requestedInputs !== null;
    const policyLoadingForDataset = datasetId != null && policyLoading;

    // Note: promptedModelId is just a string ID, we don't need to set model_status here
    // The status is handled by the backend
    if (!currentImage || !promptedModelId || prompts.length === 0) {
      setError('Missing required data: image, model, or prompts');
      return { success: false, error: 'Missing required data' };
    }

    // A model selected from the dataset policy can arrive before this hook's
    // policy request. Do not run it without the saved contract inputs during
    // that short window. Explicit callers already own their inputs and are
    // allowed to proceed.
    if (!hasExplicitInputs && policyLoadingForDataset) {
      setError(ROUTING_POLICY_LOADING_ERROR);
      return { success: false, error: ROUTING_POLICY_LOADING_ERROR };
    }

    if (!imageObject) {
      setError('Image not loaded');
      return { success: false, error: 'Image not loaded' };
    }

    setIsSubmitting(true);
    setError(null);

    try {
      // Verify session is ready
      if (!annotationSession.isReady()) {
        throw new Error('WebSocket session is not ready. Please wait for the connection to be established or try refreshing the page.');
      }

      // Check if prompted_segmentation service is available
      if (!annotationSession.isServiceAvailable('prompted_segmentation')) {
        throw new Error('AI segmentation service is not available. Please check your connection and try again.');
      }

      // Convert prompts to WebSocket format (convert pixel coordinates to normalized)
      const wsPrompts = {
        point_prompts: [],
        box_prompt: null,
        polygon_prompt: null,
      };

      prompts.forEach((prompt) => {
        if (prompt.type === 'polygon') {
          // Polygon (and freehand) prompts: convert each vertex to normalized
          // [x, y] pairs. The backend expects at least 3 vertices; if multiple
          // polygons were drawn we keep the last one (same as box_prompt).
          const vertices = (prompt.coords.points || [])
            .map((pt) => {
              const n = pixelToNormalized(pt.x, pt.y, imageObject.width, imageObject.height);
              return [n.x, n.y];
            });
          if (vertices.length >= 3) {
            wsPrompts.polygon_prompt = { vertices };
          }
        } else if (prompt.type === 'point') {
          // Convert pixel coordinates to normalized
          const normalized = pixelToNormalized(
            prompt.coords.x,
            prompt.coords.y,
            imageObject.width,
            imageObject.height
          );

          wsPrompts.point_prompts.push({
            x: normalized.x,
            y: normalized.y,
            label: prompt.label === 'positive', // Convert to boolean
          });
        } else if (prompt.type === 'box') {
          // Convert pixel coordinates to normalized
          const minNormalized = pixelToNormalized(
            prompt.coords.x1,
            prompt.coords.y1,
            imageObject.width,
            imageObject.height
          );
          const maxNormalized = pixelToNormalized(
            prompt.coords.x2,
            prompt.coords.y2,
            imageObject.width,
            imageObject.height
          );

          // Backend expects a SINGLE box_prompt object
          // If we have multiple boxes, only use the last one
          wsPrompts.box_prompt = {
            min_x: minNormalized.x,
            min_y: minNormalized.y,
            max_x: maxNormalized.x,
            max_y: maxNormalized.y,
          };
        }
      });

      // promptedModelId is already the string identifier we need
      const modelIdentifier = promptedModelId;

      let resolvedInputs = requestedInputs;
      if (!hasExplicitInputs && policy) {
        const routing = resolveRoutingBinding(
          policy,
          PROMPTED_SEGMENTATION_TASK,
          activeLabelId,
          availablePromptedModels
        );

        const selectedModelMatchesBinding =
          routing?.model &&
          routing?.isCompatible &&
          !routing?.isStale &&
          matchesModelKey(routing.model, PROMPTED_SEGMENTATION_TASK, modelIdentifier);

        if (selectedModelMatchesBinding && routing.binding?.inputs != null) {
          resolvedInputs = routing.binding.inputs;
        }
      }

      // Send segmentation request via WebSocket
      const response = await annotationSession.runSegmentation(modelIdentifier, wsPrompts, resolvedInputs);

      // Transform response to mask format
      const mask = transformResponseToMask(response);

      if (mask) {
        // Extract contour_id from the mask (it comes from backend as mask.id)
        // Ensure consistent ID format (convert to number if it's a string number)
        const contourId = mask.id;
        const normalizedId = typeof contourId === 'string' && !isNaN(contourId)
          ? Number(contourId)
          : (typeof contourId === 'number' ? contourId : contourId);

        // Handle message types generically: object_modified updates existing, object_added/success creates new
        // Note: 'success' type is a fallback for backward compatibility
        const isModified = response && response.type === 'object_modified';

        if (isModified) {
          // When in refinement mode, we know which object to update from refinementModeObjectId
          const objectToUpdate = refinementModeActive && refinementModeObjectId
            ? objectsList.find(obj => obj.id === refinementModeObjectId)
            : objectsList.find(obj => {
                const objContourId = obj.contour_id || obj.id;
                const normalizedObjContourId = typeof objContourId === 'string' && !isNaN(objContourId)
                  ? Number(objContourId)
                  : (typeof objContourId === 'number' ? objContourId : objContourId);
                return normalizedObjContourId === normalizedId;
              });

          if (objectToUpdate) {
            // Update the existing object - completely replace mask and path to show only the refined version
            const newX = mask.x || [];
            const newY = mask.y || [];
            updateObject(objectToUpdate.id, {
              mask: mask,
              contour_id: normalizedId, // Update with the NEW contour_id from backend
              pixelCount: mask.pixelCount || 0,
              // Preserve existing label if it exists, otherwise use the one from backend
              label: mask.label || objectToUpdate.label || `Object #${objectToUpdate.id}`,
              // Include x and y coordinate arrays if available from backend response
              x: newX,
              y: newY,
              // Ensure path is available for rendering - explicitly set from new refined object
              path: mask.path || null,
            });
            // Sync edit mode draft so the blue control-point overlay shows the new refined contour
            if (refinementModeActive && newX.length > 0 && newY.length > 0) {
              syncEditModeDraftFromRefinement(newX, newY);
            }
          }
        }
        // For refinement (object_modified): stay in refinement mode so user can refine again or exit via "Exit Refinement"
        consumePrompts();
        return { success: true, mask };
      }
      // Any successful object_added: canvas is updated by useWebSocketObjectHandler; do not throw
      if (response && response.success !== false && response.type === 'object_added') {
        consumePrompts();
        if (refinementModeActive) {
          try {
            await annotationSession.unselectRefinementObject();
            exitRefinementMode();
          } catch (error) {
            // Continue anyway
          }
        }
        return { success: true, mask: null };
      }

      // Other success response we couldn't parse (e.g. hierarchy); treat as success
      if (response && response.success !== false) {
        consumePrompts();
        return { success: true, mask: null };
      }

      throw new Error('No valid mask returned from server');
    } catch (err) {
      const errorMessage = err.message || 'Segmentation failed';
      setError(errorMessage);
      // Note: Model status is handled by the backend, no need to update here
      return { success: false, error: errorMessage };
    } finally {
      setIsSubmitting(false);
    }
  }, [
    currentImage,
    promptedModelId,
    prompts,
    imageObject,
    refinementModeActive,
    refinementModeObjectId,
    objectsList,
    transformResponseToMask,
    addObject,
    updateObject,
    consumePrompts,
    exitRefinementMode,
    setCurrentTool,
    syncEditModeDraftFromRefinement,
    policy,
    policyLoading,
    activeLabelId,
    availablePromptedModels,
    datasetId,
  ]);

  return {
    runSegmentation,
    error,
    isReady: currentImage && promptedModelId && prompts.length > 0,
  };
};

export default useAISegmentation;

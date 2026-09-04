/**
 * WebSocket Message Type Constants
 * 
 * Defines all message types used in the WebSocket communication
 * between the frontend and backend annotation system.
 * 
 */

// ==================== CLIENT MESSAGE TYPES ====================

/**
 * Client sends these message types to the server
 */
export const CLIENT_MESSAGE_TYPES = {
  // Point the (per-user) session at a different image without reconnecting
  SWITCH_IMAGE: 'switch_image',

  // Image Focus
  FOCUS_IMAGE: 'focus_image',
  UNFOCUS_IMAGE: 'unfocus_image',
  
  // Refinement Mode
  SELECT_REFINEMENT_OBJECT: 'refine_object',
  UNSELECT_REFINEMENT_OBJECT: 'unselect_refinement_object',
  
  // AI Segmentation
  PROMPTED_SELECT_MODEL: 'prompted_select_model',
  PROMPTED_SEGMENTATION: 'prompted_inference',
  
  // Suggestion Segmentation
  SUGGESTION_SELECT_MODEL: 'suggestion_select_model',

  // Instance Segmentation
  INSTANCE_SELECT_MODEL: 'instance_select_model',

  // Object Management
  OBJECT_ADD_MANUAL: 'object_add_manual',
  OBJECT_FINALISE: 'object_finalise',
  OBJECT_DELETE: 'object_delete',
  OBJECT_MODIFY: 'object_modify',
  
  // Session Management
  SUGGESTION_ENABLE: 'suggestion_enable',
  SUGGESTION_INFERENCE: 'suggestion_inference',
  INSTANCE_INFERENCE: 'instance_inference',
  FINISH_ANNOTATION: 'finish_annotation',
};

/**
 * How interactive instance-segmentation predictions are written.
 */
export const INSTANCE_WRITE_MODES = {
  PATCH: 'patch',
  OVERRIDE: 'override',
};

// ==================== SERVER MESSAGE TYPES ====================

/**
 * Server sends these message types to the client
 */
export const SERVER_MESSAGE_TYPES = {
  // Session
  SESSION_INITIALIZED: 'session_initialized',
  // The session now points at another image; carries that image's mask id and status.
  // The contours follow separately as an OBJECTS message.
  IMAGE_SWITCHED: 'image_switched',

  // Full object hierarchy
  OBJECTS: 'objects',

  // Object Operations
  OBJECT_ADDED: 'object_added',
  OBJECT_MODIFIED: 'object_modified',
  OBJECT_REMOVED: 'object_removed',
  OBJECT_FINALISED: 'object_finalised',
  
  // AI Segmentation
  SEGMENTATION_RESULT: 'segmentation_result',
  MODEL_SELECTED: 'model_selected',
  
  // Status Messages
  SUCCESS: 'success',
  WARNING: 'warning',
  
  // Errors
  ERROR: 'error',
};

// ==================== MESSAGE BUILDERS ====================

/**
 * Generates a unique message ID
 * @returns {string} Unique message identifier
 */
export const generateMessageId = () => {
  return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Creates a base message structure
 * @param {string} type - Message type from CLIENT_MESSAGE_TYPES
 * @param {Object} data - Message payload
 * @returns {Object} Formatted message object
 */
const createMessage = (type, data) => ({
  id: generateMessageId(),
  type,
  data,
  success: true,
  message: "",
});

/**
 * Message builders for each client message type
 * These ensure consistent message formatting
 */
export const MessageBuilders = {
  /**
   * Re-target the session at another image over the existing connection
   * @param {number|string} imageId - ID of the image to annotate next
   */
  switchImage: (imageId) => createMessage(
    CLIENT_MESSAGE_TYPES.SWITCH_IMAGE,
    { image_id: typeof imageId === 'string' ? Number(imageId) : imageId }
  ),

  /**
   * Focus on a specific contour in the image
   * @param {number} contourId - ID of the contour to focus on
   */
  focusImage: (contourId) => createMessage(
    CLIENT_MESSAGE_TYPES.FOCUS_IMAGE,
    { focussed_contour_id: contourId }
  ),

  /**
   * Remove focus from image
   */
  unfocusImage: () => createMessage(
    CLIENT_MESSAGE_TYPES.UNFOCUS_IMAGE,
    {}
  ),

  /**
   * Select object for refinement mode
   * @param {number} contourId - ID of the contour to refine
   */
  selectRefinementObject: (contourId) => createMessage(
    CLIENT_MESSAGE_TYPES.SELECT_REFINEMENT_OBJECT,
    { contour_id: contourId }
  ),

  /**
   * Exit refinement mode
   */
  unselectRefinementObject: () => createMessage(
    CLIENT_MESSAGE_TYPES.UNSELECT_REFINEMENT_OBJECT,
    {}
  ),

  /**
   * Select AI model for prompted segmentation
   * @param {string} modelName - Segmentation Model identifier
   */
  selectPromptedModel: (modelName) => createMessage(
    CLIENT_MESSAGE_TYPES.PROMPTED_SELECT_MODEL,
    { selected_model: modelName }
  ),

  /**
   * Select model for suggestion segmentation
   * @param {string} modelIdentifier - Suggestion model identifier
   */
  selectSuggestionModel: (modelIdentifier) => createMessage(
    CLIENT_MESSAGE_TYPES.SUGGESTION_SELECT_MODEL,
    { model_identifier: modelIdentifier }
  ),

  /**
   * Select model for instance segmentation
   * @param {string} modelName - Instance model identifier
   */
  selectInstanceModel: (modelName) => createMessage(
    CLIENT_MESSAGE_TYPES.INSTANCE_SELECT_MODEL,
    { selected_model: modelName }
  ),

  /**
   * Request AI segmentation with prompts
   * @param {string} modelIdentifier - Model to use for segmentation
   * @param {Object} prompts - Prompts object containing points and box
   * @param {Array} prompts.point_prompts - Array of point prompts {x: float, y: float, label: boolean}
   * @param {Object} prompts.box_prompt - Single box prompt {min_x, min_y, max_x, max_y} or null
   * @param {Object} prompts.polygon_prompt - Single polygon prompt {vertices: [[x, y], ...]} or null
   * @param {Object} prompts.circle_prompt - Single circle prompt {center_x, center_y, radius} or null
   * @param {Object|null} inputs - Optional routing inputs (parameters / conditioning)
   */
  runSegmentation: (modelIdentifier, prompts, inputs = null) => createMessage(
    CLIENT_MESSAGE_TYPES.PROMPTED_SEGMENTATION,
    {
      model_identifier: modelIdentifier,
      model_key: modelIdentifier,  // Failsafe
      prompts: {
        point_prompts: prompts.point_prompts || [],
        box_prompt: prompts.box_prompt || null,
        polygon_prompt: prompts.polygon_prompt || null,
        circle_prompt: prompts.circle_prompt || null,
      },
      ...(inputs && typeof inputs === 'object' ? { inputs } : {}),
    }
  ),

  /**
   * Add a manually drawn object
   * @param {Array<number>} x - Array of x coordinates (normalized 0-1)
   * @param {Array<number>} y - Array of y coordinates (normalized 0-1)
   * @param {string|null} label - Object label
   * @param {number|null} parentId - Parent contour ID
   * @param {number} confidence - Confidence score (0-1)
   * @param {number|null} labelId - Label to create the object with. The backend
   *   validates a Contour, whose label field is `label_id`; a freshly drawn object
   *   has none, but a half produced by splitting inherits the original's.
   */
  addObject: (x, y, label = null, parentId = null, confidence = 1.0, labelId = null) => createMessage(
    CLIENT_MESSAGE_TYPES.OBJECT_ADD_MANUAL,
    {
      x,
      y,
      label,
      label_id: labelId,
      parent_id: parentId,
      confidence,
    }
  ),

  /**
   * Finalize a temporary object
   * @param {number} contourId - ID of the contour to finalize
   */
  finalizeObject: (contourId) => createMessage(
    CLIENT_MESSAGE_TYPES.OBJECT_FINALISE,
    { contour_id: contourId }
  ),

  /**
   * Delete an object
   * @param {number} contourId - ID of the contour to delete
   */
  deleteObject: (contourId) => createMessage(
    CLIENT_MESSAGE_TYPES.OBJECT_DELETE,
    { contour_id: contourId }
  ),

  /**
   * Modify an object's properties
   * @param {number} contourId - ID of the contour to modify
   * @param {Object} fieldsToUpdate - Fields to update (e.g., {confidence: 0.95, label: 'new_label'})
   */
  modifyObject: (contourId, fieldsToUpdate) => createMessage(
    CLIENT_MESSAGE_TYPES.OBJECT_MODIFY,
    {
      contour_id: contourId,
      fields_to_be_updated: fieldsToUpdate,
    }
  ),

  /**
   * Enable suggestion mode
   */
  enableSuggestion: () => createMessage(
    CLIENT_MESSAGE_TYPES.SUGGESTION_ENABLE,
    {}
  ),

  /**
   * Request suggestion inference to find similar instances
   * @param {Array<number>} seedContourIds - Array of contour IDs to use as seeds
   * @param {string} modelKey - Suggestion model key (e.g., 'DINOv3')
   * @param {number|null} labelId - Optional label ID to assign to found instances
   */
  runSuggestion: (seedContourIds, modelKey, labelId = null, inputs = null) => createMessage(
    CLIENT_MESSAGE_TYPES.SUGGESTION_INFERENCE,
    {
      seed_contour_ids: seedContourIds,
      model_key: modelKey,
      label_id: labelId,
      ...(inputs && typeof inputs === "object" ? { inputs } : {}),
    }
  ),

  /**
   * Request instance segmentation inference
   * @param {string} modelKey - Instance model key
   * @param {'patch'|'override'} writeMode - How predictions are applied to existing contours
   * @param {Object|null} inputs - Optional routing inputs (parameters / conditioning)
   */
  runInstance: (modelKey, writeMode = INSTANCE_WRITE_MODES.PATCH, inputs = null) => createMessage(
    CLIENT_MESSAGE_TYPES.INSTANCE_INFERENCE,
    {
      model_registry_key: modelKey,
      write_mode: writeMode,
      ...(inputs && typeof inputs === 'object' ? { inputs } : {}),
    }
  ),

  /**
   * Finish annotation session
   */
  finishAnnotation: () => createMessage(
    CLIENT_MESSAGE_TYPES.FINISH_ANNOTATION,
    {}
  ),
};

// ==================== MESSAGE VALIDATORS ====================

/**
 * Validates if a message has the required base structure
 * @param {Object} message - Message to validate
 * @returns {boolean} True if valid
 */
export const isValidMessage = (message) => {
  const isValid = (
    message &&
    typeof message === 'object' &&
    typeof message.id === 'string' &&
    typeof message.type === 'string' &&
    typeof message.success === 'boolean' &&
    message.hasOwnProperty('data') &&
    // Allow optional message field
    (message.message === undefined || typeof message.message === 'string')
  );
  
  if (!isValid) {
    console.log('[MessageValidation] Invalid message:', {
      message,
      checks: {
        exists: !!message,
        isObject: typeof message === 'object',
        hasId: typeof message?.id === 'string',
        hasType: typeof message?.type === 'string',
        hasSuccess: typeof message?.success === 'boolean',
        hasData: message?.hasOwnProperty('data'),
        messageFieldValid: message?.message === undefined || typeof message?.message === 'string'
      }
    });
  }
  
  return isValid;
};

/**
 * Checks if a message is an error message
 * @param {Object} message - Message to check
 * @returns {boolean} True if error message
 */
export const isErrorMessage = (message) => {
  return message?.type === SERVER_MESSAGE_TYPES.ERROR || message?.success === false;
};

/**
 * Extracts error information from a message
 * @param {Object} message - Error message
 * @returns {Object} Error details {message, data}
 */
export const extractError = (message) => ({
  message: message?.message || 'Unknown error occurred',
  data: message?.data || null,
  id: message?.id || null,
});


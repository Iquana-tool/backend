/**
 * Annotation Session Service
 * 
 * API for WebSocket-based annotation operations.
 * Provides type-safe wrappers around WebSocket messages and manages
 * the annotation session lifecycle.
 */

import websocketService from './websocket';
import { getAuthToken } from '../api/util';
import { MessageBuilders, SERVER_MESSAGE_TYPES } from '../utils/messageTypes';

/**
 * Session states
 */
export const SessionState = {
  UNINITIALIZED: 'uninitialized',
  INITIALIZING: 'initializing',
  READY: 'ready',
  ERROR: 'error',
};

/**
 * Gets the authenticated username from the stored auth user.
 *
 * This is display information only. The backend derives the session's identity
 * from the bearer token and ignores the username in the URL, so there is no
 * anonymous fallback any more — a session without a token is refused.
 *
 * @returns {string|null} The authenticated username, or null when logged out
 */
const getUserId = () => {
  try {
    const userStr = localStorage.getItem('auth_user');
    if (userStr) {
      const user = JSON.parse(userStr);
      if (user && user.username) {
        return user.username;
      }
    }
  } catch (error) {
    console.warn('[AnnotationSession] Failed to read auth user from localStorage:', error);
  }
  return null;
};

const getWsBaseUrl = () => {
  const wsEnv = import.meta.env.VITE_WS_URL;
  if (wsEnv && wsEnv.trim()) {
    return wsEnv.trim().replace(/\/$/, '');
  }

  const apiEnv = import.meta.env.VITE_API_BASE_URL;
  if (apiEnv && apiEnv.trim()) {
    const apiBase = apiEnv.trim().replace(/\/$/, '');
    if (apiBase.startsWith('https://')) {
      return apiBase.replace(/^https:\/\//, 'wss://');
    }
    if (apiBase.startsWith('http://')) {
      return apiBase.replace(/^http:\/\//, 'ws://');
    }
    if (apiBase.startsWith('/') && typeof window !== 'undefined' && window.location.host) {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${protocol}//${window.location.host}${apiBase}`;
    }
  }

  return 'ws://localhost:8000';
};

/**
 * Annotation Session Manager
 */
class AnnotationSession {
  constructor() {
    this.sessionState = SessionState.UNINITIALIZED;
    this.currentImageId = null;
    this.currentUserId = null;
    this.runningServices = [];
    this.failedServices = [];
    this.sessionListeners = new Set();

    // Prefer explicit WS URL. Fallback derives WS from API base URL.
    this.wsBaseUrl = getWsBaseUrl();
  }

  /**
   * Initialize annotation session for an image
   *
   * The socket is opened per *user*, with the first image in the URL. Later images
   * are reached with `switchImage`, which reuses this same connection.
   *
   * @param {number|string} imageId - Image ID
   * @param {string} userId - Username for display (optional; identity comes from the token)
   * @returns {Promise<Object>} Session initialization data
   */
  async initialize(imageId, userId = null) {
    try {
      this.currentImageId = imageId;
      this.currentUserId = userId || getUserId();
      this._updateSessionState(SessionState.INITIALIZING);

      // The backend authenticates the socket and checks `annotation.create` on the
      // image's dataset before accepting it. Browsers cannot set headers on a
      // WebSocket handshake, so the token travels as a query parameter.
      const token = getAuthToken();
      if (!token) {
        this._updateSessionState(SessionState.ERROR);
        throw new Error('You must be signed in to open an annotation session.');
      }

      const wsUrl = this._buildUrl(this.currentImageId, token);

      // Connect to WebSocket
      await websocketService.connect(wsUrl, {
        reconnectAttempts: 5,
        reconnectDelay: 1000,
      });

      // Wait for session_initialized message
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          unsubscribe();
          this._updateSessionState(SessionState.ERROR);
          reject(new Error('Session initialization timeout'));
        }, 20000);

        const unsubscribe = websocketService.on(
          SERVER_MESSAGE_TYPES.SESSION_INITIALIZED,
          (message) => {
            clearTimeout(timeout);
            unsubscribe();

            // Always process the session data, regardless of success status
            this.runningServices = message.data?.running || [];
            this.failedServices = message.data?.failed || [];

            // Session is ready if we have at least one running service
            // This allows the session to work even if some services failed
            if (this.runningServices.length > 0) {
              this._updateSessionState(SessionState.READY);
            } else {
              // Only set to ERROR if no services are running at all
              this._updateSessionState(SessionState.ERROR);
            }

            // Always resolve with the session data, even if some services failed
            resolve({
              running: this.runningServices,
              failed: this.failedServices,
              objects: message.data?.objects || null,
              maskId: message.data?.mask_id ?? null,
              // The image's combined status, plus its Calibrate/Annotate/Review
              // breakdown — both arrive with SESSION_INITIALIZED so the workspace
              // needs no extra REST call to draw the status pill.
              maskStatus: message.data?.mask_status ?? null,
              phaseStatus: message.data?.phase_status ?? null,
            });
          }
        );
      });
    } catch (error) {
      console.error('[AnnotationSession] Initialization failed:', error);
      this._updateSessionState(SessionState.ERROR);
      throw error;
    }
  }

  /**
   * Close the current session
   * @param {boolean} sendFinishMessage - Whether to send finish_annotation message (default: false)
   * @returns {Promise<void>}
   */
  async close(sendFinishMessage = false) {
    try {
      if (sendFinishMessage && websocketService.isConnected()) {
        const message = MessageBuilders.finishAnnotation();
        await websocketService.send(message);
      }

      websocketService.disconnect();
      this.currentImageId = null;
      this._updateSessionState(SessionState.UNINITIALIZED);
    } catch (error) {
      console.error('[AnnotationSession] Error closing session:', error);
      // Disconnect anyway
      websocketService.disconnect();
      this.currentImageId = null;
      this._updateSessionState(SessionState.UNINITIALIZED);
    }
  }

  /**
   * Point the session at a different image.
   *
   * Sends a `switch_image` message over the existing socket rather than closing and
   * reopening it. A reconnect meant re-authenticating, re-running three backend health
   * checks and re-selecting every model before the new image's contours could even be
   * asked for — all of which the server already has loaded and can keep.
   *
   * If the socket is not up (first image, or it dropped), this falls back to opening one.
   *
   * @param {number|string} newImageId - New image ID
   * @returns {Promise<Object>} Session data for the new image
   */
  async switchImage(newImageId) {
    if (this.currentImageId === newImageId && this.isReady()) {
      return {
        running: this.runningServices,
        failed: this.failedServices,
      };
    }

    if (!websocketService.isConnected()) {
      // No live connection to switch on — open one pointed at the new image.
      await this.close(false);
      return this.initialize(newImageId, this.currentUserId);
    }

    this._updateSessionState(SessionState.INITIALIZING);

    try {
      const response = await websocketService.send(
        MessageBuilders.switchImage(newImageId),
        true
      );

      this.currentImageId = newImageId;
      // Keep the reconnect target in step, so a dropped connection comes back on the
      // image the user is looking at rather than the one the socket was opened with.
      const token = getAuthToken();
      if (token) {
        websocketService.setUrl(this._buildUrl(newImageId, token));
      }
      // The backend re-reports its services with every switch, so a backend that came
      // up (or went down) since connecting is reflected without a reconnect.
      this.runningServices = response?.data?.running || this.runningServices;
      this.failedServices = response?.data?.failed || this.failedServices;
      this._updateSessionState(
        this.runningServices.length > 0 ? SessionState.READY : SessionState.ERROR
      );

      return {
        running: this.runningServices,
        failed: this.failedServices,
        // Contours are not in this reply — they arrive as a separate OBJECTS message,
        // so the canvas can show the image immediately and the object list can fill in
        // behind its own spinner.
        objects: null,
        maskId: response?.data?.mask_id ?? null,
        maskStatus: response?.data?.mask_status ?? null,
        phaseStatus: response?.data?.phase_status ?? null,
      };
    } catch (error) {
      console.error('[AnnotationSession] switch_image failed:', error);
      this._updateSessionState(SessionState.ERROR);
      throw error;
    }
  }

  /**
   * Ask the server to re-send the current image's contours.
   *
   * Backs the "Retry" on the contour spinner. Re-switching to the image we are already
   * on is the cheapest way to do it — the server treats it as any other switch and
   * follows the reply with a fresh OBJECTS message.
   *
   * @returns {Promise<Object>} Session data for the (unchanged) image
   */
  async reloadObjects() {
    if (this.currentImageId == null) {
      throw new Error('No image to reload.');
    }
    const imageId = this.currentImageId;
    // Clear it first so switchImage does not short-circuit on "already on this image".
    this.currentImageId = null;
    try {
      return await this.switchImage(imageId);
    } catch (error) {
      this.currentImageId = imageId;
      throw error;
    }
  }

  // ==================== AI SEGMENTATION OPERATIONS ====================

  /**
   * Select AI model for prompted segmentation
   * @param {string} modelName - Segmentation Model identifier
   * @returns {Promise<Object>} Response message
   */
  async selectPromptedModel(modelName) {
    this._ensureReady();
    if (!modelName) {
      return Promise.resolve({ success: true, message: 'No model to select' });
    }

    // Only send message if prompted segmentation service is available
    if (!this.isServiceAvailable('prompted_segmentation')) {
      return Promise.resolve({ success: false, message: 'Service not available' });
    }

    const message = MessageBuilders.selectPromptedModel(modelName);
    return websocketService.send(message, true);
  }

  /**
   * Select model for suggestion segmentation
   * @param {string} modelIdentifier - Suggestion model identifier
   * @returns {Promise<Object>} Response message
   */
  async selectSuggestionModel(modelIdentifier) {
    this._ensureReady();
    if (!modelIdentifier) {
      return Promise.resolve({ success: true, message: 'No model to select' });
    }

    // Only send message if suggestion service is available
    if (!this.isServiceAvailable('suggestion_segmentation')) {
      return Promise.resolve({ success: false, message: 'Service not available' });
    }

    const message = MessageBuilders.selectSuggestionModel(modelIdentifier);
    return websocketService.send(message, true);
  }

  /**
   * Select model for instance segmentation
   * @param {string} modelName - Instance model identifier
   * @returns {Promise<Object>} Response message
   */
  async selectInstanceModel(modelName) {
    this._ensureReady();
    if (!modelName) {
      return Promise.resolve({ success: true, message: 'No model to select' });
    }

    // Only send message if instance service is available
    if (!this.isServiceAvailable('instance_segmentation')) {
      return Promise.resolve({ success: false, message: 'Service not available' });
    }

    const message = MessageBuilders.selectInstanceModel(modelName);
    return websocketService.send(message, true);
  }

  /**
   * Preload models into backend memory after session initialization
   * This sends select_model messages to preload the currently selected models
   * @param {Object} selectedModels - Object with promptedModel, suggestionModel, instanceModel
   * @returns {Promise<void>}
   */
  async preloadModels(selectedModels = {}) {
    if (!this.isReady()) {
      return;
    }

    const { promptedModel, suggestionModel, instanceModel } = selectedModels;

    // Extract model IDs (handle both string IDs and model objects)
    const promptedModelId = typeof promptedModel === 'string' ? promptedModel : promptedModel?.id;
    const suggestionModelId = typeof suggestionModel === 'string' ? suggestionModel : suggestionModel?.id;
    const instanceModelId = typeof instanceModel === 'string' ? instanceModel : instanceModel?.id;

    // Send model selection messages to preload models into memory
    // These calls won't throw errors if services aren't available
    const promises = [];

    if (promptedModelId && this.isServiceAvailable('prompted_segmentation')) {
      promises.push(
        this.selectPromptedModel(promptedModelId).catch(() => {
          // Error handled silently
        })
      );
    }

    if (suggestionModelId && this.isServiceAvailable('suggestion_segmentation')) {
      promises.push(
        this.selectSuggestionModel(suggestionModelId).catch(() => {
          // Error handled silently
        })
      );
    }

    if (instanceModelId && this.isServiceAvailable('instance_segmentation')) {
      promises.push(
        this.selectInstanceModel(instanceModelId).catch(() => {
          // Error handled silently
        })
      );
    }

    // Wait for all preload operations to complete
    await Promise.allSettled(promises);
  }

  /**
   * @deprecated Use selectPromptedModel instead
   * Select AI model for segmentation (legacy method)
   * @param {string} modelName - Segmentation Model identifier
   * @returns {Promise<Object>} Response message
   */
  async selectModel(modelName) {
    return this.selectPromptedModel(modelName);
  }

  /**
   * Run AI segmentation with prompts
   * @param {string} modelIdentifier - Model to use
   * @param {Object} prompts - Prompts object {points, boxes, masks}
   * @param {Object|null} inputs - Optional routing inputs (parameters / conditioning)
   * @returns {Promise<Object>} Segmentation result
   */
  async runSegmentation(modelIdentifier, prompts, inputs = null) {
    this._ensureReady();
    const message = MessageBuilders.runSegmentation(modelIdentifier, prompts, inputs);
    return websocketService.send(message, true);
  }

  // ==================== OBJECT OPERATIONS ====================

  /**
   * Add a manually drawn object
   * @param {Array<number>} x - X coordinates (normalized)
   * @param {Array<number>} y - Y coordinates (normalized)
   * @param {string|null} label - Object label
   * @param {number|null} parentId - Parent contour ID
   * @param {number} confidence - Confidence score
   * @param {number|null} labelId - Label the new object is created with (optional)
   * @returns {Promise<Object>} Response with added objects
   */
  async addObject(x, y, label = null, parentId = null, confidence = 1.0, labelId = null) {
    this._ensureReady();
    const message = MessageBuilders.addObject(x, y, label, parentId, confidence, labelId);
    return websocketService.send(message, true);
  }

  /**
   * Finalize a temporary object
   * @param {number} contourId - Contour ID to finalize
   * @returns {Promise<Object>} Response message
   */
  async finalizeObject(contourId) {
    this._ensureReady();
    const message = MessageBuilders.finalizeObject(contourId);
    return websocketService.send(message, true);
  }

  /**
   * Delete an object
   * @param {number} contourId - Contour ID to delete
   * @returns {Promise<void>}
   */
  async deleteObject(contourId) {
    this._ensureReady();
    const message = MessageBuilders.deleteObject(contourId);
    // Fire-and-forget: the backend broadcasts OBJECT_REMOVED which is handled
    // by the useWebSocketObjectHandler listener. Waiting for a correlated
    // response here would always time out because OBJECT_REMOVED carries a
    // different message ID than the one we sent.
    return websocketService.send(message, false);
  }

  /**
   * Modify object properties
   * @param {number} contourId - Contour ID to modify
   * @param {Object} updates - Fields to update
   * @returns {Promise<Object>} Response message
   */
  async modifyObject(contourId, updates) {
    this._ensureReady();
    const message = MessageBuilders.modifyObject(contourId, updates);
    return websocketService.send(message, true);
  }

  // ==================== IMAGE FOCUS OPERATIONS ====================

  /**
   * Focus on a specific contour
   * @param {number} contourId - Contour ID to focus on
   * @returns {Promise<void>}
   */
  async focusImage(contourId) {
    this._ensureReady();
    const message = MessageBuilders.focusImage(contourId);
    return websocketService.send(message);
  }

  /**
   * Remove focus from image
   * @returns {Promise<void>}
   */
  async unfocusImage() {
    this._ensureReady();
    const message = MessageBuilders.unfocusImage();
    return websocketService.send(message);
  }

  // ==================== REFINEMENT MODE ====================

  /**
   * Select an object for refinement
   * @param {number} contourId - Contour ID to refine
   * @returns {Promise<void>}
   */
  async selectRefinementObject(contourId) {
    this._ensureReady();
    const message = MessageBuilders.selectRefinementObject(contourId);
    return websocketService.send(message, true);
  }

  /**
   * Exit refinement mode
   * @returns {Promise<void>}
   */
  async unselectRefinementObject() {
    this._ensureReady();
    const message = MessageBuilders.unselectRefinementObject();
    return websocketService.send(message, true);
  }

  // ==================== SUGGESTION SEGMENTATION ====================

  /**
   * Run suggestion segmentation to find similar instances
   * @param {Array<number>} seedContourIds - Array of contour IDs to use as seeds
   * @param {number|null} labelId - Optional label ID to assign to found instances
   * @returns {Promise<Object>} Response with added objects (objects are added via OBJECT_ADDED WebSocket messages)
   */
  async runSuggestion(seedContourIds, modelKey, labelId = null, inputs = null) {
    this._ensureReady();

    // Check if suggestion service is available
    if (!this.isServiceAvailable('suggestion_segmentation')) {
      throw new Error('Suggestion segmentation service is not available. Please check your connection.');
    }

    const message = MessageBuilders.runSuggestion(seedContourIds, modelKey, labelId, inputs);
    return websocketService.send(message, true);
  }

  // ==================== INSTANCE SEGMENTATION ====================

  /**
   * Run instance segmentation inference
   * @param {string} modelKey - Instance model key
   * @param {'patch'|'override'} writeMode - How predictions are applied to existing contours
   * @param {Object|null} inputs - Optional routing inputs (parameters / conditioning)
   * @returns {Promise<Object>} Response with added objects (objects are added via OBJECT_ADDED WebSocket messages)
   */
  async runInstance(modelKey, writeMode = 'patch', inputs = null) {
    this._ensureReady();

    // Check if instance service is available
    if (!this.isServiceAvailable('instance_segmentation')) {
      throw new Error('Instance segmentation service is not available. Please check your connection.');
    }

    const message = MessageBuilders.runInstance(modelKey, writeMode, inputs);
    return websocketService.send(message, true);
  }

  // ==================== SESSION MANAGEMENT ====================

  /**
   * Enable suggestion mode
   * @returns {Promise<void>}
   */
  async enableSuggestion() {
    this._ensureReady();
    const message = MessageBuilders.enableSuggestion();
    return websocketService.send(message);
  }

  // ==================== EVENT SUBSCRIPTIONS ====================

  /**
   * Subscribe to session state changes
   * @param {Function} callback - Callback (state) => void
   * @returns {Function} Unsubscribe function
   */
  onSessionStateChange(callback) {
    this.sessionListeners.add(callback);
    return () => this.sessionListeners.delete(callback);
  }

  /**
   * Subscribe to WebSocket messages
   * @param {string} messageType - Message type to listen for
   * @param {Function} callback - Callback (message) => void
   * @returns {Function} Unsubscribe function
   */
  onMessage(messageType, callback) {
    return websocketService.on(messageType, callback);
  }

  /**
   * Subscribe to all WebSocket messages
   * @param {Function} callback - Callback (message) => void
   * @returns {Function} Unsubscribe function
   */
  onAnyMessage(callback) {
    return websocketService.onAny(callback);
  }

  /**
   * Subscribe to connection state changes
   * @param {Function} callback - Callback (state) => void
   * @returns {Function} Unsubscribe function
   */
  onConnectionStateChange(callback) {
    return websocketService.onConnectionStateChange(callback);
  }

  // ==================== STATE GETTERS ====================

  /**
   * Get current session state
   * @returns {string} Session state
   */
  getSessionState() {
    return this.sessionState;
  }

  /**
   * Get current image ID
   * @returns {number|string|null} Image ID
   */
  getCurrentImageId() {
    return this.currentImageId;
  }

  /**
   * Get running services
   * @returns {Array<string>} Running services
   */
  getRunningServices() {
    return [...this.runningServices];
  }

  /**
   * Get failed services
   * @returns {Array<string>} Failed services
   */
  getFailedServices() {
    return [...this.failedServices];
  }

  /**
   * Check if session is ready
   * @returns {boolean} True if ready
   */
  isReady() {
    return this.sessionState === SessionState.READY && websocketService.isConnected();
  }

  /**
   * Check if a specific service is available
   * @param {string} serviceName - Service name (e.g., 'prompted_segmentation')
   * @returns {boolean} True if service is running
   */
  isServiceAvailable(serviceName) {
    return this.runningServices.includes(serviceName);
  }

  // ==================== PRIVATE METHODS ====================

  /**
   * Build the session URL for an image.
   *
   * The image rides in the path even though the socket is per-user: it is what an
   * automatic reconnect replays, and the server pre-selects it so the reconnected
   * session comes back with the right contours and needs no follow-up switch.
   *
   * @private
   * @param {number|string|null} imageId - Image to open on, or null for no image
   * @param {string} token - Bearer token (browsers cannot set handshake headers)
   * @returns {string} The WebSocket URL
   */
  _buildUrl(imageId, token) {
    const imageSegment = imageId != null ? `/${imageId}` : '';
    return `${this.wsBaseUrl}/annotation_session/ws/${this.currentUserId}` +
      `${imageSegment}?token=${encodeURIComponent(token)}`;
  }

  /**
   * Ensure session is ready
   * @private
   * @throws {Error} If session is not ready
   */
  _ensureReady() {
    if (!this.isReady()) {
      throw new Error('Session not ready. Initialize session first.');
    }
  }

  /**
   * Update session state and notify listeners
   * @private
   * @param {string} newState - New session state
   */
  _updateSessionState(newState) {
    if (this.sessionState !== newState) {
      this.sessionState = newState;

      this.sessionListeners.forEach(callback => {
        try {
          callback(newState);
        } catch (error) {
          console.error('[AnnotationSession] Listener error:', error);
        }
      });
    }
  }
}

// Singleton instance
const annotationSession = new AnnotationSession();

export default annotationSession;

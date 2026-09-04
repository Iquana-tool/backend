import { TASK_ORDER, TASKS, getTaskMeta } from "../../../constants/tasks";

/**
 * Visual category mapping for the Model Orchestration UI.
 * Aggregates 4 canonical backend task keys into 3 user-facing visual categories.
 * Backend routing policy and save payloads always retain the 4 canonical task keys.
 */
export const ORCHESTRATION_CATEGORIES = [
  {
    key: "interactive",
    label: "Interactive segmentation",
    description: "Point, box, and within-image interactive inference tools.",
    tasks: ["prompted-segmentation", "instance-suggestion"],
  },
  {
    key: "instance",
    label: "Instance segmentation",
    description: "Autonomous batch segmentation across full images.",
    tasks: ["instance-segmentation"],
  },
  {
    key: "cross-image",
    label: "Cross-image suggestion",
    description: "Exemplar and embedding retrieval across dataset images.",
    tasks: ["cross-image-suggestion"],
  },
];

/**
 * Canonical route status descriptors.
 */
export const ROUTE_STATUS = {
  EXPLICIT: "explicit",
  INHERITED: "inherited",
  UNBOUND: "unbound",
  UNBOUND_INCOMPATIBLE: "unbound-incompatible-default",
  STALE: "stale",
  STALE_DEFAULT: "stale-default",
};

/**
 * Normalizes task and labelId into a canonical selector key.
 * Format: `${task}::${labelId == null ? "default" : Number(labelId)}`
 *
 * @param {string} task
 * @param {string|number|null|undefined} labelId
 * @returns {string}
 */
export function normalizeSelector(task, labelId) {
  const normLabel = labelId == null || labelId === "" ? "default" : Number(labelId);
  return `${task}::${normLabel}`;
}

/**
 * Parses a canonical selector key back into task and numeric labelId (or null).
 *
 * @param {string} key
 * @returns {{ task: string, labelId: number | null }}
 */
export function parseSelectorKey(key) {
  if (!key || typeof key !== "string") {
    return { task: "", labelId: null };
  }
  const [task, labelPart] = key.split("::");
  const labelId = labelPart === "default" || labelPart == null ? null : Number(labelPart);
  return { task: task || "", labelId };
}

/**
 * Normalizes a single binding object.
 *
 * @param {Object} binding
 * @returns {Object|null}
 */
export function normalizeBinding(binding) {
  if (!binding || !binding.task) return null;
  const labelId = binding.label_id == null || binding.label_id === "" ? null : Number(binding.label_id);
  return {
    task: binding.task,
    label_id: labelId,
    model_registry_key: binding.model_registry_key || null,
    inputs: {
      conditioning: binding.inputs?.conditioning ? { ...binding.inputs.conditioning } : {},
      parameters: binding.inputs?.parameters ? { ...binding.inputs.parameters } : {},
    },
  };
}

/**
 * Converts a list of bindings into a Map keyed by canonical selector.
 *
 * @param {Array<Object>} bindings
 * @returns {Map<string, Object>}
 */
export function normalizeBindingsMap(bindings) {
  const map = new Map();
  if (!Array.isArray(bindings)) return map;
  for (const b of bindings) {
    if (!b || !b.task) continue;
    const norm = normalizeBinding(b);
    if (!norm) continue;
    const key = normalizeSelector(norm.task, norm.label_id);
    map.set(key, norm);
  }
  return map;
}

/**
 * Finds duplicate selector occurrences in an array of bindings.
 *
 * @param {Array<Object>} bindings
 * @returns {Array<string>} list of duplicate selector keys
 */
export function findDuplicateSelectors(bindings) {
  if (!Array.isArray(bindings)) return [];
  const seen = new Set();
  const duplicates = [];
  for (const b of bindings) {
    if (!b || !b.task) continue;
    const key = normalizeSelector(b.task, b.label_id);
    if (seen.has(key)) {
      if (!duplicates.includes(key)) {
        duplicates.push(key);
      }
    } else {
      seen.add(key);
    }
  }
  return duplicates;
}

/**
 * Deep equality helper for objects and primitives.
 */
export function deepEqual(a, b) {
  if (a === b) return true;
  if (a == null || b == null) return a === b;
  if (typeof a !== "object" || typeof b !== "object") return a === b;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(a)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (!deepEqual(a[i], b[i])) return false;
    }
    return true;
  }
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  if (keysA.length !== keysB.length) return false;
  for (const key of keysA) {
    if (!Object.prototype.hasOwnProperty.call(b, key)) return false;
    if (!deepEqual(a[key], b[key])) return false;
  }
  return true;
}

/**
 * Checks if two binding inputs objects are structurally equal.
 *
 * @param {Object} inputsA
 * @param {Object} inputsB
 * @returns {boolean}
 */
export function areInputsEqual(inputsA, inputsB) {
  const condA = inputsA?.conditioning || {};
  const condB = inputsB?.conditioning || {};
  const paramsA = inputsA?.parameters || {};
  const paramsB = inputsB?.parameters || {};
  return deepEqual(condA, condB) && deepEqual(paramsA, paramsB);
}

/**
 * Calculates route coverage across visual categories and overall.
 *
 * Slot count per task = 1 (default) + numberOfLabels.
 * Overall possible = slotCount * 4 (tasks).
 * Interactive possible = slotCount * 2.
 * Instance possible = slotCount * 1.
 * Cross-image possible = slotCount * 1.
 *
 * @param {Array<Object>|Map<string, Object>} draftBindings
 * @param {Object|number} labelsInput - labelsById object or numeric count of labels
 * @param {Object|Array} catalogInput - catalog object with `.models` or array of models
 * @returns {Object} coverage breakdown
 */
export function calculateCoverage(draftBindings, labelsInput = {}, catalogInput = {}) {
  const isObjectLabels = typeof labelsInput === "object" && labelsInput !== null && !Array.isArray(labelsInput);
  const validLabelIds = isObjectLabels ? new Set(Object.keys(labelsInput).map(Number)) : null;
  const numLabels = typeof labelsInput === "number" ? labelsInput : Object.keys(labelsInput || {}).length;
  const slotCount = 1 + Math.max(0, numLabels);

  const models = Array.isArray(catalogInput) ? catalogInput : catalogInput?.models || [];
  const bindingsList = draftBindings instanceof Map ? Array.from(draftBindings.values()) : Array.isArray(draftBindings) ? draftBindings : [];

  const coverageByCategory = {};
  let overallBound = 0;
  let overallStale = 0;
  const overallPossible = slotCount * TASK_ORDER.length;

  for (const category of ORCHESTRATION_CATEGORIES) {
    const categoryTasks = category.tasks;
    const categoryPossible = slotCount * categoryTasks.length;
    let categoryBound = 0;
    let categoryStale = 0;

    // Track unique selectors per category
    const boundSelectors = new Set();

    for (const b of bindingsList) {
      if (!b || !categoryTasks.includes(b.task) || !b.model_registry_key) continue;

      // Ignore orphaned bindings for deleted labels not present in the active label set
      if (b.label_id != null && validLabelIds && !validLabelIds.has(Number(b.label_id))) {
        continue;
      }

      const key = normalizeSelector(b.task, b.label_id);
      if (boundSelectors.has(key)) continue;
      boundSelectors.add(key);
      categoryBound += 1;

      // Check if model exists in catalog for this task and is compatible if label_id is present
      const model = models.find(
        (m) => m.task === b.task && m.registry_key === b.model_registry_key
      );
      const isStale =
        !model ||
        (b.label_id != null &&
          model.label_ids &&
          model.label_ids.length > 0 &&
          !model.label_ids.includes(Number(b.label_id)));

      if (isStale) {
        categoryStale += 1;
      }
    }

    coverageByCategory[category.key] = {
      bound: categoryBound,
      possible: categoryPossible,
      stale: categoryStale,
      percentage: categoryPossible > 0 ? Math.round((categoryBound / categoryPossible) * 100) : 0,
    };

    overallBound += categoryBound;
    overallStale += categoryStale;
  }

  return {
    overall: {
      bound: overallBound,
      possible: overallPossible,
      stale: overallStale,
      percentage: overallPossible > 0 ? Math.round((overallBound / overallPossible) * 100) : 0,
    },
    categories: coverageByCategory,
  };
}

/**
 * Resolves the effective route status for a specific (task, labelId).
 *
 * Rules:
 * 1. Explicit override on (task, labelId) -> explicit (or stale if model not in catalog or incompatible with label).
 * 2. If labelId == null: task default (or stale default, or unbound).
 * 3. If labelId != null and no override:
 *    - If no task default: unbound.
 *    - If task default model missing in catalog: stale-default.
 *    - If default model is class-agnostic (!label_ids || label_ids.length === 0): inherited.
 *    - If default model is class-specific and includes labelId: inherited.
 *    - If default model is class-specific and does not include labelId: unbound-incompatible-default.
 *
 * @param {Object} params
 * @param {string} params.task
 * @param {number|null} [params.labelId=null]
 * @param {Array<Object>|Map<string, Object>} params.bindings
 * @param {Object|Array} params.catalog
 * @returns {Object} effective route resolution
 */
export function resolveEffectiveRoute({ task, labelId = null, bindings, catalog = {} }) {
  const models = Array.isArray(catalog) ? catalog : catalog?.models || [];
  const bindingsMap = bindings instanceof Map ? bindings : normalizeBindingsMap(bindings);

  const numericLabelId = labelId == null || labelId === "" ? null : Number(labelId);
  const targetKey = normalizeSelector(task, numericLabelId);
  const defaultKey = normalizeSelector(task, null);

  const explicitBinding = bindingsMap.get(targetKey);
  const defaultBinding = bindingsMap.get(defaultKey);

  // 1. Explicit binding requested
  if (numericLabelId != null && explicitBinding && explicitBinding.model_registry_key) {
    const model = models.find(
      (m) => m.task === task && m.registry_key === explicitBinding.model_registry_key
    ) || null;

    if (!model) {
      return {
        status: ROUTE_STATUS.STALE,
        binding: explicitBinding,
        model: null,
        isExplicit: true,
        inheritedFrom: null,
        reason: "stale_model",
      };
    }

    const isClassAgnostic = !model.label_ids || model.label_ids.length === 0;
    const isCompatible = isClassAgnostic || model.label_ids.includes(numericLabelId);

    if (!isCompatible) {
      return {
        status: ROUTE_STATUS.STALE,
        binding: explicitBinding,
        model,
        isExplicit: true,
        inheritedFrom: null,
        reason: "incompatible_model",
      };
    }

    return {
      status: ROUTE_STATUS.EXPLICIT,
      binding: explicitBinding,
      model,
      isExplicit: true,
      inheritedFrom: null,
      reason: null,
    };
  }

  // 2. Task default itself requested
  if (numericLabelId == null) {
    if (defaultBinding && defaultBinding.model_registry_key) {
      const model = models.find(
        (m) => m.task === task && m.registry_key === defaultBinding.model_registry_key
      ) || null;
      return {
        status: model ? ROUTE_STATUS.EXPLICIT : ROUTE_STATUS.STALE,
        binding: defaultBinding,
        model,
        isExplicit: true,
        inheritedFrom: null,
        reason: model ? null : "stale_default_model",
      };
    }
    return {
      status: ROUTE_STATUS.UNBOUND,
      binding: null,
      model: null,
      isExplicit: false,
      inheritedFrom: null,
      reason: "no_default",
    };
  }

  // 3. Label override missing -> check inheritance from task default
  if (!defaultBinding || !defaultBinding.model_registry_key) {
    return {
      status: ROUTE_STATUS.UNBOUND,
      binding: null,
      model: null,
      isExplicit: false,
      inheritedFrom: null,
      reason: "no_default",
    };
  }

  const defaultModel = models.find(
    (m) => m.task === task && m.registry_key === defaultBinding.model_registry_key
  );

  if (!defaultModel) {
    return {
      status: ROUTE_STATUS.STALE_DEFAULT,
      binding: defaultBinding,
      model: null,
      isExplicit: false,
      inheritedFrom: "task-default",
      reason: "stale_default",
    };
  }

  // Class-agnostic default: model has no label_ids or empty label_ids
  const isClassAgnostic = !defaultModel.label_ids || defaultModel.label_ids.length === 0;
  if (isClassAgnostic || defaultModel.label_ids.includes(numericLabelId)) {
    return {
      status: ROUTE_STATUS.INHERITED,
      binding: defaultBinding,
      model: defaultModel,
      isExplicit: false,
      inheritedFrom: "task-default",
      reason: null,
    };
  }

  // Class-specific default that excludes this label
  return {
    status: ROUTE_STATUS.UNBOUND_INCOMPATIBLE,
    binding: defaultBinding,
    model: defaultModel,
    isExplicit: false,
    inheritedFrom: null,
    reason: "incompatible_default",
  };
}

/**
 * Calculates a change-set diff between canonical saved bindings and the editor draft.
 *
 * Types of changes:
 * - `added`: selector exists only in draft
 * - `removed`: selector exists only in saved
 * - `model_changed`: same selector, different model_registry_key
 * - `inputs_changed`: same selector & model, different parameters/conditioning
 *
 * @param {Array<Object>} savedBindings
 * @param {Array<Object>} draftBindings
 * @returns {Array<Object>} list of change objects
 */
export function calculatePolicyDiff(savedBindings = [], draftBindings = []) {
  const savedMap = normalizeBindingsMap(savedBindings);
  const draftMap = normalizeBindingsMap(draftBindings);

  const changes = [];

  // Check all selectors in draft
  for (const [key, draftBinding] of draftMap.entries()) {
    if (!draftBinding.model_registry_key) continue;
    const { task, labelId } = parseSelectorKey(key);
    const savedBinding = savedMap.get(key);

    if (!savedBinding || !savedBinding.model_registry_key) {
      changes.push({
        type: "added",
        selectorKey: key,
        task,
        labelId,
        draftBinding,
        savedBinding: null,
      });
    } else if (savedBinding.model_registry_key !== draftBinding.model_registry_key) {
      changes.push({
        type: "model_changed",
        selectorKey: key,
        task,
        labelId,
        draftBinding,
        savedBinding,
      });
    } else if (!areInputsEqual(savedBinding.inputs, draftBinding.inputs)) {
      changes.push({
        type: "inputs_changed",
        selectorKey: key,
        task,
        labelId,
        draftBinding,
        savedBinding,
      });
    }
  }

  // Check for removed selectors that existed in saved but not in draft
  for (const [key, savedBinding] of savedMap.entries()) {
    if (!savedBinding.model_registry_key) continue;
    const { task, labelId } = parseSelectorKey(key);
    const draftBinding = draftMap.get(key);

    if (!draftBinding || !draftBinding.model_registry_key) {
      changes.push({
        type: "removed",
        selectorKey: key,
        task,
        labelId,
        draftBinding: null,
        savedBinding,
      });
    }
  }

  return changes;
}

/**
 * Generates human-readable summary copy for policy changes.
 *
 * @param {Array<Object>} changes
 * @param {Object} context
 * @param {Object} [context.labelsById={}]
 * @param {Object|Array} [context.catalog={}]
 * @returns {{ totalCount: number, items: Array<string>, summaryText: string }}
 */
export function formatChangeSummary(changes = [], { labelsById = {}, catalog = {} } = {}) {
  const models = Array.isArray(catalog) ? catalog : catalog?.models || [];
  const totalCount = changes.length;

  if (totalCount === 0) {
    return {
      totalCount: 0,
      items: [],
      summaryText: "No unsaved changes",
    };
  }

  const getTargetLabel = (task, labelId) => {
    const taskMeta = getTaskMeta(task);
    if (labelId == null) {
      return `${taskMeta.short || taskMeta.label} default`;
    }
    const label = labelsById[labelId] || labelsById[String(labelId)];
    return label?.name || `Label #${labelId}`;
  };

  const getModelName = (task, modelKey) => {
    if (!modelKey) return "None";
    const found = models.find((m) => m.task === task && m.registry_key === modelKey);
    return found?.name || modelKey;
  };

  const items = changes.map((c) => {
    const target = getTargetLabel(c.task, c.labelId);
    switch (c.type) {
      case "added": {
        const modelName = getModelName(c.task, c.draftBinding?.model_registry_key);
        return `Bound ${modelName} to ${target}`;
      }
      case "removed": {
        return `Unbound ${target}`;
      }
      case "model_changed": {
        const modelName = getModelName(c.task, c.draftBinding?.model_registry_key);
        return `Changed model for ${target} to ${modelName}`;
      }
      case "inputs_changed": {
        return `Updated parameters for ${target}`;
      }
      default:
        return `Modified route for ${target}`;
    }
  });

  let summaryText = "";
  if (items.length === 1) {
    summaryText = items[0];
  } else if (items.length === 2) {
    summaryText = `${items[0]}, ${items[1]}`;
  } else {
    summaryText = `${items[0]}, ${items[1]} (+${items.length - 2} more)`;
  }

  return {
    totalCount,
    items,
    summaryText,
  };
}

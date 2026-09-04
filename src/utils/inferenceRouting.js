/**
 * Reusable routing policy resolution utilities (Issue #31 Task-Aware Model Routing).
 *
 * Provides resolution of (task, label) bindings, inheritance cascades, class-aware
 * model compatibility checks, and deterministic batch plan construction.
 */

import { BATCH_INFERENCE_TASKS } from "../constants/tasks";

/**
 * Checks whether a catalog model is compatible with a given label ID.
 * Returns true if the model is class-agnostic or explicitly predicts labelId.
 */
export const isModelCompatibleWithLabel = (model, labelId) => {
    if (!model || labelId == null) return true;
    if (!model.label_ids || model.label_ids.length === 0) return true;
    return model.label_ids.includes(Number(labelId));
};

/**
 * Resolves the effective routing binding for a (task, label) pair against a policy.
 *
 * Checks exact label override first, then falls back to task default.
 * If a task default model is class-aware and does not predict the target label,
 * isCompatible is marked false.
 *
 * @param {object|null} policy - Stored DatasetModelRoutingRead object
 * @param {string} task - Routing task key
 * @param {number|null} labelId - Target label ID or null for task default
 * @param {Array} catalogModels - Live models in the catalog for contract and compatibility lookup
 * @returns {object|null} { binding, model, isOverride, isTaskDefault, isCompatible, isStale }
 */
export const matchesModelKey = (m, task, key) => {
    if (!m || !key) return false;
    const taskMatch = !m.task || m.task === task;
    const keyMatch =
        m.registry_key === key ||
        m.id === key ||
        m.identifier === key ||
        String(m.id) === String(key);
    return taskMatch && keyMatch;
};

export const resolveRoutingBinding = (policy, task, labelId = null, catalogModels = null) => {
    if (!policy || !Array.isArray(policy.bindings)) return null;
    const hasCatalog = catalogModels != null && Array.isArray(catalogModels);

    // 1. Check exact label override
    if (labelId != null) {
        const override = policy.bindings.find(
            (b) => b.task === task && Number(b.label_id) === Number(labelId)
        );
        if (override) {
            const model = hasCatalog
                ? catalogModels.find((m) => matchesModelKey(m, task, override.model_registry_key))
                : null;
            const isStale = Boolean(hasCatalog && !model);
            const isCompatible = isStale
                ? false
                : !hasCatalog
                ? true
                : isModelCompatibleWithLabel(model, labelId);
            return {
                binding: override,
                model: model || null,
                isOverride: true,
                isTaskDefault: false,
                isCompatible,
                isStale,
            };
        }
    }

    // 2. Check task default
    const defaultBinding = policy.bindings.find(
        (b) => b.task === task && (b.label_id == null || b.label_id === undefined)
    );
    if (defaultBinding) {
        const model = hasCatalog
            ? catalogModels.find((m) => matchesModelKey(m, task, defaultBinding.model_registry_key))
            : null;
        const isStale = Boolean(hasCatalog && !model);
        const isCompatible = isStale
            ? false
            : !hasCatalog || labelId == null
            ? true
            : isModelCompatibleWithLabel(model, labelId);
        return {
            binding: defaultBinding,
            model: model || null,
            isOverride: false,
            isTaskDefault: true,
            isCompatible,
            isStale,
        };
    }

    return null;
};

/**
 * Builds a deterministic batch step mapping from policy bindings.
 *
 * @param {object|null} policy - Stored routing policy
 * @param {object} labelsById - Map of label ID -> label object
 * @param {Array} catalogModels - Live models list
 * @param {string|null} preferredTask - Optional explicit task filter ("instance-segmentation" | "cross-image-suggestion")
 * @returns {object} Map of labelId -> step object
 */
export const getBatchStepsFromPolicy = (
    policy,
    labelsById = {},
    catalogModels = [],
    preferredTask = null
) => {
    if (!policy || !Array.isArray(policy.bindings)) return {};

    const steps = {};
    const labels = Object.values(labelsById || {});
    const tasksToCheck = preferredTask
        ? [preferredTask]
        : ["instance-segmentation", "cross-image-suggestion"];

    labels.forEach((label) => {
        for (const task of tasksToCheck) {
            if (!BATCH_INFERENCE_TASKS.includes(task)) continue;
            const resolved = resolveRoutingBinding(policy, task, label.id, catalogModels);
            if (resolved && resolved.binding && resolved.isCompatible && !resolved.isStale) {
                steps[label.id] = {
                    label_id: Number(label.id),
                    model_registry_key: resolved.binding.model_registry_key,
                    task: resolved.binding.task,
                    inputs: resolved.binding.inputs || null,
                    min_confidence: 0.0,
                };
                break; // Found highest-priority valid batch step for this label
            }
        }
    });

    return steps;
};

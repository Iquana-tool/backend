import React, { useState, useEffect, useMemo, useRef } from "react";
import {
  X,
  Search,
  Check,
  ChevronRight,
  ArrowLeft,
  Star,
  Trash2,
  AlertTriangle,
} from "lucide-react";
import { getTaskMeta } from "../../../../constants/tasks";
import {
  getEffectiveContract,
  getDefaultConditioning,
  getDefaultParameters,
} from "../../plannerContractUtils";
import DynamicHyperParameter from "../../../datasets/training/DynamicHyperParameter";
import { resolveLabelColor } from "../../../annotationPage/workspace/labelColorUtils";

const hasOwn = (value, key) =>
  Object.prototype.hasOwnProperty.call(value || {}, key);

const normalizeLabelId = (value) =>
  value == null || value === "" ? null : Number(value);

const findModelForTask = (models, task, registryKey) =>
  models.find(
    (model) => model.task === task && model.registry_key === registryKey
  ) || null;

const isModelCompatibleWithTarget = (model, task, labelId) => {
  if (!model || model.task !== task) return false;
  const numericLabelId = normalizeLabelId(labelId);
  if (numericLabelId == null) return true;
  return (
    !model.label_ids ||
    model.label_ids.length === 0 ||
    model.label_ids.includes(numericLabelId)
  );
};

const getActiveStrategies = (strategies) =>
  strategies.filter((strategy) => strategy?.key && strategy.available === true);

const usesRetrievalStrategy = (task, condSpec) =>
  condSpec?.kind === "reference_images" ||
  condSpec?.kind === "embeddings" ||
  (task === "cross-image-suggestion" && condSpec?.kind === "instances");

const getCountBounds = (condSpec) => {
  const minUnits = Number.isFinite(condSpec?.min_units)
    ? Math.trunc(condSpec.min_units)
    : 0;
  const maxUnits =
    condSpec?.max_units == null || !Number.isFinite(condSpec.max_units)
      ? null
      : Math.trunc(condSpec.max_units);

  return { minUnits, maxUnits };
};

const sanitizeCount = (value, condSpec, fallback = 5) => {
  const { minUnits, maxUnits } = getCountBounds(condSpec);
  const numericValue =
    typeof value === "number"
      ? value
      : typeof value === "string" && value.trim() !== ""
      ? Number(value)
      : Number.NaN;
  const numericFallback =
    typeof fallback === "number" && Number.isFinite(fallback)
      ? fallback
      : 5;
  const candidate = Number.isFinite(numericValue)
    ? Math.trunc(numericValue)
    : Math.trunc(numericFallback);
  const lowerBounded = Math.max(candidate, minUnits);

  // null max_units is the contract's explicit unbounded case.
  return maxUnits === null ? lowerBounded : Math.min(lowerBounded, maxUnits);
};

const getDefaultCount = (condSpec, defaultCount) =>
  sanitizeCount(defaultCount ?? 5, condSpec, 5);

const toFiniteNumber = (value) => {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const numericValue = Number(value);
    return Number.isFinite(numericValue) ? numericValue : null;
  }
  return null;
};

const coerceParameterValue = (value, type) => {
  if (type === "int" || type === "float") {
    const numericValue = toFiniteNumber(value);
    if (numericValue === null) return { valid: false, value: undefined };
    return {
      valid: true,
      value: type === "int" ? Math.trunc(numericValue) : numericValue,
    };
  }

  if (type === "bool") {
    if (typeof value === "boolean") return { valid: true, value };
    if (typeof value === "string") {
      const normalized = value.trim().toLowerCase();
      if (normalized === "true") return { valid: true, value: true };
      if (normalized === "false") return { valid: true, value: false };
    }
    return { valid: false, value: undefined };
  }

  if (type === "str") {
    return value == null
      ? { valid: false, value: undefined }
      : { valid: true, value: String(value) };
  }

  return { valid: false, value: undefined };
};

const getParameterType = (parameter) => parameter?.type || "float";

const getNumericParameterBounds = (parameter, type) => {
  const minValue = toFiniteNumber(parameter?.min_value);
  const maxValue = toFiniteNumber(parameter?.max_value);

  return {
    minValue: type === "int" && minValue !== null ? Math.ceil(minValue) : minValue,
    maxValue: type === "int" && maxValue !== null ? Math.floor(maxValue) : maxValue,
  };
};

const normalizeNumericParameter = (parameter, rawValue, defaultValue, type) => {
  const { minValue, maxValue } = getNumericParameterBounds(parameter, type);
  const rawResult = coerceParameterValue(rawValue, type);
  const defaultResult = coerceParameterValue(defaultValue, type);
  const finiteBound = minValue ?? maxValue ?? 0;
  let value = rawResult.valid
    ? rawResult.value
    : defaultResult.valid
    ? defaultResult.value
    : finiteBound;

  if (minValue !== null) value = Math.max(value, minValue);
  if (maxValue !== null) value = Math.min(value, maxValue);

  return {
    valid: Number.isFinite(value),
    value: type === "int" ? Math.trunc(value) : value,
  };
};

const valuesMatch = (left, right) => left === right;

const normalizeParameterValue = (parameter, rawValue, defaultValue) => {
  const type = getParameterType(parameter);
  const hasOptions = Array.isArray(parameter?.options);

  if (type === "int" || type === "float") {
    const normalizedNumeric = normalizeNumericParameter(
      parameter,
      rawValue,
      defaultValue,
      type
    );
    if (!hasOptions) return normalizedNumeric;
  }

  if (hasOptions) {
    const optionValues = parameter.options
      .map((option) => coerceParameterValue(option, type))
      .filter((option) => option.valid)
      .map((option) => option.value);

    if (optionValues.length === 0) {
      return { valid: false, value: undefined };
    }

    const rawResult = coerceParameterValue(rawValue, type);
    const defaultResult = coerceParameterValue(defaultValue, type);
    const matchingRaw = rawResult.valid
      ? optionValues.find((option) => valuesMatch(option, rawResult.value))
      : undefined;
    const matchingDefault = defaultResult.valid
      ? optionValues.find((option) => valuesMatch(option, defaultResult.value))
      : undefined;

    return {
      valid: true,
      value: matchingRaw ?? matchingDefault ?? optionValues[0],
    };
  }

  const rawResult = coerceParameterValue(rawValue, type);
  const defaultResult = coerceParameterValue(defaultValue, type);
  if (rawResult.valid) return rawResult;
  if (defaultResult.valid) return defaultResult;
  if (type === "bool") return { valid: true, value: false };
  if (type === "str") return { valid: true, value: "" };
  return { valid: false, value: undefined };
};

const bindingIssueMessage = (issue) => {
  if (issue === "incompatible") {
    return "Existing route is incompatible with this label. Pick a compatible model to repair it.";
  }
  return "Existing route is stale or missing. Pick a compatible current model to repair it.";
};

/**
 * Build the smallest inputs envelope accepted by the selected model contract.
 * Existing values are retained only when the contract declares the corresponding
 * input surface. Retrieval strategies are additionally checked against the
 * dataset-scoped catalog because unavailable placeholders cannot be persisted.
 */
const normalizeInputsForContract = ({
  contract,
  task,
  rawInputs,
  strategies,
  label,
}) => {
  const condSpec = contract?.conditioning;
  const defaultConditioning = getDefaultConditioning(contract, strategies, label);
  const sourceInputs =
    rawInputs && typeof rawInputs === "object" && !Array.isArray(rawInputs)
      ? rawInputs
      : {};
  const rawConditioning =
    sourceInputs.conditioning &&
    typeof sourceInputs.conditioning === "object" &&
    !Array.isArray(sourceInputs.conditioning)
      ? sourceInputs.conditioning
      : {};
  const rawParameters =
    sourceInputs.parameters &&
    typeof sourceInputs.parameters === "object" &&
    !Array.isArray(sourceInputs.parameters)
      ? sourceInputs.parameters
      : {};

  const conditioning = {};
  const kind = condSpec?.kind || "none";

  if (kind === "concept_text") {
    const defaultText =
      typeof defaultConditioning.concept_text === "string"
        ? defaultConditioning.concept_text
        : label?.name || "";
    conditioning.concept_text =
      typeof rawConditioning.concept_text === "string"
        ? rawConditioning.concept_text
        : defaultText;
  } else if (kind !== "none") {
    const { minUnits, maxUnits } = getCountBounds(condSpec);
    const hasFixedCount = maxUnits !== null && minUnits === maxUnits;
    const shouldIncludeCount =
      Boolean(condSpec?.user_selectable_count) ||
      hasFixedCount ||
      hasOwn(rawConditioning, "count");

    if (shouldIncludeCount) {
      conditioning.count = sanitizeCount(
        rawConditioning.count,
        condSpec,
        defaultConditioning.count ?? getDefaultCount(condSpec, minUnits)
      );
    }

    const retrievalRequired = usesRetrievalStrategy(task, condSpec);
    const activeStrategies = getActiveStrategies(strategies);
    const requestedStrategy = hasOwn(rawConditioning, "strategy")
      ? rawConditioning.strategy
      : defaultConditioning.strategy;
    const selectedStrategy = activeStrategies.find(
      (strategy) => strategy.key === requestedStrategy
    );
    const fallbackStrategy = activeStrategies[0];

    if (retrievalRequired) {
      const strategy = selectedStrategy || fallbackStrategy;
      if (!strategy) {
        return {
          valid: false,
          inputs: { conditioning, parameters: {} },
          message:
            "No available retrieval strategy. Select a model with an available retrieval strategy before applying this route.",
        };
      }
      conditioning.strategy = strategy.key;
    } else if (selectedStrategy) {
      // Instances may optionally use retrieval. Preserve only an active key.
      conditioning.strategy = selectedStrategy.key;
    } else if (requestedStrategy != null && fallbackStrategy) {
      // An unavailable optional strategy is repaired when an active fallback exists.
      conditioning.strategy = fallbackStrategy.key;
    }
  }

  const parameters = {};
  const declaredParameters = Array.isArray(contract?.parameters)
    ? contract.parameters
    : [];
  const defaultParameters = getDefaultParameters(contract);
  for (const parameter of declaredParameters) {
    if (!parameter?.key) {
      return {
        valid: false,
        inputs: { conditioning, parameters },
        message: "Selected model declares an invalid parameter key.",
      };
    }

    const rawValue = hasOwn(rawParameters, parameter.key)
      ? rawParameters[parameter.key]
      : undefined;
    const normalized = normalizeParameterValue(
      parameter,
      rawValue,
      defaultParameters[parameter.key]
    );
    if (!normalized.valid) {
      return {
        valid: false,
        inputs: { conditioning, parameters },
        message: `Parameter \"${parameter.key}\" cannot be normalized for this model.`,
      };
    }
    parameters[parameter.key] = normalized.value;
  }

  return {
    valid: true,
    inputs: { conditioning, parameters },
    message: null,
  };
};

/**
 * MatrixSideDrawer Component (Design B)
 *
 * 2-stage right slide-over drawer:
 * Stage 1: "BIND A MODEL" (Model selector)
 * Stage 2: "MODEL INPUTS" (Conditioning & parameter sliders)
 */
export default function MatrixSideDrawer({
  isOpen = false,
  onClose,
  target = null, // { task, labelId }
  labelsById = {},
  catalog = { models: [], retrieval_strategies: [] },
  draftBindings = [],
  onSaveRoute,
  onUnbindRoute,
  canEdit = true,
}) {
  const models = Array.isArray(catalog?.models) ? catalog.models : [];
  const strategies = Array.isArray(catalog?.retrieval_strategies)
    ? catalog.retrieval_strategies
    : [];

  const task = target?.task ?? "prompted-segmentation";
  const labelId = target?.labelId ?? null;
  const label = labelId != null ? labelsById[labelId] : null;
  const taskMeta = getTaskMeta(task);

  // Existing binding for this exact (task, labelId)
  const existingBinding = useMemo(() => {
    if (!target) return null;
    const targetLabelId = normalizeLabelId(labelId);
    return draftBindings.find(
      (b) =>
        b.task === task && normalizeLabelId(b.label_id) === targetLabelId
    );
  }, [draftBindings, target, task, labelId]);

  // Drawer View State: 'select' | 'configure'
  const [view, setView] = useState("select");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedModelKey, setSelectedModelKey] = useState(null);
  const [inputs, setInputs] = useState({ conditioning: {}, parameters: {} });

  const drawerRef = useRef(null);
  const triggerElementRef = useRef(null);

  // Compatible models for this task & label
  const compatibleModels = useMemo(() => {
    return models.filter((model) =>
      isModelCompatibleWithTarget(model, task, labelId)
    );
  }, [models, task, labelId]);

  // Filtered by search
  const filteredModels = useMemo(() => {
    if (!searchQuery.trim()) return compatibleModels;
    const q = searchQuery.toLowerCase();
    return compatibleModels.filter(
      (m) =>
        (m.name && m.name.toLowerCase().includes(q)) ||
        (m.registry_key && m.registry_key.toLowerCase().includes(q)) ||
        (m.description && m.description.toLowerCase().includes(q))
    );
  }, [compatibleModels, searchQuery]);

  const existingBindingModel = useMemo(
    () =>
      findModelForTask(
        models,
        task,
        existingBinding?.model_registry_key
      ),
    [models, task, existingBinding]
  );

  const existingBindingIssue = useMemo(() => {
    if (!existingBinding) return null;
    if (!existingBinding.model_registry_key || !existingBindingModel) {
      return "missing";
    }
    if (!isModelCompatibleWithTarget(existingBindingModel, task, labelId)) {
      return "incompatible";
    }
    return null;
  }, [existingBinding, existingBindingModel, task, labelId]);

  // Reset drawer state when opened or target changes
  useEffect(() => {
    if (!isOpen || !target) return;

    setView("select");
    setSearchQuery("");

    if (existingBinding && existingBindingModel && !existingBindingIssue) {
      setSelectedModelKey(existingBinding.model_registry_key);
      const normalized = normalizeInputsForContract({
        contract: getEffectiveContract(existingBindingModel),
        task,
        rawInputs: existingBinding.inputs,
        strategies,
        label,
      });
      setInputs(normalized.inputs);
    } else {
      setSelectedModelKey(null);
      setInputs({ conditioning: {}, parameters: {} });
    }
  }, [
    isOpen,
    target,
    existingBinding,
    existingBindingModel,
    existingBindingIssue,
    strategies,
    label,
    task,
  ]);

  // Focus trap & Escape listener
  useEffect(() => {
    if (!isOpen) return;

    triggerElementRef.current = document.activeElement;

    const focusTimer = setTimeout(() => {
      if (drawerRef.current) {
        const firstFocusable = drawerRef.current.querySelector(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        );
        if (firstFocusable && typeof firstFocusable.focus === "function") {
          firstFocusable.focus();
        }
      }
    }, 50);

    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }

      if (e.key === "Tab") {
        if (!drawerRef.current) return;
        const focusableElements = drawerRef.current.querySelectorAll(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        );
        if (focusableElements.length === 0) return;

        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
          }
        } else {
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      clearTimeout(focusTimer);
      window.removeEventListener("keydown", handleKeyDown);
      if (triggerElementRef.current && typeof triggerElementRef.current.focus === "function") {
        triggerElementRef.current.focus();
      }
    };
  }, [isOpen, onClose]);

  // Selected model details
  const selectedModel = findModelForTask(models, task, selectedModelKey);
  const selectedModelIsCompatible = isModelCompatibleWithTarget(
    selectedModel,
    task,
    labelId
  );
  const contract = selectedModel ? getEffectiveContract(selectedModel) : null;
  const condSpec = contract?.conditioning;
  const paramsSpec = contract?.parameters || [];
  const activeStrategies = useMemo(
    () => getActiveStrategies(strategies),
    [strategies]
  );
  const showConditioning = Boolean(condSpec && condSpec.kind !== "none");
  const showRetrievalStrategy = usesRetrievalStrategy(task, condSpec);
  const countBounds = getCountBounds(condSpec);
  const defaultCount = getDefaultCount(
    condSpec,
    getDefaultConditioning(contract, strategies, label).count
  );
  const currentCount = sanitizeCount(
    inputs.conditioning?.count,
    condSpec,
    defaultCount
  );
  const countPresets = useMemo(() => {
    if (!condSpec?.user_selectable_count) return [];
    const withinContract = [5, 10, 20].filter(
      (value) =>
        value >= countBounds.minUnits &&
        (countBounds.maxUnits === null || value <= countBounds.maxUnits)
    );
    return withinContract.length > 0 ? withinContract : [defaultCount];
  }, [condSpec, countBounds.minUnits, countBounds.maxUnits, defaultCount]);

  const routeValidation = useMemo(() => {
    if (!selectedModelKey || !selectedModel || !selectedModelIsCompatible) {
      return {
        valid: false,
        inputs: null,
        message: existingBindingIssue
          ? bindingIssueMessage(existingBindingIssue)
          : selectedModelKey
          ? "Pick a compatible current model before applying this route."
          : null,
      };
    }

    return normalizeInputsForContract({
      contract,
      task,
      rawInputs: inputs,
      strategies,
      label,
    });
  }, [
    selectedModelKey,
    selectedModel,
    selectedModelIsCompatible,
    existingBindingIssue,
    contract,
    task,
    inputs,
    strategies,
    label,
  ]);

  const canApplyRoute = Boolean(
    canEdit &&
      onSaveRoute &&
      selectedModelKey &&
      selectedModel &&
      selectedModelIsCompatible &&
      routeValidation.valid
  );

  const handleSelectModel = (modelKey) => {
    if (!canEdit) return;
    const model = findModelForTask(models, task, modelKey);
    if (!isModelCompatibleWithTarget(model, task, labelId)) return;

    const normalized = normalizeInputsForContract({
      contract: getEffectiveContract(model),
      task,
      rawInputs: {},
      strategies,
      label,
    });
    setSelectedModelKey(modelKey);
    setInputs(normalized.inputs);
  };

  const handleConfigureClick = (modelKey, e) => {
    if (e) e.stopPropagation();
    if (!canEdit && modelKey !== selectedModelKey) return;
    if (modelKey !== selectedModelKey) {
      handleSelectModel(modelKey);
    }
    setView("configure");
  };

  const handleApplyRoute = () => {
    if (!canApplyRoute || !selectedModel || !onSaveRoute) return;
    const normalized = normalizeInputsForContract({
      contract,
      task,
      rawInputs: inputs,
      strategies,
      label,
    });
    if (!normalized.valid) return;

    onSaveRoute(task, labelId, {
      model_registry_key: selectedModelKey,
      task,
      label_id: labelId,
      inputs: normalized.inputs,
    });
    onClose();
  };

  const handleCountChange = (value) => {
    if (!canEdit) return;
    setInputs((prev) => ({
      ...prev,
      conditioning: {
        ...prev.conditioning,
        count: sanitizeCount(value, condSpec, defaultCount),
      },
    }));
  };

  const handleUnbind = () => {
    if (!canEdit || !onUnbindRoute) return;
    onUnbindRoute(task, labelId);
    onClose();
  };

  if (!isOpen || !target) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end"
      role="dialog"
      aria-modal="true"
      aria-labelledby="matrix-drawer-title"
    >
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-xs transition-opacity"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Slide-over Panel */}
      <div
        ref={drawerRef}
        className="relative w-full max-w-md bg-p1 border-l border-ln shadow-2xl h-full flex flex-col z-10 text-xs overflow-hidden"
        data-testid="matrix-side-drawer"
      >
        {/* ================= STAGE 1: BIND A MODEL ================= */}
        {view === "select" && (
          <>
            {/* Header */}
            <div className="p-5 border-b border-ln bg-well/30 shrink-0">
              <div className="flex items-center justify-between">
                <h3
                  id="matrix-drawer-title"
                  className="text-xs font-bold uppercase tracking-wider text-t3"
                >
                  Bind a Model
                </h3>
                <button
                  type="button"
                  onClick={onClose}
                  className="p-1 rounded-lg text-t3 hover:text-t1 hover:bg-well transition"
                  aria-label="Close drawer"
                >
                  <X size={16} />
                </button>
              </div>

              {/* Breadcrumb pills */}
              <div className="flex items-center gap-2 mt-3">
                <span className="px-2 py-0.5 rounded text-[11px] font-medium text-teal-500 bg-teal-500/10 border border-teal-500/20">
                  {taskMeta.label}
                </span>
                <span className="text-t3/60">→</span>
                <div className="flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-medium text-t1 bg-well border border-ln">
                  {labelId == null ? (
                    <>
                      <Star size={11} className="text-amber-500 fill-amber-500" />
                      <span>Task default</span>
                    </>
                  ) : (
                    <>
                      <span
                        className="w-2 h-2 rounded-full ring-1 ring-black/10 dark:ring-white/10"
                        style={{ backgroundColor: resolveLabelColor(label) }}
                      />
                      <span>{label?.name || "Label"}</span>
                    </>
                  )}
                </div>
              </div>

              {/* Search Bar */}
              <div className="relative mt-4">
                <Search
                  size={14}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-t3 pointer-events-none"
                />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder={`Search ${compatibleModels.length} compatible models`}
                  className="w-full pl-9 pr-3 py-2 rounded-xl bg-well border border-ln2 text-xs text-t1 placeholder:text-t3/50 focus:outline-none focus:border-teal-500/60"
                />
              </div>

              {routeValidation.message && (
                <div
                  role="alert"
                  className="mt-3 flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-[11px] leading-relaxed text-amber-700 dark:text-amber-300"
                >
                  <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                  <span>{routeValidation.message}</span>
                </div>
              )}
            </div>

            {/* Model Cards List */}
            <div className="flex-1 overflow-y-auto p-5 space-y-3">
              {filteredModels.length === 0 ? (
                <div className="p-8 text-center text-t3 text-xs">
                  No compatible models found for this search.
                </div>
              ) : (
                filteredModels.map((m) => {
                  const isSelected = selectedModelKey === m.registry_key;
                  const isTrained = Boolean(m.trained_on_dataset || m.is_fine_tuned || m.is_trained_here);
                  const badgesList = Array.isArray(m.badges) && m.badges.length > 0
                    ? m.badges
                    : [m.latency_badge, m.model_size].filter(Boolean);

                  return (
                    <div
                      key={m.registry_key}
                      onClick={() => handleSelectModel(m.registry_key)}
                      className={`p-3.5 rounded-xl border transition cursor-pointer flex flex-col gap-2 ${
                        isSelected
                          ? "border-teal-500/80 bg-teal-500/5 shadow-xs ring-1 ring-teal-500/30"
                          : "border-ln bg-well/40 hover:bg-well/80 hover:border-ln2"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-t1 truncate">
                              {m.name || m.registry_key}
                            </span>
                            {isTrained && (
                              <span className="px-1.5 py-0.2 rounded text-[10px] font-semibold bg-teal-500/10 text-teal-500 dark:text-teal-400 border border-teal-500/30">
                                trained here
                              </span>
                            )}
                          </div>
                          <p className="text-[11px] text-t3 truncate mt-0.5">
                            {m.description || (m.label_ids?.length ? `${m.label_ids.length} class model` : "Class-agnostic")}
                          </p>
                        </div>

                        {isSelected && (
                          <div className="w-4 h-4 rounded-full bg-teal-500 text-white dark:text-slate-950 flex items-center justify-center shrink-0 mt-0.5">
                            <Check size={11} strokeWidth={3} />
                          </div>
                        )}
                      </div>

                      {/* Canonical Badges and Configure Link */}
                      <div className="flex items-center justify-between pt-1 border-t border-ln/50 text-[11px]">
                        <div className="flex items-center gap-1.5 flex-wrap">
                          {badgesList.map((badge, bIdx) => (
                            <span
                              key={bIdx}
                              className="px-1.5 py-0.2 rounded bg-well text-t3 border border-ln2 text-[10px]"
                            >
                              {badge}
                            </span>
                          ))}
                        </div>

                        <button
                          type="button"
                          onClick={(e) => handleConfigureClick(m.registry_key, e)}
                          disabled={!canEdit && !isSelected}
                          className="inline-flex items-center gap-1 text-teal-500 dark:text-teal-400 hover:underline font-medium transition"
                        >
                          <span>configure</span>
                          <ChevronRight size={13} />
                        </button>
                      </div>
                    </div>
                  );
                })
              )}

              {/* Helper Note */}
              <div className="p-3.5 rounded-xl border border-ln bg-well/30 text-t3 text-[11px] leading-relaxed">
                Leaving this cell unbound falls back to the task default. When no task default is configured, interactive tools use your personal favorite or the first compatible model.
              </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-ln bg-well/40 flex items-center justify-between gap-3 shrink-0">
              <div>
                {existingBinding ? (
                  <button
                    type="button"
                    onClick={handleUnbind}
                    disabled={!canEdit}
                    className="inline-flex items-center gap-1 text-red-500 hover:text-red-600 font-medium transition"
                  >
                    <Trash2 size={13} />
                    <span>Unbind Route</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={onClose}
                    className="text-t3 hover:text-t1 font-medium transition"
                  >
                    Cancel
                  </button>
                )}
              </div>

              <div>
                {selectedModelKey ? (
                  <button
                    type="button"
                    onClick={handleApplyRoute}
                    disabled={!canApplyRoute}
                    className="px-4 py-2 rounded-xl bg-teal-500 hover:bg-teal-400 text-white dark:text-slate-950 font-bold shadow-xs transition"
                  >
                    Save route
                  </button>
                ) : (
                  <span className="text-t3/70 text-[11px]">
                    Pick a model to set its inputs
                  </span>
                )}
              </div>
            </div>
          </>
        )}

        {/* ================= STAGE 2: MODEL INPUTS ================= */}
        {view === "configure" && selectedModel && (
          <>
            {/* Header with back arrow */}
            <div className="p-5 border-b border-ln bg-well/30 shrink-0">
              <div className="flex items-center justify-between">
                <button
                  type="button"
                  onClick={() => setView("select")}
                  className="inline-flex items-center gap-1.5 text-xs text-teal-500 dark:text-teal-400 hover:underline font-semibold transition"
                >
                  <ArrowLeft size={14} />
                  <span>models · INPUTS</span>
                </button>

                <button
                  type="button"
                  onClick={onClose}
                  className="p-1 rounded-lg text-t3 hover:text-t1 hover:bg-well transition"
                  aria-label="Close drawer"
                >
                  <X size={16} />
                </button>
              </div>

              <div className="mt-3">
                <h3 id="matrix-drawer-title" className="text-base font-bold text-t1">
                  {selectedModel.name || selectedModel.registry_key}
                </h3>
                <p className="text-[11px] text-t3 font-mono truncate mt-0.5">
                  registry://{selectedModel.registry_key}/contract
                </p>
              </div>
            </div>

            {/* Inputs & Conditioning sliders */}
            <div className="flex-1 overflow-y-auto p-5 space-y-6">
              {routeValidation.message && (
                <div
                  role="alert"
                  className="flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-[11px] leading-relaxed text-amber-700 dark:text-amber-300"
                >
                  <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                  <span>{routeValidation.message}</span>
                </div>
              )}

              {/* Conditioning */}
              {showConditioning && (
                <div className="space-y-4">
                  <h4 className="text-[10px] font-bold uppercase tracking-wider text-t3">
                    Conditioning
                  </h4>

                  {/* Context-driven exemplar source */}
                  {task === "cross-image-suggestion" || condSpec.kind === "reference_images" ? (
                    <div className="p-3 rounded-xl bg-well/60 border border-ln2 text-xs">
                      <span className="text-[11px] font-bold uppercase tracking-wider text-t3 block mb-1">
                        Exemplars
                      </span>
                      <span className="font-semibold text-t1 block">
                        Reference examples from other images
                      </span>
                      <span className="text-[11px] text-t3 mt-0.5 block">
                        Retrieves exemplar annotations from other dataset images.
                      </span>
                    </div>
                  ) : condSpec.kind === "instances" || task === "instance-suggestion" ? (
                    <div className="p-3 rounded-xl bg-well/60 border border-ln2 text-xs">
                      <span className="text-[11px] font-bold uppercase tracking-wider text-t3 block mb-1">
                        Exemplars
                      </span>
                      <span className="font-semibold text-t1 block">
                        Selected objects in this image
                      </span>
                      <span className="text-[11px] text-t3 mt-0.5 block">
                        Uses contours drawn or selected on the active image canvas.
                      </span>
                    </div>
                  ) : null}

                  {condSpec.kind === "concept_text" && (
                    <div>
                      <label
                        htmlFor="matrix-drawer-concept-text"
                        className="block text-xs font-medium text-t2 mb-1.5"
                      >
                        Prompt concept
                      </label>
                      <input
                        id="matrix-drawer-concept-text"
                        type="text"
                        value={inputs.conditioning?.concept_text ?? label?.name ?? ""}
                        onChange={(e) =>
                          canEdit &&
                          setInputs((prev) => ({
                            ...prev,
                            conditioning: {
                              ...prev.conditioning,
                              concept_text: e.target.value,
                            },
                          }))
                        }
                        disabled={!canEdit}
                        aria-label={`Prompt concept for ${label?.name || "label"}`}
                        className="w-full px-3 py-2 rounded-xl bg-well border border-ln2 text-xs text-t1 focus:outline-none focus:border-teal-500/60 disabled:opacity-50 disabled:cursor-not-allowed"
                      />
                    </div>
                  )}

                  {/* Count presets (only when declared user_selectable_count is true) */}
                  {condSpec.user_selectable_count && (
                    <div>
                      <label className="block text-xs font-medium text-t2 mb-1.5">
                        {condSpec.unit === "instance" ? "Instances count" : "Exemplars count"}
                      </label>
                      <div className="flex items-center gap-2">
                        {countPresets.map((countVal) => {
                          const isActive =
                            currentCount === sanitizeCount(countVal, condSpec, defaultCount);

                          return (
                            <button
                              key={countVal}
                              type="button"
                              onClick={() => handleCountChange(countVal)}
                              disabled={!canEdit}
                              className={`flex-1 py-2 rounded-xl border font-mono text-xs font-semibold transition ${
                                isActive
                                  ? "border-teal-500 bg-teal-500/10 text-teal-600 dark:text-teal-300"
                                  : "border-ln bg-well text-t3 hover:text-t1 hover:border-ln2"
                              }`}
                            >
                              {countVal}
                            </button>
                          );
                        })}
                      </div>
                      <input
                        id="matrix-drawer-count"
                        type="number"
                        min={countBounds.minUnits}
                        max={countBounds.maxUnits === null ? undefined : countBounds.maxUnits}
                        step={1}
                        value={currentCount}
                        onChange={(e) => handleCountChange(e.target.value)}
                        disabled={!canEdit}
                        aria-label={
                          condSpec.unit === "instance"
                            ? "Instances count"
                            : "Exemplars count"
                        }
                        className="mt-2 w-full px-3 py-2 rounded-xl bg-well border border-ln2 text-xs font-mono text-t1 focus:outline-none focus:border-teal-500/60 disabled:opacity-50 disabled:cursor-not-allowed"
                      />
                    </div>
                  )}

                  {/* Strategy selection for retrieval conditioning — restricted to cross-image suggestion */}
                  {showRetrievalStrategy && strategies.length > 0 && (
                      <div>
                        <label
                          htmlFor="matrix-drawer-retrieval-strategy"
                          className="block text-xs font-medium text-t2 mb-1.5"
                        >
                          Retrieval strategy
                        </label>
                        <select
                          id="matrix-drawer-retrieval-strategy"
                          value={inputs.conditioning?.strategy ?? ""}
                          onChange={(e) =>
                            canEdit &&
                            setInputs((prev) => ({
                              ...prev,
                              conditioning: { ...prev.conditioning, strategy: e.target.value },
                            }))
                          }
                          disabled={!canEdit || activeStrategies.length === 0}
                          className="w-full px-3 py-2 rounded-xl bg-well border border-ln2 text-xs text-t1 focus:outline-none focus:border-teal-500/60"
                        >
                          {strategies.map((strat) => (
                            <option key={strat.key} value={strat.key} disabled={!strat.available}>
                              {strat.label || strat.key} {!strat.available ? "(unavailable)" : ""}
                            </option>
                          ))}
                        </select>
                      </div>
                    )}
                </div>
              )}

              {/* Parameters */}
              <div className="space-y-4">
                <h4 className="text-[10px] font-bold uppercase tracking-wider text-t3">
                  Parameters
                </h4>

                <fieldset
                  disabled={!canEdit}
                  className="space-y-4 border-0 p-0 m-0 min-w-0"
                >
                  {paramsSpec.length === 0 ? (
                    <div className="p-4 rounded-xl border border-dashed border-ln bg-well/30 text-center text-t3 text-xs">
                      This model has no configurable hyper-parameters.
                    </div>
                  ) : (
                    paramsSpec.map((param) => {
                      const currentVal =
                        inputs.parameters?.[param.key] ?? param.default_value;

                      return (
                        <div
                          key={param.key}
                          className="p-3.5 rounded-xl bg-well/30 border border-ln"
                        >
                          <DynamicHyperParameter
                            param={param}
                            idPrefix="matrix-drawer"
                            value={currentVal}
                            onChange={(key, val) => {
                              if (!canEdit) return;
                              setInputs((prev) => ({
                                ...prev,
                                parameters: {
                                  ...prev.parameters,
                                  [key]: val,
                                },
                              }));
                            }}
                            disabled={!canEdit}
                          />
                        </div>
                      );
                    })
                  )}
                </fieldset>
              </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-ln bg-well/40 flex items-center justify-between gap-3 shrink-0">
              <button
                type="button"
                onClick={() => setView("select")}
                className="px-4 py-2 rounded-xl border border-ln bg-well text-t1 hover:bg-well/80 transition font-medium text-xs"
              >
                Back
              </button>

              <button
                type="button"
                onClick={handleApplyRoute}
                disabled={!canApplyRoute}
                className="inline-flex items-center gap-1.5 px-5 py-2 rounded-xl bg-teal-500 hover:bg-teal-400 text-white dark:text-slate-950 font-bold shadow-xs transition"
              >
                <span>Apply route</span>
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
